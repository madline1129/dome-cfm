import random

import torch as th

from .gaussian_diffusion import append_dims, mean_flat
from utils.trajectory_condition import (
    compute_plan_metrics,
    extract_commands_from_metas,
    extract_trajectory_from_metas,
)


class JointTrajectoryOccupancyFlowMatching:
    """Joint flow matching for occupancy latents and future trajectories.

    This is intentionally separate from FlowMatching so the existing DOME-CFM
    path remains unchanged. The occupancy branch keeps the old diffusion-like
    interface, while trajectory targets are read from model_kwargs["metas"].
    """

    def __init__(
        self,
        *,
        sigma=0.0,
        num_timesteps=20,
        replace_cond_frames=False,
        cond_frames_choices=None,
        model_time_scale=1000.0,
        traj_key="rel_poses",
        traj_start_index=4,
        traj_len=6,
        traj_dim=2,
        traj_loss_weight=10.0,
        num_command_modes=3,
        command_lateral_index=0,
        command_fallback_threshold=0.5,
    ):
        self.sigma = float(sigma)
        self.num_timesteps = int(num_timesteps)
        self.replace_cond_frames = replace_cond_frames
        self.cond_frames_choices = cond_frames_choices
        self.model_time_scale = float(model_time_scale)
        self.traj_key = traj_key
        self.traj_start_index = int(traj_start_index)
        self.traj_len = int(traj_len)
        self.traj_dim = int(traj_dim)
        self.traj_loss_weight = float(traj_loss_weight)
        self.num_command_modes = int(num_command_modes)
        self.command_lateral_index = int(command_lateral_index)
        self.command_fallback_threshold = float(command_fallback_threshold)

    def _sample_cond_mask(self, x):
        bs, num_frames = x.shape[:2]
        cond_mask = th.zeros(bs, num_frames, device=x.device, dtype=x.dtype)
        if not self.replace_cond_frames:
            return cond_mask

        cond_frames_choices = self.cond_frames_choices or [[]]
        for each_cond_mask in cond_mask:
            assert len(cond_frames_choices[-1]) < num_frames
            weights = [2**n for n in range(len(cond_frames_choices))]
            cond_indices = random.choices(cond_frames_choices, weights=weights, k=1)[0]
            if cond_indices:
                each_cond_mask[cond_indices] = 1
        return cond_mask

    def _model_time(self, t):
        return t * self.model_time_scale

    @staticmethod
    def _linear_flow(x0, x1, t):
        t_bc = append_dims(t, x1.ndim)
        x_t = (1 - t_bc) * x0 + t_bc * x1
        u_t = x1 - x0
        return x_t, u_t

    @staticmethod
    def _split_model_output(model_output):
        if not isinstance(model_output, dict):
            raise TypeError(
                "Joint flow expects the model to return a dict with keys "
                "'occ' and 'traj' or 'traj_modes'."
            )
        occ = model_output["occ"]
        traj = model_output.get("traj")
        traj_modes = model_output.get("traj_modes")
        return occ, traj, traj_modes

    def _targets_from_kwargs(self, model_kwargs, x_start):
        metas = (model_kwargs or {}).get("metas")
        if metas is None:
            raise KeyError("Joint flow requires model_kwargs['metas'] for trajectory targets.")

        traj_start = extract_trajectory_from_metas(
            metas,
            key=self.traj_key,
            start_index=self.traj_start_index,
            traj_len=self.traj_len,
            traj_dim=self.traj_dim,
            device=x_start.device,
            dtype=x_start.dtype,
        )
        commands = extract_commands_from_metas(
            metas,
            traj=traj_start,
            num_modes=self.num_command_modes,
            device=x_start.device,
            fallback_lateral_index=self.command_lateral_index,
            fallback_threshold=self.command_fallback_threshold,
        )
        return traj_start, commands

    def training_losses(self, model, x_start, t=None, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        traj_start, commands = self._targets_from_kwargs(model_kwargs, x_start)
        traj_noise = th.randn_like(traj_start)

        t = th.rand(x_start.shape[0], device=x_start.device, dtype=x_start.dtype)
        x_t, u_occ = self._linear_flow(noise, x_start, t)
        traj_t, u_traj = self._linear_flow(traj_noise, traj_start, t)

        cond_mask = self._sample_cond_mask(x_start)
        cond_mask_bc = append_dims(cond_mask, x_start.ndim)
        if self.replace_cond_frames:
            x_t = cond_mask_bc * x_start + (1 - cond_mask_bc) * x_t

        model_output = model(
            x_t,
            self._model_time(t),
            traj_t=traj_t,
            commands=commands,
            **model_kwargs,
        )
        pred_occ, pred_traj, pred_traj_modes = self._split_model_output(model_output)
        assert pred_occ.shape == u_occ.shape == x_start.shape

        if pred_traj is None:
            if pred_traj_modes is None:
                raise KeyError("JointDome output must include 'traj' or 'traj_modes'.")
            gather_index = commands.view(-1, 1, 1, 1).expand(
                -1, 1, self.traj_len, self.traj_dim
            )
            pred_traj = pred_traj_modes.gather(1, gather_index).squeeze(1)
        assert pred_traj.shape == u_traj.shape == traj_start.shape

        occ_mse = pred_occ - u_occ
        if self.replace_cond_frames:
            occ_mse = occ_mse * (1 - cond_mask_bc)
        traj_mse = pred_traj - u_traj

        terms = {}
        terms["occ_mse"] = mean_flat(occ_mse**2)
        terms["traj_mse"] = mean_flat(traj_mse**2)
        terms["loss"] = terms["occ_mse"] + self.traj_loss_weight * terms["traj_mse"]
        with th.no_grad():
            t_bc = append_dims(t, traj_start.ndim)
            pred_traj_start = traj_t + (1 - t_bc) * pred_traj
            terms.update(compute_plan_metrics(pred_traj_start, traj_start))
        return terms

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        initial_cond_indices=None,
        initial_cond_frames=None,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            initial_cond_indices=initial_cond_indices,
            initial_cond_frames=initial_cond_frames,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop(self, *args, **kwargs):
        return self.p_sample_loop(*args, **kwargs)

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        model_kwargs=None,
        device=None,
        progress=False,
        initial_cond_indices=None,
        initial_cond_frames=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        metas = model_kwargs.get("metas")
        if metas is None:
            traj = th.randn(shape[0], self.traj_len, self.traj_dim, device=device, dtype=img.dtype)
            commands = th.full((shape[0],), 2, device=device, dtype=th.long)
        else:
            traj_start, commands = self._targets_from_kwargs(model_kwargs, img)
            traj = th.randn_like(traj_start)

        cond_mask = None
        cond_frame = None
        if self.replace_cond_frames and initial_cond_indices is not None and initial_cond_frames is not None:
            bs, num_frames = shape[:2]
            cond_mask = th.zeros((bs, num_frames), device=device, dtype=img.dtype)
            cond_mask[:, initial_cond_indices] = 1
            cond_mask = append_dims(cond_mask, len(shape))

            assert num_frames <= initial_cond_frames.size(1), (
                f"{num_frames}==>{initial_cond_frames.size(1)}"
            )
            cond_frame = th.zeros(shape, device=device, dtype=initial_cond_frames.dtype)
            cond_frame[:, initial_cond_indices] = initial_cond_frames[:, initial_cond_indices]
            img = img * (1 - cond_mask) + cond_frame * cond_mask

        indices = range(self.num_timesteps)
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        times = th.linspace(0, 1, self.num_timesteps + 1, device=device, dtype=img.dtype)
        for i in indices:
            t = th.full((shape[0],), times[i].item(), device=device, dtype=img.dtype)
            dt = times[i + 1] - times[i]
            with th.no_grad():
                model_output = model(
                    img,
                    self._model_time(t),
                    traj_t=traj,
                    commands=commands,
                    **model_kwargs,
                )
                velocity, traj_velocity, traj_velocity_modes = self._split_model_output(model_output)
                if traj_velocity is None:
                    gather_index = commands.view(-1, 1, 1, 1).expand(
                        -1, 1, self.traj_len, self.traj_dim
                    )
                    traj_velocity = traj_velocity_modes.gather(1, gather_index).squeeze(1)
                img = img + dt * velocity
                traj = traj + dt * traj_velocity
                if cond_mask is not None and cond_frame is not None:
                    img = img * (1 - cond_mask) + cond_frame * cond_mask
                out = {"sample": img, "pred_xstart": img, "traj": traj}
                yield out
