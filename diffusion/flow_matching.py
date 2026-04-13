import random

import torch as th

try:
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
except ImportError:  # pragma: no cover - surfaced at construction time.
    ConditionalFlowMatcher = None

from .gaussian_diffusion import append_dims, mean_flat


class FlowMatching:
    """
    Conditional flow matching wrapper with a diffusion-like interface.

    The DOME network already accepts (x_t, t, **conditions). This class swaps
    the Gaussian diffusion objective for the torchcfm velocity objective and
    uses a simple Euler ODE sampler from t=0 noise to t=1 data.
    """

    def __init__(
        self,
        *,
        sigma=0.0,
        num_timesteps=20,
        replace_cond_frames=False,
        cond_frames_choices=None,
        model_time_scale=1000.0,
    ):
        if ConditionalFlowMatcher is None:
            raise ImportError(
                "torchcfm is required for flow matching. Install it with "
                "`pip install git+https://github.com/atong01/conditional-flow-matching.git`."
            )
        self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        self.num_timesteps = int(num_timesteps)
        self.replace_cond_frames = replace_cond_frames
        self.cond_frames_choices = cond_frames_choices
        self.model_time_scale = float(model_time_scale)

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
    def _as_velocity(model_output, channels):
        try:
            model_output = model_output.sample
        except AttributeError:
            pass
        if model_output.shape[2] == channels * 2:
            model_output, _ = th.split(model_output, channels, dim=2)
        return model_output

    def training_losses(self, model, x_start, t=None, model_kwargs=None, noise=None):
        """
        Compute the conditional flow matching velocity regression loss.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # torchcfm samples t in [0, 1], interpolated location x_t, and target
        # velocity u_t for the pair (noise -> data).
        t, x_t, u_t = self.flow_matcher.sample_location_and_conditional_flow(noise, x_start)

        cond_mask = self._sample_cond_mask(x_start)
        cond_mask_bc = append_dims(cond_mask, x_start.ndim)
        if self.replace_cond_frames:
            x_t = cond_mask_bc * x_start + (1 - cond_mask_bc) * x_t

        model_output = model(x_t, self._model_time(t), **model_kwargs)
        model_output = self._as_velocity(model_output, x_start.shape[2])
        assert model_output.shape == u_t.shape == x_start.shape

        mse = model_output - u_t
        if self.replace_cond_frames:
            mse = mse * (1 - cond_mask_bc)

        terms = {}
        terms["mse"] = mean_flat(mse**2)
        terms["loss"] = terms["mse"]
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
                velocity = model(img, self._model_time(t), **model_kwargs)
                velocity = self._as_velocity(velocity, shape[2])
                img = img + dt * velocity
                if cond_mask is not None and cond_frame is not None:
                    img = img * (1 - cond_mask) + cond_frame * cond_mask
                out = {"sample": img, "pred_xstart": img}
                yield out

    def p_sample_loop_cond_rollout(
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
        input_latents=None,
        rolling_sampling_n=1,
        n_conds=None,
        n_conds_roll=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        latents_all = []
        latents = input_latents
        initial_cond_indices = None
        end = shape[1]
        mid = 0
        if n_conds:
            initial_cond_indices = [index for index in range(n_conds)]
            mid = n_conds
        n_conds_roll = n_conds_roll if n_conds_roll is not None else n_conds
        assert n_conds_roll == n_conds, "bug fix"

        for i in range(rolling_sampling_n):
            model_kwargs["pose_st_offset"] = i * (end - mid)
            latents = self.p_sample_loop(
                model=model,
                shape=shape,
                noise=noise,
                model_kwargs=model_kwargs,
                progress=progress,
                device=device,
                initial_cond_indices=initial_cond_indices,
                initial_cond_frames=latents,
            )
            if i != 0 and n_conds_roll:
                latents_all.append(latents[:, n_conds_roll:])
            else:
                latents_all.append(latents)
            latents_ = th.zeros_like(latents)
            latents_[:, :n_conds_roll] = latents[:, -n_conds_roll:]
            latents = latents_
        return th.concat(latents_all, dim=1)
