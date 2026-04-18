import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from einops import rearrange
from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
from mmengine.runner import set_random_seed
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion import create_joint_flow_matching
from utils.trajectory_condition import (
    extract_commands_from_metas,
    extract_trajectory_from_metas,
)


def find_checkpoint(work_dir):
    latest = osp.join(work_dir, "latest.pth")
    if osp.exists(latest):
        return latest

    ckpts = [
        name
        for name in os.listdir(work_dir)
        if name.endswith(".pth")
        and name.replace(".pth", "").replace("epoch_", "").isdigit()
    ]
    if not ckpts:
        return ""
    ckpts.sort(key=lambda name: int(name.replace(".pth", "").replace("epoch_", "")))
    return osp.join(work_dir, ckpts[-1])


def build_joint_flow(cfg, num_sampling_steps=None):
    if cfg.sample.get("sample_method", "ddpm") != "joint_flow":
        raise ValueError("tools/eval_joint_traj.py requires sample.sample_method='joint_flow'")

    return create_joint_flow_matching(
        num_sampling_steps=num_sampling_steps or cfg.sample.get("num_sampling_steps", 20),
        sigma=cfg.sample.get("flow_sigma", 0.0),
        replace_cond_frames=cfg.replace_cond_frames,
        cond_frames_choices=cfg.cond_frames_choices,
        model_time_scale=cfg.sample.get("model_time_scale", 1000.0),
        traj_key=cfg.sample.get("traj_key", "rel_poses"),
        traj_start_index=cfg.sample.get("traj_start_index", cfg.sample.get("n_conds", 4)),
        traj_len=cfg.sample.get("traj_len", 6),
        traj_dim=cfg.sample.get("traj_dim", 2),
        traj_loss_weight=cfg.sample.get("traj_loss_weight", 10.0),
        num_command_modes=cfg.sample.get("num_command_modes", 3),
        command_lateral_index=cfg.sample.get("command_lateral_index", 0),
        command_fallback_threshold=cfg.sample.get("command_fallback_threshold", 0.5),
    )


def load_state_dict(module, checkpoint, prefer_ema=True):
    if prefer_ema and "ema" in checkpoint:
        load_key = "ema"
    elif "state_dict" in checkpoint:
        load_key = "state_dict"
    else:
        load_key = None

    state_dict = checkpoint if load_key is None else checkpoint[load_key]
    return load_key or "raw", module.load_state_dict(state_dict, strict=False)


def load_vae_state_dict(vae, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    return vae.load_state_dict(state_dict, strict=False)


def encode_occupancy(vae, input_occs, scaling_factor):
    bs = input_occs.shape[0]
    encoded_latent, _ = vae.forward_encoder(input_occs)
    encoded_latent, _, _ = vae.sample_z(encoded_latent)
    input_latents = encoded_latent * scaling_factor

    if input_latents.dim() == 4:
        return rearrange(input_latents, "(b f) c h w -> b f c h w", b=bs).contiguous()
    if input_latents.dim() == 5:
        return rearrange(input_latents, "b c f h w -> b f c h w", b=bs).contiguous()
    raise NotImplementedError(f"Unsupported latent shape: {tuple(input_latents.shape)}")


def sample_joint_trajectory(
    diffusion,
    model,
    noise_shape,
    model_kwargs,
    input_latents,
    n_conds,
    device,
):
    initial_cond_indices = list(range(n_conds)) if n_conds else None
    final = None
    for sample in diffusion.p_sample_loop_progressive(
        model,
        noise_shape,
        noise=None,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        initial_cond_indices=initial_cond_indices,
        initial_cond_frames=input_latents,
    ):
        final = sample
    if final is None or "traj" not in final:
        raise RuntimeError("joint_flow sampler did not return a trajectory")
    return final["traj"]


def update_command_stats(command_stats, commands, l2):
    for command_id in commands.detach().cpu().tolist():
        command_stats.setdefault(command_id, {"count": 0, "ade_sum": 0.0, "fde_sum": 0.0})

    for idx, command_id in enumerate(commands.detach().cpu().tolist()):
        command_stats[command_id]["count"] += 1
        command_stats[command_id]["ade_sum"] += l2[idx].mean().item()
        command_stats[command_id]["fde_sum"] += l2[idx, -1].item()


def main(args):
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Trajectory eval expects CUDA because the project metrics/models use CUDA tensors.")

    cfg = Config.fromfile(args.py_config)
    if args.batch_size is not None:
        cfg.val_loader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.val_loader.num_workers = args.num_workers

    os.makedirs(args.work_dir, exist_ok=True)
    log_file = osp.join(args.work_dir, f"eval_joint_traj_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = MMLogger("genocc", log_file=log_file, distributed=False)
    MMLogger._instance_dict["genocc"] = logger
    logger.info(f"Config:\n{cfg.pretty_text}")

    import model  # noqa: F401
    from dataset import get_dataloader

    model = MODELS.build(cfg.model.world_model).to(device)
    model.eval()

    vae = MODELS.build(cfg.model.vae).to(device)
    vae.requires_grad_(False)
    vae.eval()

    checkpoint_path = args.resume_from or find_checkpoint(args.work_dir)
    vae_checkpoint_path = args.vae_resume_from or cfg.get("vae_load_from", "")
    logger.info(f"resume from: {checkpoint_path}")
    logger.info(f"vae resume from: {vae_checkpoint_path}")
    assert checkpoint_path and osp.exists(checkpoint_path)
    assert vae_checkpoint_path and osp.exists(vae_checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_key, load_msg = load_state_dict(model, checkpoint, prefer_ema=not args.no_ema)
    logger.info(f"loaded model key: {load_key}")
    logger.info(str(load_msg))
    logger.info(str(load_vae_state_dict(vae, vae_checkpoint_path)))

    _, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        iter_resume=False,
    )

    diffusion = build_joint_flow(cfg, num_sampling_steps=args.num_sampling_steps)
    vae_scale_factor = 2 ** (len(cfg.model.vae.encoder_cfg.ch_mult) - 1)
    latent_resolution = cfg.model.vae.encoder_cfg.resolution // vae_scale_factor
    end_frame = cfg.get("end_frame", 10)
    n_conds = cfg.sample.get("n_conds", 0)

    total_count = 0
    ade_sum = 0.0
    fde_sum = 0.0
    mse_sum = 0.0
    per_step_l2_sum = None
    command_stats = {}

    traj_key = cfg.sample.get("traj_key", "rel_poses")
    traj_start_index = cfg.sample.get("traj_start_index", cfg.sample.get("n_conds", 4))
    traj_len = cfg.sample.get("traj_len", 6)
    traj_dim = cfg.sample.get("traj_dim", 2)

    with torch.no_grad():
        for batch_idx, (input_occs, target_occs, metas) in enumerate(tqdm(val_dataset_loader)):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            input_occs = input_occs.to(device)
            bs = input_occs.shape[0]
            input_latents = encode_occupancy(vae, input_occs, cfg.model.vae.scaling_factor)

            model_kwargs = {"metas": metas}
            noise_shape = (
                bs,
                end_frame,
                cfg.base_channel + cfg.get("len_additonal_channel", 0),
                latent_resolution,
                latent_resolution,
            )
            pred_traj = sample_joint_trajectory(
                diffusion,
                model,
                noise_shape,
                model_kwargs,
                input_latents,
                n_conds,
                device,
            )
            target_traj = extract_trajectory_from_metas(
                metas,
                key=traj_key,
                start_index=traj_start_index,
                traj_len=traj_len,
                traj_dim=traj_dim,
                device=device,
                dtype=pred_traj.dtype,
            )
            commands = extract_commands_from_metas(
                metas,
                traj=target_traj,
                num_modes=cfg.sample.get("num_command_modes", 3),
                device=device,
                fallback_lateral_index=cfg.sample.get("command_lateral_index", 0),
                fallback_threshold=cfg.sample.get("command_fallback_threshold", 0.5),
            )

            diff = pred_traj - target_traj
            l2 = torch.linalg.norm(diff, dim=-1)
            sq_error = diff.pow(2).mean(dim=(1, 2))

            total_count += bs
            ade_sum += l2.mean(dim=1).sum().item()
            fde_sum += l2[:, -1].sum().item()
            mse_sum += sq_error.sum().item()
            step_sum = l2.sum(dim=0).detach().cpu()
            per_step_l2_sum = step_sum if per_step_l2_sum is None else per_step_l2_sum + step_sum
            update_command_stats(command_stats, commands, l2)

            if batch_idx % args.print_freq == 0:
                logger.info(
                    f"[EVAL_TRAJ] Iter {batch_idx:5d}/{len(val_dataset_loader):5d}: "
                    f"ADE {ade_sum / total_count:.4f}, "
                    f"FDE {fde_sum / total_count:.4f}, "
                    f"MSE {mse_sum / total_count:.4f}"
                )

    if total_count == 0:
        raise RuntimeError("No validation samples were evaluated.")

    per_step_l2 = (per_step_l2_sum / total_count).numpy()
    logger.info(f"Trajectory samples: {total_count}")
    logger.info(f"ADE: {ade_sum / total_count:.6f}")
    logger.info(f"FDE: {fde_sum / total_count:.6f}")
    logger.info(f"MSE: {mse_sum / total_count:.6f}")
    logger.info(f"RMSE: {np.sqrt(mse_sum / total_count):.6f}")
    logger.info(f"Per-step L2: {per_step_l2.tolist()}")

    for command_id in sorted(command_stats):
        stats = command_stats[command_id]
        count = stats["count"]
        logger.info(
            f"Command {command_id}: count={count}, "
            f"ADE={stats['ade_sum'] / count:.6f}, "
            f"FDE={stats['fde_sum'] / count:.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate joint-flow trajectory metrics.")
    parser.add_argument("--py-config", default="config/train_dome_joint_flow_resample.py")
    parser.add_argument("--work-dir", type=str, default="./work_dir/dome_joint_flow_resample")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--vae-resume-from", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ema", action="store_true")
    main(parser.parse_args())
