import argparse
import torch

import train_diffusion as base_train
from diffusion import create_diffusion, create_flow_matching, create_joint_flow_matching


def build_generation_process(cfg, timestep_respacing):
    sample_method = cfg.sample.get("sample_method", "ddpm")
    if sample_method == "joint_flow":
        return create_joint_flow_matching(
            num_sampling_steps=cfg.sample.get("num_sampling_steps", 20),
            sigma=cfg.sample.get("flow_sigma", 0.0),
            replace_cond_frames=cfg.replace_cond_frames,
            cond_frames_choices=cfg.cond_frames_choices,
            model_time_scale=cfg.sample.get("model_time_scale", 1000.0),
            traj_key=cfg.sample.get("traj_key", "rel_poses"),
            traj_start_index=cfg.sample.get("traj_start_index", cfg.sample.get("n_conds", 4)),
            traj_len=cfg.sample.get("traj_len", 6),
            traj_dim=cfg.sample.get("traj_dim", 2),
            traj_loss_weight=cfg.sample.get("traj_loss_weight", 10.0),
            use_hist_traj_condition=cfg.sample.get("use_hist_traj_condition", False),
            hist_traj_key=cfg.sample.get("hist_traj_key", cfg.sample.get("traj_key", "rel_poses")),
            hist_traj_start_index=cfg.sample.get("hist_traj_start_index", 0),
            hist_traj_len=cfg.sample.get("hist_traj_len", cfg.sample.get("n_conds", 0)),
            hist_condition_drop_prob=cfg.sample.get("hist_condition_drop_prob", 0.0),
            num_command_modes=cfg.sample.get("num_command_modes", 3),
            command_lateral_index=cfg.sample.get("command_lateral_index", 0),
            command_fallback_threshold=cfg.sample.get("command_fallback_threshold", 0.5),
        )
    if sample_method == "flow":
        return create_flow_matching(
            num_sampling_steps=cfg.sample.get("num_sampling_steps", 20),
            sigma=cfg.sample.get("flow_sigma", 0.0),
            replace_cond_frames=cfg.replace_cond_frames,
            cond_frames_choices=cfg.cond_frames_choices,
            model_time_scale=cfg.sample.get("model_time_scale", 1000.0),
        )
    return create_diffusion(
        timestep_respacing=timestep_respacing,
        beta_start=cfg.schedule.beta_start,
        beta_end=cfg.schedule.beta_end,
        replace_cond_frames=cfg.replace_cond_frames,
        cond_frames_choices=cfg.cond_frames_choices,
        predict_xstart=cfg.schedule.get("predict_xstart", False),
    )


if __name__ == "__main__":
    base_train.build_generation_process = build_generation_process

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--py-config", default="config/train_dome_cfmv6.py")
    parser.add_argument("--work-dir", type=str, default="./work_dir/dome-cfmv6")
    parser.add_argument("--tb-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--iter-resume", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--vae_load_from", type=str, default=None)
    parser.add_argument("--ema", type=bool, default=True)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(base_train.main, args=(args,), nprocs=args.gpus)
    else:
        base_train.main(0, args)
