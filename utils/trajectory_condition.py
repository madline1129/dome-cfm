import numpy as np
import torch


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pad_or_trim(sequence, length):
    if sequence.shape[0] >= length:
        return sequence[:length]
    if sequence.shape[0] == 0:
        pad = np.zeros((length, *sequence.shape[1:]), dtype=sequence.dtype)
        return pad
    pad = np.repeat(sequence[-1:], length - sequence.shape[0], axis=0)
    return np.concatenate([sequence, pad], axis=0)


def extract_trajectory_from_metas(
    metas,
    *,
    key="rel_poses",
    start_index=4,
    traj_len=6,
    traj_dim=2,
    device=None,
    dtype=torch.float32,
):
    """Extract future trajectory targets from dataloader metas.

    The current datasets store ego motion as per-frame relative poses. For the
    first joint-flow version we use a fixed future slice, usually frames 4:10.
    """
    trajectories = []
    for meta in metas:
        if key not in meta:
            raise KeyError(f"meta does not contain trajectory key {key!r}")
        traj = _to_numpy(meta[key]).astype(np.float32)
        if traj.ndim == 1:
            traj = traj[:, None]
        traj = traj[start_index : start_index + traj_len, :traj_dim]
        traj = _pad_or_trim(traj, traj_len)
        trajectories.append(traj)
    return torch.as_tensor(np.stack(trajectories), device=device, dtype=dtype)


def _command_from_meta_value(value, num_modes):
    arr = _to_numpy(value)
    if arr.ndim == 0:
        return int(arr.item())

    first = arr[0]
    first = np.asarray(first)
    if first.ndim > 0 and first.shape[-1] == num_modes:
        return int(first.argmax())
    if arr.ndim == 1 and arr.shape[0] == num_modes:
        return int(arr.argmax())
    return int(np.asarray(first).reshape(-1)[0])


def extract_commands_from_metas(
    metas,
    *,
    traj=None,
    num_modes=3,
    device=None,
    fallback_lateral_index=0,
    fallback_threshold=0.5,
):
    """Extract command ids.

    Preferred sources are explicit command-like fields. If resampled data only
    has poses, fall back to a coarse turn estimate from the future displacement:
    0=right, 1=left, 2=straight, matching the existing dataset comment.
    """
    commands = []
    for idx, meta in enumerate(metas):
        command = None
        for key in ("command", "cmd", "traj_mode", "gt_mode"):
            if key in meta:
                command = _command_from_meta_value(meta[key], num_modes)
                break

        if command is None:
            if traj is None:
                raise KeyError(
                    "meta does not contain command/cmd/traj_mode/gt_mode and no "
                    "trajectory fallback was provided"
                )
            traj_i = traj[idx].detach()
            lateral = traj_i[:, fallback_lateral_index].sum().item()
            if lateral < -fallback_threshold:
                command = 0
            elif lateral > fallback_threshold:
                command = 1
            else:
                command = 2

        commands.append(max(0, min(int(command), num_modes - 1)))
    return torch.as_tensor(commands, device=device, dtype=torch.long)


def compute_plan_metrics(pred_traj, target_traj, prefix="plan"):
    """Compute OccWorld-style trajectory L2 planning metrics.

    Returns per-sample tensors so callers can average locally or reduce across
    distributed workers.
    """
    diff = pred_traj - target_traj
    l2 = torch.linalg.norm(diff, dim=-1)

    metrics = {
        f"{prefix}_ade": l2.mean(dim=1),
        f"{prefix}_fde": l2[:, -1],
        f"{prefix}_mse": diff.pow(2).mean(dim=(1, 2)),
    }
    for step_idx in range(l2.shape[1]):
        metrics[f"{prefix}_l2_step_{step_idx + 1}"] = l2[:, step_idx]
    return metrics
