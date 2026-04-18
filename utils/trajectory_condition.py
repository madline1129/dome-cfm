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


def _truthy_flag(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return bool(value.item())
        return True
    arr = np.asarray(value)
    if arr.size == 1:
        return bool(arr.item())
    return True


def _has_planning_ann(meta):
    return all(key in meta for key in ("gt_bboxes_3d", "attr_labels", "fut_valid_flag"))


def compute_occworld_plan_metrics(
    pred_ego_fut_trajs,
    gt_ego_fut_trajs,
    metas=None,
    planning_metric=None,
    future_second=3,
):
    """Compute planning metrics with the same STP3/OccWorld convention.

    The input trajectories are treated as per-step deltas, then converted to
    cumulative ego future trajectories before evaluating 1s/2s/3s horizons.
    """
    from utils.metric_stp3 import PlanningMetric

    if planning_metric is None:
        planning_metric = PlanningMetric()

    device = pred_ego_fut_trajs.device
    dtype = pred_ego_fut_trajs.dtype
    pred_cum = torch.cumsum(pred_ego_fut_trajs[..., :2], dim=1).detach().cpu()
    gt_cum = torch.cumsum(gt_ego_fut_trajs[..., :2], dim=1).detach().cpu()
    batch, num_frames, _ = pred_cum.shape

    metrics = {}
    for i in range(future_second):
        horizon = i + 1
        cur_time = horizon * 2
        metrics[f"plan_L2_{horizon}s"] = []
        metrics[f"plan_L2_{horizon}s_single"] = []
        metrics[f"plan_obj_col_{horizon}s"] = []
        metrics[f"plan_obj_box_col_{horizon}s"] = []
        metrics[f"plan_obj_col_{horizon}s_single"] = []
        metrics[f"plan_obj_box_col_{horizon}s_single"] = []

        for batch_idx in range(batch):
            meta = metas[batch_idx] if metas is not None else {}
            fut_valid_flag = _truthy_flag(meta.get("fut_valid_flag", True))
            if not fut_valid_flag or cur_time > num_frames:
                for key in (
                    f"plan_L2_{horizon}s",
                    f"plan_L2_{horizon}s_single",
                    f"plan_obj_col_{horizon}s",
                    f"plan_obj_box_col_{horizon}s",
                    f"plan_obj_col_{horizon}s_single",
                    f"plan_obj_box_col_{horizon}s_single",
                ):
                    metrics[key].append(0.0)
                continue

            pred = pred_cum[batch_idx]
            gt = gt_cum[batch_idx]
            metrics[f"plan_L2_{horizon}s"].append(
                planning_metric.compute_L2(pred[:cur_time], gt[:cur_time])
            )
            metrics[f"plan_L2_{horizon}s_single"].append(
                planning_metric.compute_L2(pred[cur_time - 1 : cur_time], gt[cur_time - 1 : cur_time])
            )

            if _has_planning_ann(meta):
                gt_agent_feats = torch.as_tensor(meta["attr_labels"])
                segmentation, pedestrian = planning_metric.get_label(
                    meta["gt_bboxes_3d"], gt_agent_feats[None]
                )
                occupancy = torch.logical_or(segmentation, pedestrian)
                obj_coll, obj_box_coll = planning_metric.evaluate_coll(
                    pred[None, :cur_time],
                    gt[None, :cur_time],
                    occupancy,
                )
                obj_coll_single, obj_box_coll_single = planning_metric.evaluate_coll(
                    pred[None, cur_time - 1 : cur_time],
                    gt[None, cur_time - 1 : cur_time],
                    occupancy[:, cur_time - 1 : cur_time],
                )
                metrics[f"plan_obj_col_{horizon}s"].append(obj_coll.mean().item())
                metrics[f"plan_obj_box_col_{horizon}s"].append(obj_box_coll.mean().item())
                metrics[f"plan_obj_col_{horizon}s_single"].append(obj_coll_single.item())
                metrics[f"plan_obj_box_col_{horizon}s_single"].append(obj_box_coll_single.item())

    return {
        key: torch.as_tensor(value, device=device, dtype=dtype)
        for key, value in metrics.items()
        if value
    }
