#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-./config/train_dome.py}"
WORK_DIR="${WORK_DIR:-/mnt/data2/whz/dome-cfm/work_dir/dome_flow_no_resample}"
VAE_CKPT="${VAE_CKPT:-ckpts/occvae_latest.pth}"
DOME_CKPT="${DOME_CKPT:-}"
EVAL_DIR="${EVAL_DIR:-$WORK_DIR/eval_logs}"
SEED="${SEED:-42}"

if [[ -z "$DOME_CKPT" ]]; then
  DOME_CKPT="$(
    "$PYTHON_BIN" - "$WORK_DIR" <<'PY'
import glob
import os
import sys

work_dir = sys.argv[1]
patterns = ("latest.pth", "iter.pth", "epoch_*.pth", "*.pth")
ckpts = []
for pattern in patterns:
    ckpts.extend(glob.glob(os.path.join(work_dir, pattern)))
ckpts = sorted(set(ckpts), key=os.path.getmtime, reverse=True)
print(ckpts[0] if ckpts else "")
PY
  )"
fi

if [[ -z "$DOME_CKPT" || ! -f "$DOME_CKPT" ]]; then
  echo "[错误] 找不到 DOME checkpoint。"
  echo "       当前 WORK_DIR: $WORK_DIR"
  echo "       可以用 DOME_CKPT=/path/to/checkpoint.pth 覆盖。"
  exit 1
fi

if [[ ! -f "$VAE_CKPT" ]]; then
  echo "[错误] 找不到 OccVAE checkpoint: $VAE_CKPT"
  echo "       可以用 VAE_CKPT=/path/to/occvae_latest.pth 覆盖。"
  exit 1
fi

mkdir -p "$EVAL_DIR"

cmd=(
  "$PYTHON_BIN" tools/eval_metric.py
  --py-config "$CFG"
  --work-dir "$WORK_DIR"
  --dst-dir "$EVAL_DIR"
  --resume-from "$DOME_CKPT"
  --vae-resume-from "$VAE_CKPT"
  --seed "$SEED"
)

echo "[模式] no-resample IoU / mIoU evaluation"
echo "[配置] $CFG"
echo "[WORK_DIR] $WORK_DIR"
echo "[EVAL_DIR] $EVAL_DIR"
echo "[DOME] $DOME_CKPT"
echo "[VAE] $VAE_CKPT"
echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设置>}"
echo "[运行] ${cmd[*]}"
"${cmd[@]}"

latest_log="$(ls -t "$EVAL_DIR"/eval_stp3_*.log | head -n 1)"
summary_file="$EVAL_DIR/latest_iou_miou.txt"

{
  echo "checkpoint: $DOME_CKPT"
  echo "log: $latest_log"
  grep -E "Current val iou|Current val miou|avg val iou|avg val miou" "$latest_log"
} | tee "$summary_file"

echo "[完成] 汇总指标已保存到: $summary_file"
