#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-./config/train_dome.py}"
WORK_DIR="${WORK_DIR:-/mnt/data2/whz/dome-cfm/work_dir/dome_flow_no_resample}"
VAE_CKPT="${VAE_CKPT:-ckpts/occvae_latest.pth}"
DOME_CKPT="${DOME_CKPT:-}"

DIR_NAME="${DIR_NAME:-vis_flow_no_resample}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-20}"
SCENE_IDX="${SCENE_IDX:-0}"
ROLLING_SAMPLING_N="${ROLLING_SAMPLING_N:-1}"
N_CONDS_ROLL="${N_CONDS_ROLL:-}"
RETURN_LEN="${RETURN_LEN:-}"
END_FRAME="${END_FRAME:-}"
TEST_INDEX_OFFSET="${TEST_INDEX_OFFSET:-0}"
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
  echo "[错误] 找不到 no-resample DOME checkpoint。"
  echo "       当前 WORK_DIR: $WORK_DIR"
  echo "       可以用 DOME_CKPT=/path/to/checkpoint.pth 覆盖。"
  exit 1
fi

if [[ ! -f "$VAE_CKPT" ]]; then
  echo "[错误] 找不到 OccVAE checkpoint: $VAE_CKPT"
  echo "       可以用 VAE_CKPT=/path/to/occvae_latest.pth 覆盖。"
  exit 1
fi

cmd=(
  "$PYTHON_BIN" tools/visualize_demo.py
  --py-config "$CFG"
  --work-dir "$WORK_DIR"
  --resume-from "$DOME_CKPT"
  --vae-resume-from "$VAE_CKPT"
  --dir-name "$DIR_NAME"
  --num_sampling_steps "$NUM_SAMPLING_STEPS"
  --rolling_sampling_n "$ROLLING_SAMPLING_N"
  --test_index_offset "$TEST_INDEX_OFFSET"
  --seed "$SEED"
  --scene-idx
)

# SCENE_IDX 支持空格分隔，例如: SCENE_IDX="0 6 7"
for scene_id in $SCENE_IDX; do
  cmd+=("$scene_id")
done

if [[ -n "$N_CONDS_ROLL" ]]; then
  cmd+=(--n_conds_roll "$N_CONDS_ROLL")
fi

if [[ -n "$RETURN_LEN" ]]; then
  cmd+=(--return_len "$RETURN_LEN")
fi

if [[ -n "$END_FRAME" ]]; then
  cmd+=(--end_frame "$END_FRAME")
fi

echo "[模式] no-resample flow inference"
echo "[配置] $CFG"
echo "[WORK_DIR] $WORK_DIR"
echo "[DOME] $DOME_CKPT"
echo "[VAE] $VAE_CKPT"
echo "[输出前缀] $WORK_DIR/${DIR_NAME}_<timestamp>"
echo "[场景] $SCENE_IDX"
echo "[运行] ${cmd[*]}"
"${cmd[@]}"
