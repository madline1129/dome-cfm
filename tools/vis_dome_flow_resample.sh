#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-./config/train_dome_flow_resample.py}"
WORK_DIR="${WORK_DIR:-./work_dir/dome_flow_resample}"
VAE_CKPT="${VAE_CKPT:-ckpts/occvae_latest.pth}"
DOME_CKPT="${DOME_CKPT:-$WORK_DIR/latest.pth}"
DIR_NAME="${DIR_NAME:-vis_flow_resample}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-20}"
SCENE_IDX="${SCENE_IDX:-6 7}"
ROLLING_SAMPLING_N="${ROLLING_SAMPLING_N:-1}"
N_CONDS_ROLL="${N_CONDS_ROLL:-}"
RETURN_LEN="${RETURN_LEN:-}"
END_FRAME="${END_FRAME:-}"
SEED="${SEED:-42}"

if [[ ! -f "$DOME_CKPT" ]]; then
  latest_ckpt="$(ls -1 "$WORK_DIR"/epoch_*.pth 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "$latest_ckpt" ]]; then
    DOME_CKPT="$latest_ckpt"
  fi
fi

if [[ ! -f "$DOME_CKPT" ]]; then
  echo "[错误] 找不到 DOME checkpoint: $DOME_CKPT"
  echo "       可以用 DOME_CKPT=/path/to/dome.pth 覆盖。"
  exit 1
fi

if [[ ! -f "$VAE_CKPT" ]]; then
  echo "[错误] 找不到 OccVAE checkpoint: $VAE_CKPT"
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
  --seed "$SEED"
  --scene-idx
)

# SCENE_IDX 支持空格分隔，例如: SCENE_IDX="6 7 16"
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

echo "[运行] ${cmd[*]}"
"${cmd[@]}"
