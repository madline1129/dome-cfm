#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-./config/train_dome_flow_resample.py}"
WORK_DIR="${WORK_DIR:-./work_dir/dome_flow_resample}"
VAE_CKPT="${VAE_CKPT:-ckpts/occvae_latest.pth}"
DOME_CKPT="${DOME_CKPT:-$WORK_DIR/latest.pth}"
DST_DIR="${DST_DIR:-}"
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
  "$PYTHON_BIN" tools/eval_metric.py
  --py-config "$CFG"
  --work-dir "$WORK_DIR"
  --resume-from "$DOME_CKPT"
  --vae-resume-from "$VAE_CKPT"
  --seed "$SEED"
)

if [[ -n "$DST_DIR" ]]; then
  mkdir -p "$DST_DIR"
  cmd+=(--dst-dir "$DST_DIR")
fi

echo "[运行] ${cmd[*]}"
"${cmd[@]}"
