#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
CFG="${CFG:-./config/train_dome_cfmv6.py}"
WORK_DIR="${WORK_DIR:-./work_dir/dome-cfmv6}"
LOAD_FROM="${LOAD_FROM:-}"
VAE_CKPT="${VAE_CKPT:-ckpts/occvae_latest.pth}"
RESUME_FROM="${RESUME_FROM:-}"
TB_DIR="${TB_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EMA="${EMA:-true}"
BATCH_SIZE="${BATCH_SIZE:-}"

echo "[任务] DOME-CFMv6 一键训练"
echo "[配置] GPU=$GPU"
echo "[配置] CFG=$CFG"
echo "[配置] WORK_DIR=$WORK_DIR"
echo "[配置] LOAD_FROM=$LOAD_FROM"
echo "[配置] VAE_CKPT=$VAE_CKPT"
if [[ -n "$RESUME_FROM" ]]; then
  echo "[配置] RESUME_FROM=$RESUME_FROM"
fi
if [[ -n "$TB_DIR" ]]; then
  echo "[配置] TB_DIR=$TB_DIR"
fi
if [[ -n "$BATCH_SIZE" ]]; then
  echo "[配置] BATCH_SIZE=$BATCH_SIZE"
fi

if [[ -z "$LOAD_FROM" ]]; then
  echo "[错误] 必须提供 occ checkpoint:"
  echo "       LOAD_FROM=/path/to/occ_only_checkpoint.pth bash tools/train_dome_cfmv6_oneclick.sh"
  exit 1
fi

if [[ ! -f "$LOAD_FROM" ]]; then
  echo "[错误] 找不到 occ checkpoint: $LOAD_FROM"
  exit 1
fi

if [[ ! -f "$VAE_CKPT" ]]; then
  echo "[错误] 找不到 VAE checkpoint: $VAE_CKPT"
  exit 1
fi

cmd=(
  "$PYTHON_BIN" tools/train_joint_flow_v6.py
  --py-config "$CFG"
  --work-dir "$WORK_DIR"
  --load_from "$LOAD_FROM"
  --vae_load_from "$VAE_CKPT"
  --ema "$EMA"
)

if [[ -n "$RESUME_FROM" ]]; then
  cmd+=(--resume-from "$RESUME_FROM")
fi

if [[ -n "$TB_DIR" ]]; then
  cmd+=(--tb-dir "$TB_DIR")
fi

if [[ -n "$BATCH_SIZE" ]]; then
  cmd+=(--batch-size "$BATCH_SIZE")
fi

echo "[执行] CUDA_VISIBLE_DEVICES=$GPU ${cmd[*]}"
CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
