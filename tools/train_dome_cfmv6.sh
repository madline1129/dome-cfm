#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CFG="${CFG:-./config/train_dome_cfmv6.py}"
WORK_DIR="${WORK_DIR:-./work_dir/dome-cfmv6}"
VAE_CKPT="${VAE_CKPT:-ckpts/occvae_latest.pth}"
LOAD_FROM="${LOAD_FROM:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME_FROM="${RESUME_FROM:-}"
TB_DIR="${TB_DIR:-}"
EMA="${EMA:-true}"

echo "[任务] DOME-CFMv6 joint flow: history traj condition + future traj joint flow"
echo "[配置] CFG=$CFG"
echo "[配置] WORK_DIR=$WORK_DIR"
echo "[配置] VAE_CKPT=$VAE_CKPT"
if [[ -n "$LOAD_FROM" ]]; then
  echo "[配置] LOAD_FROM=$LOAD_FROM"
fi

if [[ ! -f "$VAE_CKPT" ]]; then
  echo "[错误] 找不到 VAE checkpoint: $VAE_CKPT"
  echo "       可以用 VAE_CKPT=/path/to/occvae_latest.pth 覆盖。"
  exit 1
fi

if [[ -z "$LOAD_FROM" ]]; then
  echo "[错误] v6 finetune 必须提供 occ checkpoint: LOAD_FROM=/path/to/occ_only_checkpoint.pth"
  exit 1
fi

cmd=(
  "$PYTHON_BIN" tools/train_joint_flow_v6.py
  --py-config "$CFG"
  --work-dir "$WORK_DIR"
  --vae_load_from "$VAE_CKPT"
  --ema "$EMA"
)

if [[ -n "$RESUME_FROM" ]]; then
  cmd+=(--resume-from "$RESUME_FROM")
fi

if [[ -n "$LOAD_FROM" ]]; then
  cmd+=(--load_from "$LOAD_FROM")
fi

if [[ -n "$TB_DIR" ]]; then
  cmd+=(--tb-dir "$TB_DIR")
fi

echo "[执行] ${cmd[*]}"
"${cmd[@]}"
