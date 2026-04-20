#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-./config/train_dome_joint_flow_v3.py}"
WORK_DIR="${WORK_DIR:-./work_dir/dome_joint_flow_v3}"
VAE_CKPT="${VAE_CKPT:-ckpts/occvae_latest.pth}"
LOAD_FROM="${LOAD_FROM:-}"
RESUME_FROM="${RESUME_FROM:-}"
TB_DIR="${TB_DIR:-}"
SEED="${SEED:-42}"
EMA="${EMA:-True}"

echo "[任务] Non-resample DOMEv3 joint flow: occupancy latent + trajectory latent tokens"
echo "[配置] $CFG"
echo "[输出] $WORK_DIR"
echo "[OccVAE] $VAE_CKPT"

if [[ ! -f "$VAE_CKPT" ]]; then
  echo "[错误] 找不到 OccVAE checkpoint: $VAE_CKPT"
  echo "       可以用 VAE_CKPT=/path/to/occvae_latest.pth 覆盖。"
  exit 1
fi

if [[ ! -d "data/nuscenes" ]]; then
  echo "[错误] 找不到 data/nuscenes"
  echo "       可以用软链接或修改 config/train_dome_joint_flow_v3.py 里的 data_path。"
  exit 1
fi

if [[ ! -f "data/nuscenes_infos_train_temporal_v3_scene.pkl" ]]; then
  echo "[错误] 找不到 data/nuscenes_infos_train_temporal_v3_scene.pkl"
  echo "       请确认 nuScenes temporal infos 已准备好。"
  exit 1
fi

cmd=(
  "$PYTHON_BIN" tools/train_joint_flow.py
  --py-config "$CFG"
  --work-dir "$WORK_DIR"
  --vae_load_from "$VAE_CKPT"
  --seed "$SEED"
  --ema "$EMA"
)

if [[ -n "$LOAD_FROM" ]]; then
  cmd+=(--load_from "$LOAD_FROM")
fi

if [[ -n "$RESUME_FROM" ]]; then
  cmd+=(--resume-from "$RESUME_FROM")
fi

if [[ -n "$TB_DIR" ]]; then
  cmd+=(--tb-dir "$TB_DIR")
fi

echo "[运行] ${cmd[*]}"
"${cmd[@]}"
