#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT=25109
export CUDA_VISIBLE_DEVICES=1,3,5,6

CFG="./config/train_dome.py"
WORK_DIR="./work_dir/dome_flow_no_resample"
VAE_CKPT="ckpts/occvae_latest.pth"
TB_DIR="${WORK_DIR}/tb_log"
RESUME_CKPT="${WORK_DIR}/latest.pth"

CMD=(
  python tools/train_diffusion.py
  --py-config "${CFG}"
  --work-dir "${WORK_DIR}"
  --vae_load_from "${VAE_CKPT}"
  --seed 42
  --ema True
  --tb-dir "${TB_DIR}"
)

if [[ -f "${RESUME_CKPT}" ]]; then
  echo "[Resume] Found checkpoint: ${RESUME_CKPT}"
  CMD+=(--resume-from "${RESUME_CKPT}")
else
  echo "[Train] No latest checkpoint found. Training from scratch."
fi

echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[PORT] MASTER_PORT=${MASTER_PORT}"
echo "[CMD] ${CMD[*]}"

"${CMD[@]}"

tensorboard --logdir /mnt/data2/whz/dome-cfm/work_dir/dome_flow_no_resample/tb_log --port 6005 --host 0.0.0.0