#!/usr/bin/env bash
set -euo pipefail

# Edit these four variables before running.
GPU=0
BATCH_SIZE=2
LOAD_FROM=/path/to/occ_only_checkpoint.pth
VAE_CKPT=/path/to/occvae_latest.pth

# Optional overrides.
WORK_DIR=./work_dir/dome-cfmv6
CFG=./config/train_dome_cfmv6.py
RESUME_FROM=
TB_DIR=

GPU="$GPU" \
BATCH_SIZE="$BATCH_SIZE" \
LOAD_FROM="$LOAD_FROM" \
VAE_CKPT="$VAE_CKPT" \
WORK_DIR="$WORK_DIR" \
CFG="$CFG" \
RESUME_FROM="$RESUME_FROM" \
TB_DIR="$TB_DIR" \
bash tools/train_dome_cfmv6_oneclick.sh
