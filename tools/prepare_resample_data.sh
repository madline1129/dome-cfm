#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/nuscenes}"
IMAGESET="${IMAGESET:-$ROOT_DIR/data/nuscenes_infos_train_temporal_v3_scene.pkl}"
DST="${DST:-$ROOT_DIR/data/resampled_occ}"
INPUT_DATASET="${INPUT_DATASET:-gts}"
RANK_ID="${RANK_ID:-0}"
N_RANK="${N_RANK:-1}"

echo "[检查] nuScenes 数据目录: $DATA_PATH"
echo "[检查] train imageset: $IMAGESET"
echo "[输出] resample 目录: $DST"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "[错误] 找不到 DATA_PATH=$DATA_PATH"
  exit 1
fi

if [[ ! -f "$IMAGESET" ]]; then
  echo "[错误] 找不到 IMAGESET=$IMAGESET"
  exit 1
fi

mkdir -p "$DST"

cd "$ROOT_DIR/resample"
"$PYTHON_BIN" launch.py \
  --rank "$RANK_ID" \
  --n_rank "$N_RANK" \
  --dst "$DST" \
  --imageset "$IMAGESET" \
  --input_dataset "$INPUT_DATASET" \
  --data_path "$DATA_PATH"

echo "[完成] resample 数据已写入: $DST"
