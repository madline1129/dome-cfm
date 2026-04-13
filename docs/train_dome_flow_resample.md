# 只训练 DOME 并开启 Resample

这个流程用于已经准备好 DOME 项目页所需数据和 checkpoint 的情况：OccVAE 不再训练，只从 checkpoint 加载并冻结；训练阶段只更新 DOME / world model。项目页说明 DOME 由 Occ-VAE 和 DOME 两部分组成，并提供了 nuScenes 数据、OccVAE checkpoint、DOME checkpoint 与 resample 相关说明：https://gusongen.github.io/DOME/

## 目录约定

默认路径如下，路径不一致时用环境变量覆盖：

```bash
data/nuscenes/
data/nuscenes_infos_train_temporal_v3_scene.pkl
data/nuscenes_infos_val_temporal_v3_scene.pkl
ckpts/occvae_latest.pth
data/resampled_occ/
```

## 1. 检查当前配置

```bash
python tools/check_dome_resample_setup.py
```

如果提示缺少 `data/resampled_occ`，先执行下一步。

## 2. 生成 resample 数据

```bash
bash tools/prepare_resample_data.sh
```

常用覆盖方式：

```bash
DATA_PATH=/path/to/nuscenes \
IMAGESET=/path/to/nuscenes_infos_train_temporal_v3_scene.pkl \
DST=/path/to/resampled_occ \
bash tools/prepare_resample_data.sh
```

多进程或多机切分时，可以分别设置 `RANK_ID` 和 `N_RANK`：

```bash
RANK_ID=0 N_RANK=4 bash tools/prepare_resample_data.sh
RANK_ID=1 N_RANK=4 bash tools/prepare_resample_data.sh
RANK_ID=2 N_RANK=4 bash tools/prepare_resample_data.sh
RANK_ID=3 N_RANK=4 bash tools/prepare_resample_data.sh
```

## 3. 只训练 DOME

```bash
bash tools/train_dome_flow_resample.sh
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/train_dome_flow_resample.sh
```

指定 OccVAE checkpoint 或输出目录：

```bash
VAE_CKPT=/path/to/occvae_latest.pth \
WORK_DIR=./work_dir/dome_flow_resample \
bash tools/train_dome_flow_resample.sh
```

从某个 DOME checkpoint 继续：

```bash
RESUME_FROM=./work_dir/dome_flow_resample/epoch_100.pth \
bash tools/train_dome_flow_resample.sh
```

从已有 DOME 权重微调：

```bash
LOAD_FROM=ckpts/dome_latest.pth \
bash tools/train_dome_flow_resample.sh
```

## 4. 评估

```bash
bash tools/eval_dome_flow_resample.sh
```

指定 checkpoint：

```bash
DOME_CKPT=./work_dir/dome_flow_resample/epoch_200.pth \
bash tools/eval_dome_flow_resample.sh
```

## 5. 可视化查看效果

```bash
bash tools/vis_dome_flow_resample.sh
```

指定场景、采样步数：

```bash
SCENE_IDX="6 7 16" NUM_SAMPLING_STEPS=30 \
bash tools/vis_dome_flow_resample.sh
```

可视化结果会写到 `WORK_DIR/DIR_NAME_时间戳/` 下。

## 关键点

`tools/train_diffusion.py` 中的 OccVAE 会执行 `vae.requires_grad_(False)`，训练 loss 只回传到 DOME / world model。`config/train_dome_flow_resample.py` 使用 `sample_method='flow'`，因此训练和采样都会走当前项目里接入的 flow matching 逻辑。
