_base_ = ['./train_dome_resample.py']

# 这个配置用于“只训练 DOME / world model”的 resample 实验：
# - OccVAE 只从 checkpoint 加载，并在 tools/train_diffusion.py 中 requires_grad_(False)
# - DOME 使用 flow matching 训练
# - 训练集使用 resample 生成后的 data/resampled_occ

work_dir = './work_dir/dome_flow_resample'

load_from = ''
vae_load_from = 'ckpts/occvae_latest.pth'

sample = dict(
    sample_method='flow',
    num_sampling_steps=20,
    flow_sigma=0.0,
    model_time_scale=1000.0,
    n_conds=4,
    guidance_scale=7.5,
    seed=None,
    run_time=0,
    enable_temporal_attentions=True,
    enable_vae_temporal_decoder=True,
)

train_dataset_config = dict(
    type='nuScenesSceneDatasetLidarResample',
    data_path='data/resampled_occ',
    imageset='data/nuscenes_infos_train_temporal_v3_scene.pkl',
    offset=0,
    return_len=11,
    times=1,
    raw_times=10,
    resample_times=1,
)

val_dataset_config = dict(
    type='nuScenesSceneDatasetLidar',
    data_path='data/nuscenes/',
    imageset='data/nuscenes_infos_val_temporal_v3_scene.pkl',
    offset=0,
    return_len=10,
    times=1,
    test_mode=True,
    new_rel_pose=True,
)
