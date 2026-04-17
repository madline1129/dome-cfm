_base_ = ['./train_dome_joint_flow_resample.py']

# Non-resample joint flow training:
# - reads original nuScenes occupancy labels from data/nuscenes/gts
# - uses gt_mode/traj_mode from nuScenes infos as navigation command
# - jointly velocity-matches occupancy latent and future trajectory

work_dir = './work_dir/dome_joint_flow'

train_dataset_config = dict(
    _delete_=True,
    type='nuScenesSceneDatasetLidar',
    data_path='data/nuscenes/',
    imageset='data/nuscenes_infos_train_temporal_v3_scene.pkl',
    offset=0,
    return_len=11,
    times=5,
    test_mode=False,
)

val_dataset_config = dict(
    _delete_=True,
    type='nuScenesSceneDatasetLidar',
    data_path='data/nuscenes/',
    imageset='data/nuscenes_infos_val_temporal_v3_scene.pkl',
    offset=0,
    return_len=10,
    times=1,
    test_mode=True,
    new_rel_pose=True,
)
