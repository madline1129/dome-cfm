_base_ = ['./train_dome_joint_flow.py']

work_dir = './work_dir/dome-cfmv6'

find_unused_parameters = True
p_use_pose_condition = 1.0
load_from = ''

train_loader = dict(batch_size=2, num_workers=1, shuffle=True)

sample = dict(
    sample_method='joint_flow',
    traj_key='rel_poses',
    traj_start_index=4,
    traj_len=6,
    traj_dim=2,
    traj_loss_weight=0.05,
    use_hist_traj_condition=True,
    hist_traj_key='rel_poses',
    hist_traj_start_index=0,
    hist_traj_len=4,
    hist_condition_drop_prob=0.1,
)

model = dict(
    world_model=dict(
        type='JointDomeCFMV6',
        use_token_planning_head=True,
        hist_frame_start=0,
        hist_len=4,
        traj_frame_start=4,
        use_hist_cfg=True,
        trajectory_encoder=dict(
            type='PoseEncoder_fourier',
            in_channels=2,
            out_channels=768,
            num_layers=2,
            num_modes=3,
            num_fut_ts=1,
            do_proj=False,
            max_length=6,
        ),
    ),
)

finetune = dict(
    enabled=True,
    require_load_from=True,
    freeze_epochs=5,
    freeze_prefixes=['x_embedder', 't_embedder', 'blocks', 'final_layer'],
    always_freeze_prefixes=['pos_embed', 'temp_embed'],
    backbone_prefixes=['x_embedder', 't_embedder', 'blocks', 'final_layer', 'pos_embed', 'temp_embed'],
    backbone_lr_mult=0.1,
    new_module_lr_mult=1.0,
)
