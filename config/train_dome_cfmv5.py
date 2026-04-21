_base_ = ['./train_dome_joint_flow.py']

work_dir = './work_dir/dome-cfmv5'

find_unused_parameters = True

train_loader = dict(batch_size=2, num_workers=1, shuffle=True)

sample = dict(
    traj_loss_weight=1.0,
)

model = dict(
    world_model=dict(
        type='JointDomeCFMV5',
        use_token_planning_head=True,
        traj_frame_start=4,
        traj_pool_type='attention',
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
