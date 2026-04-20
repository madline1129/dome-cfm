_base_ = ['./train_dome_joint_flow_resample.py']

work_dir = './work_dir/dome_joint_flow_v3_resample'

sample = dict(
    traj_loss_weight=0.3,
)

model = dict(
    world_model=dict(
        type='JointDomeV3',
        use_token_planning_head=True,
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
