_base_ = ['./train_dome_flow_resample.py']

# First-pass joint flow setup:
# - occupancy latent and future trajectory are both noised and velocity-matched
# - clean future pose is no longer injected as a condition
# - command is embedded and injected where the old pose condition used to enter
# - trajectory velocity is predicted from an ego token as 3 command modes

work_dir = './work_dir/dome_joint_flow_resample'

p_use_pose_condition = 1.0

sample = dict(
    sample_method='joint_flow',
    num_sampling_steps=20,
    flow_sigma=0.0,
    model_time_scale=1000.0,
    n_conds=4,
    traj_key='rel_poses',
    traj_start_index=4,
    traj_len=6,
    traj_dim=2,
    traj_loss_weight=10.0,
    num_command_modes=3,
    command_lateral_index=0,
    command_fallback_threshold=0.5,
    guidance_scale=7.5,
    seed=None,
    run_time=0,
    enable_temporal_attentions=True,
    enable_vae_temporal_decoder=True,
)

val_dataset_config = dict(
    return_planning_ann=True,
)

model = dict(
    world_model=dict(
        _delete_=True,
        type='JointDome',
        attention_mode='xformers',
        class_dropout_prob=0.1,
        depth=28,
        extras=1,
        hidden_size=768,
        in_channels=64,
        input_size=25,
        learn_sigma=False,
        mlp_ratio=4.0,
        num_classes=1000,
        num_frames=11,
        num_heads=12,
        patch_size=1,
        traj_dim=2,
        traj_len=6,
        num_command_modes=3,
        command_dropout_prob=0.0,
        trajectory_encoder=dict(
            type='PoseEncoder_fourier',
            in_channels=2,
            out_channels=768,
            num_layers=2,
            num_modes=3,
            num_fut_ts=1,
            do_proj=True,
            max_length=6,
        ),
        planning_decoder=dict(
            type='PoseDecoder',
            in_channels=768,
            num_layers=2,
            num_modes=3,
            num_fut_ts=6,
            out_dim=2,
        ),
    ),
)
