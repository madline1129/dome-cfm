_base_ = ['./train_dome.py']

# ckpts/dome_latest.pth uses the original DOME diffusion head:
# final_layer outputs 2 * in_channels = 128 channels, so learn_sigma must be
# True when building the model for evaluation.
model = dict(
    world_model=dict(
        learn_sigma=True,
    )
)

# The bundled dome_latest checkpoint is the original diffusion checkpoint, not
# the flow-matching checkpoint. Use the DDPM sampler path for metric evaluation.
sample = dict(
    sample_method='ddpm',
)
