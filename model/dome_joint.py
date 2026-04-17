import torch
import torch.nn as nn

from einops import rearrange, repeat
from mmengine.registry import MODELS

from .dome import Dome
from utils.trajectory_condition import extract_commands_from_metas


@MODELS.register_module()
class JointDome(Dome):
    """DOME variant for joint occupancy and trajectory flow matching.

    Occupancy still uses the original DOME patch/token path. A noised future
    trajectory is encoded into one ego token, injected into spatial attention,
    and decoded with a multi-mode planning head.
    """

    def __init__(
        self,
        *args,
        traj_dim=2,
        traj_len=6,
        num_command_modes=3,
        trajectory_encoder=None,
        planning_decoder=None,
        command_dropout_prob=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.traj_dim = int(traj_dim)
        self.traj_len = int(traj_len)
        self.num_command_modes = int(num_command_modes)
        self.command_dropout_prob = float(command_dropout_prob)

        if trajectory_encoder is None:
            self.trajectory_encoder = nn.Sequential(
                nn.Linear(self.traj_len * self.traj_dim, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.trajectory_encoder_uses_sequence = False
        else:
            self.trajectory_encoder = MODELS.build(trajectory_encoder)
            self.trajectory_encoder_uses_sequence = True

        self.command_embedder = nn.Embedding(self.num_command_modes + 1, self.hidden_size)
        self.null_command_id = self.num_command_modes
        self.ego_token_norm = nn.LayerNorm(self.hidden_size)
        if planning_decoder is None:
            planning_decoder = dict(
                type="PoseDecoder",
                in_channels=self.hidden_size,
                num_layers=2,
                num_modes=self.num_command_modes,
                num_fut_ts=self.traj_len,
                out_dim=self.traj_dim,
            )
        self.planning_decoder_norm = nn.LayerNorm(self.hidden_size)
        self.planning_decoder = MODELS.build(planning_decoder)

    def _encode_traj(self, traj_t):
        if traj_t is None:
            raise ValueError("JointDome.forward requires traj_t with shape (B, T, D).")
        if self.trajectory_encoder_uses_sequence:
            return self.trajectory_encoder(traj_t)
        return self.trajectory_encoder(rearrange(traj_t, "b t d -> b (t d)"))

    def _get_commands(self, commands, metas, batch, device):
        if commands is not None:
            commands = commands.to(device=device, dtype=torch.long)
        elif metas is not None:
            commands = extract_commands_from_metas(
                metas,
                num_modes=self.num_command_modes,
                device=device,
            )
        else:
            commands = torch.full((batch,), self.null_command_id, device=device, dtype=torch.long)

        if self.training and self.command_dropout_prob > 0:
            drop = torch.rand(batch, device=device) < self.command_dropout_prob
            commands = torch.where(
                drop,
                torch.full_like(commands, self.null_command_id),
                commands,
            )
        return commands.clamp_(0, self.null_command_id)

    @torch.cuda.amp.autocast()
    def forward(
        self,
        x,
        t,
        y=None,
        text_embedding=None,
        use_fp16=False,
        metas=None,
        pose_st_offset=0,
        traj_t=None,
        commands=None,
    ):
        if use_fp16:
            x = x.to(dtype=torch.float16)

        batches, frames, channels, high, weight = x.shape
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed

        t_emb = self.t_embedder(t, use_fp16=use_fp16)
        commands = self._get_commands(commands, metas, batches, x.device)
        command_emb = self.command_embedder(commands)
        cond_base = t_emb + command_emb

        timestep_spatial = repeat(cond_base, "n d -> (n c) d", c=frames)
        timestep_temp = repeat(cond_base, "n d -> (n c) d", c=self.pos_embed.shape[1])

        ego_token = self._encode_traj(traj_t)
        ego_token = self.ego_token_norm(ego_token + cond_base)

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]

            ego_frame_token = repeat(ego_token, "b d -> (b f) 1 d", f=frames)
            x = torch.cat([ego_frame_token, x], dim=1)
            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(spatial_block),
                x,
                timestep_spatial,
                use_reentrant=False,
            )
            ego_frame_token, x = x[:, :1], x[:, 1:]
            ego_token = rearrange(ego_frame_token, "(b f) 1 d -> b f d", b=batches).mean(dim=1)

            x = rearrange(x, "(b f) t d -> (b t) f d", b=batches)
            if i == 0:
                x = x + self.temp_embed[:, :frames]

            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(temp_block),
                x,
                timestep_temp,
                use_reentrant=False,
            )
            x = rearrange(x, "(b t) f d -> (b f) t d", b=batches)

        x = self.final_layer(x, timestep_spatial)
        occ = self.unpatchify(x)
        occ = rearrange(occ, "(b f) c h w -> b f c h w", b=batches)

        traj_modes = self.planning_decoder(self.planning_decoder_norm(ego_token))
        traj = traj_modes.gather(
            1,
            commands.clamp(max=self.num_command_modes - 1)
            .view(-1, 1, 1, 1)
            .expand(-1, 1, self.traj_len, self.traj_dim),
        ).squeeze(1)

        return {
            "occ": occ,
            "traj": traj,
            "traj_modes": traj_modes,
            "commands": commands,
        }
