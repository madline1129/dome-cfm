import torch
import torch.nn as nn

from einops import rearrange, repeat
from mmengine.registry import MODELS

from .dome import Dome, get_1d_sincos_temp_embed
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
            raise ValueError(
                "JointDome.forward requires traj_t with shape (B, T, D). "
                "This usually means the config is using JointDome but the "
                "generation process is not joint_flow."
            )
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


@MODELS.register_module()
class JointDomeV3(JointDome):
    """Joint DOME with separated noised latent tokens and AdaLN conditions.

    DOMEv3 keeps noised trajectory as a trajectory-token sequence. Timestep and
    command are fused only as the AdaLN condition, instead of being added into
    the noised trajectory token.
    """

    def __init__(
        self,
        *args,
        use_token_planning_head=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_token_planning_head = bool(use_token_planning_head)

        self.cond_fuser = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.traj_pos_embed = nn.Parameter(
            torch.zeros(1, self.traj_len, self.hidden_size),
            requires_grad=False,
        )
        traj_pos_embed = get_1d_sincos_temp_embed(self.hidden_size, self.traj_len)
        self.traj_pos_embed.data.copy_(torch.from_numpy(traj_pos_embed).float().unsqueeze(0))

        if self.use_token_planning_head:
            self.traj_final_layer = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.num_command_modes * self.traj_dim),
            )

    def _encode_traj_tokens(self, traj_t):
        traj_tokens = self._encode_traj(traj_t)
        if traj_tokens.dim() == 2:
            traj_tokens = traj_tokens[:, None, :].expand(-1, self.traj_len, -1)
        if traj_tokens.shape[1] != self.traj_len:
            raise ValueError(
                f"JointDomeV3 expects {self.traj_len} trajectory tokens, "
                f"got {traj_tokens.shape[1]}. Set trajectory_encoder.do_proj=False."
            )
        return traj_tokens + self.traj_pos_embed.to(device=traj_tokens.device, dtype=traj_tokens.dtype)

    def _decode_traj(self, traj_tokens, commands):
        if self.use_token_planning_head:
            traj_modes = self.traj_final_layer(traj_tokens)
            traj_modes = rearrange(
                traj_modes,
                "b f (m d) -> b m f d",
                m=self.num_command_modes,
                d=self.traj_dim,
            )
        else:
            traj_summary = traj_tokens.mean(dim=1)
            traj_modes = self.planning_decoder(self.planning_decoder_norm(traj_summary))

        traj = traj_modes.gather(
            1,
            commands.clamp(max=self.num_command_modes - 1)
            .view(-1, 1, 1, 1)
            .expand(-1, 1, self.traj_len, self.traj_dim),
        ).squeeze(1)
        return traj, traj_modes

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
        cond_base = self.cond_fuser(torch.cat([t_emb, command_emb], dim=-1))

        timestep_spatial = repeat(cond_base, "n d -> (n c) d", c=frames)
        timestep_temp = repeat(cond_base, "n d -> (n c) d", c=self.pos_embed.shape[1])

        traj_tokens = self._encode_traj_tokens(traj_t)

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]

            traj_frame_tokens = repeat(traj_tokens, "b k d -> (b f) k d", f=frames)
            x = torch.cat([traj_frame_tokens, x], dim=1)
            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(spatial_block),
                x,
                timestep_spatial,
                use_reentrant=False,
            )
            traj_frame_tokens, x = x[:, : self.traj_len], x[:, self.traj_len :]
            traj_tokens = rearrange(
                traj_frame_tokens,
                "(b f) k d -> b f k d",
                b=batches,
            ).mean(dim=1)

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

        traj, traj_modes = self._decode_traj(traj_tokens, commands)

        return {
            "occ": occ,
            "traj": traj,
            "traj_modes": traj_modes,
            "commands": commands,
        }


@MODELS.register_module()
class JointDomeV4(JointDomeV3):
    """Joint DOME with unified spatial-temporal attention over occ + traj tokens.

    Exploits ``traj_len == frames`` so that each trajectory token can be assigned
    to exactly one frame.  The trajectory tokens are concatenated with the
    occupancy patch tokens *before* the transformer loop, and the loop body is
    the plain alternating spatial / temporal attention used in the base
    :class:`Dome`.  After the loop the two streams are split and sent to their
    respective decoders.
    """

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
        assert frames == self.traj_len, (
            f"JointDomeV4 requires frames == traj_len, got {frames} vs {self.traj_len}"
        )

        # --- patch embedding: [(B*F), T, D] --------------------------------
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed
        num_patches = x.shape[1]  # T

        # --- condition -------------------------------------------------------
        t_emb = self.t_embedder(t, use_fp16=use_fp16)
        commands = self._get_commands(commands, metas, batches, x.device)
        command_emb = self.command_embedder(commands)
        cond_base = self.cond_fuser(torch.cat([t_emb, command_emb], dim=-1))

        # spatial cond has (B*F) entries; temporal cond has (B*(T+1)) entries
        timestep_spatial = repeat(cond_base, "n d -> (n c) d", c=frames)
        timestep_temp = repeat(cond_base, "n d -> (n c) d", c=num_patches + 1)

        # --- encode trajectory tokens: [B, K, D] → [(B*F), 1, D] -----------
        traj_tokens = self._encode_traj_tokens(traj_t)  # [B, K=F, D]
        traj_tokens = rearrange(traj_tokens, "b f d -> (b f) 1 d")

        # --- concat traj token with occ tokens: [(B*F), T+1, D] -------------
        x = torch.cat([traj_tokens, x], dim=1)

        # --- transformer loop (same structure as Dome.forward) ---------------
        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]

            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(spatial_block),
                x,
                timestep_spatial,
                use_reentrant=False,
            )

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

        # --- split traj / occ tokens ----------------------------------------
        traj_tokens = x[:, :1, :]   # [(B*F), 1, D]
        x = x[:, 1:, :]             # [(B*F), T, D]

        # --- occ decoder -----------------------------------------------------
        x = self.final_layer(x, timestep_spatial)
        occ = self.unpatchify(x)
        occ = rearrange(occ, "(b f) c h w -> b f c h w", b=batches)

        # --- traj decoder ----------------------------------------------------
        traj_tokens = rearrange(traj_tokens, "(b f) 1 d -> b f d", b=batches)
        traj, traj_modes = self._decode_traj(traj_tokens, commands)

        return {
            "occ": occ,
            "traj": traj,
            "traj_modes": traj_modes,
            "commands": commands,
        }


@MODELS.register_module()
class JointDomeCFMV5(JointDomeV3):
    """Joint DOME-CFM v5 with full trajectory tokens kept in the backbone.

    Each occupancy frame receives the full trajectory-token sequence before the
    transformer loop. The trajectory tokens therefore participate in the same
    alternating spatial and temporal attention as occupancy tokens. After the
    loop, only the future-frame copies are pooled over the occupancy-frame
    dimension and decoded as trajectory.
    """

    def __init__(
        self,
        *args,
        traj_frame_start=4,
        traj_pool_type="attention",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.traj_frame_start = int(traj_frame_start)
        self.traj_pool_type = traj_pool_type

        if self.traj_pool_type not in ("attention", "mean"):
            raise ValueError(
                "JointDomeCFMV5 supports traj_pool_type='attention' or 'mean', "
                f"got {self.traj_pool_type!r}."
            )
        if self.traj_pool_type == "attention":
            self.traj_frame_pool = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, 1),
            )

    def _pool_future_traj_tokens(self, traj_tokens, frames):
        future_start = self.traj_frame_start
        future_end = future_start + self.traj_len
        if future_start < 0 or future_end > frames:
            raise ValueError(
                "JointDomeCFMV5 requires 0 <= traj_frame_start and "
                "traj_frame_start + traj_len <= frames, got "
                f"{future_start} + {self.traj_len} > {frames}."
            )

        future_traj_tokens = traj_tokens[:, future_start:future_end]
        if self.traj_pool_type == "mean":
            return future_traj_tokens.mean(dim=1)

        pool_logits = self.traj_frame_pool(future_traj_tokens).squeeze(-1)
        pool_weights = pool_logits.softmax(dim=1)
        return (future_traj_tokens * pool_weights.unsqueeze(-1)).sum(dim=1)

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
        num_patches = x.shape[1]

        t_emb = self.t_embedder(t, use_fp16=use_fp16)
        commands = self._get_commands(commands, metas, batches, x.device)
        command_emb = self.command_embedder(commands)
        cond_base = self.cond_fuser(torch.cat([t_emb, command_emb], dim=-1))

        timestep_spatial = repeat(cond_base, "n d -> (n c) d", c=frames)
        timestep_temp = repeat(
            cond_base,
            "n d -> (n c) d",
            c=num_patches + self.traj_len,
        )

        traj_tokens = self._encode_traj_tokens(traj_t)
        traj_frame_tokens = repeat(traj_tokens, "b k d -> (b f) k d", f=frames)
        x = torch.cat([traj_frame_tokens, x], dim=1)

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]

            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(spatial_block),
                x,
                timestep_spatial,
                use_reentrant=False,
            )

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

        traj_tokens, x = x[:, : self.traj_len, :], x[:, self.traj_len :, :]

        x = self.final_layer(x, timestep_spatial)
        occ = self.unpatchify(x)
        occ = rearrange(occ, "(b f) c h w -> b f c h w", b=batches)

        traj_tokens = rearrange(
            traj_tokens,
            "(b f) k d -> b f k d",
            b=batches,
        )
        traj_tokens = self._pool_future_traj_tokens(traj_tokens, frames)
        traj, traj_modes = self._decode_traj(traj_tokens, commands)

        return {
            "occ": occ,
            "traj": traj,
            "traj_modes": traj_modes,
            "commands": commands,
        }


@MODELS.register_module()
class JointDomeV5(JointDomeCFMV5):
    """Registry alias for JointDomeCFMV5."""

    pass


@MODELS.register_module()
class JointDomeCFMV6(JointDomeV3):
    """Joint DOME-CFM v6 with frame-aligned trajectory tokens.

    History trajectory is used as a condition on the observed frames, while
    future trajectory remains the joint flow target on future frames. Each
    frame receives exactly one trajectory token instead of the full trajectory
    sequence being broadcast to every frame.
    """

    def __init__(
        self,
        *args,
        hist_frame_start=0,
        hist_len=4,
        traj_frame_start=4,
        use_hist_cfg=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hist_frame_start = int(hist_frame_start)
        self.hist_len = int(hist_len)
        self.traj_frame_start = int(traj_frame_start)
        self.use_hist_cfg = bool(use_hist_cfg)

        self.frame_traj_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_frames, self.hidden_size),
            requires_grad=False,
        )
        frame_pos_embed = get_1d_sincos_temp_embed(self.hidden_size, self.num_frames)
        self.frame_traj_pos_embed.data.copy_(
            torch.from_numpy(frame_pos_embed).float().unsqueeze(0)
        )
        self.null_traj_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def _encode_traj_sequence(self, traj):
        if traj is None:
            return None
        traj_tokens = self._encode_traj(traj)
        if traj_tokens.dim() == 2:
            traj_tokens = traj_tokens[:, None, :]
        return traj_tokens

    def _scatter_tokens(self, frame_tokens, encoded_tokens, start_index, batch_mask=None):
        if encoded_tokens is None:
            return frame_tokens

        seq_len = encoded_tokens.shape[1]
        end_index = start_index + seq_len
        if start_index < 0 or end_index > frame_tokens.shape[1]:
            raise ValueError(
                f"trajectory token range [{start_index}, {end_index}) exceeds num_frames="
                f"{frame_tokens.shape[1]}"
            )

        pos_embed = self.frame_traj_pos_embed[:, start_index:end_index].to(
            device=encoded_tokens.device,
            dtype=encoded_tokens.dtype,
        )
        encoded_tokens = encoded_tokens + pos_embed
        if batch_mask is not None:
            null_tokens = self.null_traj_token.to(
                device=encoded_tokens.device,
                dtype=encoded_tokens.dtype,
            ).expand(encoded_tokens.shape[0], seq_len, -1)
            encoded_tokens = torch.where(
                batch_mask[:, None, None],
                null_tokens,
                encoded_tokens,
            )
        frame_tokens[:, start_index:end_index] = encoded_tokens
        return frame_tokens

    def _build_frame_traj_tokens(
        self,
        *,
        traj_t,
        hist_traj,
        batches,
        frames,
        device,
        dtype,
        hist_drop_mask=None,
    ):
        frame_tokens = self.null_traj_token.to(device=device, dtype=dtype).expand(
            batches, frames, -1
        ).clone()
        frame_tokens = frame_tokens + self.frame_traj_pos_embed[:, :frames].to(
            device=device,
            dtype=dtype,
        )

        hist_tokens = self._encode_traj_sequence(hist_traj)
        frame_tokens = self._scatter_tokens(
            frame_tokens,
            hist_tokens,
            self.hist_frame_start,
            batch_mask=hist_drop_mask if self.use_hist_cfg else None,
        )

        future_tokens = self._encode_traj_sequence(traj_t)
        frame_tokens = self._scatter_tokens(
            frame_tokens,
            future_tokens,
            self.traj_frame_start,
        )
        return frame_tokens

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
        hist_traj=None,
        hist_drop_mask=None,
        commands=None,
    ):
        if use_fp16:
            x = x.to(dtype=torch.float16)

        batches, frames, channels, high, weight = x.shape

        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed
        num_patches = x.shape[1]

        t_emb = self.t_embedder(t, use_fp16=use_fp16)
        commands = self._get_commands(commands, metas, batches, x.device)
        command_emb = self.command_embedder(commands)
        cond_base = self.cond_fuser(torch.cat([t_emb, command_emb], dim=-1))

        timestep_spatial = repeat(cond_base, "n d -> (n c) d", c=frames)
        timestep_temp = repeat(
            cond_base,
            "n d -> (n c) d",
            c=num_patches + 1,
        )

        frame_traj_tokens = self._build_frame_traj_tokens(
            traj_t=traj_t,
            hist_traj=hist_traj,
            batches=batches,
            frames=frames,
            device=x.device,
            dtype=x.dtype,
            hist_drop_mask=hist_drop_mask,
        )
        frame_traj_tokens = rearrange(frame_traj_tokens, "b f d -> (b f) 1 d")
        x = torch.cat([frame_traj_tokens, x], dim=1)

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]

            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(spatial_block),
                x,
                timestep_spatial,
                use_reentrant=False,
            )

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

        traj_tokens, x = x[:, :1, :], x[:, 1:, :]

        x = self.final_layer(x, timestep_spatial)
        occ = self.unpatchify(x)
        occ = rearrange(occ, "(b f) c h w -> b f c h w", b=batches)

        traj_tokens = rearrange(traj_tokens, "(b f) 1 d -> b f d", b=batches)
        future_end = self.traj_frame_start + self.traj_len
        if future_end > frames:
            raise ValueError(
                f"future trajectory range [{self.traj_frame_start}, {future_end}) exceeds "
                f"num_frames={frames}"
            )
        future_traj_tokens = traj_tokens[:, self.traj_frame_start:future_end]
        traj, traj_modes = self._decode_traj(future_traj_tokens, commands)

        return {
            "occ": occ,
            "traj": traj,
            "traj_modes": traj_modes,
            "commands": commands,
        }


@MODELS.register_module()
class JointDomeV6(JointDomeCFMV6):
    """Registry alias for JointDomeCFMV6."""

    pass
