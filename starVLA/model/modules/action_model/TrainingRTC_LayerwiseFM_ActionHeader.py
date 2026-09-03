"""Layerwise flow-matching head with training-time RTC action conditioning.

This is a parameter-compatible extension of ``LayerwiseFlowmatchingActionHead``.
It follows Algorithm 1 of *Training-Time Action Conditioning for Efficient
Real-Time Chunking* (Black et al., 2025): a randomly sampled action prefix is
kept clean at flow time 1, postfix actions are noised at the sampled flow time,
and the velocity loss is evaluated only on the postfix.

No learnable parameter is added or renamed, so a vanilla QwenPI_v3 checkpoint
can be loaded before RTC fine-tuning.
"""

from __future__ import annotations

import torch

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    swish,
)


class TrainingRTCLayerwiseFlowmatchingActionHead(
    LayerwiseFlowmatchingActionHead
):
    """LayerwiseFM with per-token flow time and prefix-conditioned loss."""

    def __init__(self, global_config, **kwargs):
        super().__init__(global_config=global_config, **kwargs)
        self.rtc_max_delay_steps = int(
            getattr(self.config, "rtc_max_delay_steps", 10)
        )
        if not 0 <= self.rtc_max_delay_steps < self.action_horizon:
            raise ValueError(
                "rtc_max_delay_steps must satisfy "
                f"0 <= delay < action_horizon={self.action_horizon}; "
                f"got {self.rtc_max_delay_steps}"
            )
        delay_sampling = str(
            getattr(self.config, "rtc_delay_sampling", "uniform")
        )
        if delay_sampling != "uniform":
            raise ValueError(
                "Only rtc_delay_sampling='uniform' is currently supported; "
                f"got {delay_sampling!r}"
            )
        self.rtc_delay_sampling = delay_sampling

    def _encode_actions_per_token(
        self,
        actions: torch.Tensor,
        timestep_buckets: torch.Tensor,
    ) -> torch.Tensor:
        """Run the existing ActionEncoder with a timestep for every row."""

        batch_size, horizon, _ = actions.shape
        if timestep_buckets.shape != (batch_size, horizon):
            raise ValueError(
                "Per-token timesteps must have shape "
                f"{(batch_size, horizon)}, got {tuple(timestep_buckets.shape)}"
            )
        action_embedding = self.action_encoder.layer1(actions)
        time_embedding = self.action_encoder.pos_encoding(
            timestep_buckets
        ).to(dtype=action_embedding.dtype)
        encoded = torch.cat([action_embedding, time_embedding], dim=-1)
        encoded = swish(self.action_encoder.layer2(encoded))
        return self.action_encoder.layer3(encoded)

    @staticmethod
    def _block_forward_per_token(
        block,
        hidden_states: torch.Tensor,
        encoder_hidden_states,
        encoder_attention_mask,
        token_time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """BasicTransformerBlock forward with token-wise AdaLN modulation."""

        if block.norm_type == "ada_norm":
            modulation = block.norm1.linear(
                block.norm1.silu(token_time_embedding)
            )
            scale, shift = modulation.chunk(2, dim=-1)
            normalized = block.norm1.norm(hidden_states) * (1 + scale) + shift
        else:
            normalized = block.norm1(hidden_states)

        if block.pos_embed is not None:
            normalized = block.pos_embed(normalized)

        attention_output = block.attn1(
            normalized,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        if block.final_dropout is not None:
            attention_output = block.final_dropout(attention_output)
        hidden_states = attention_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        feed_forward_output = block.ff(block.norm3(hidden_states))
        hidden_states = feed_forward_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states

    def _dit_forward_per_token(
        self,
        hidden_states: torch.Tensor,
        vl_embs_list: list,
        token_timestep_buckets: torch.Tensor,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        """Use existing DiT weights with a timestep embedding per token."""

        batch_size, token_count, _ = hidden_states.shape
        if token_timestep_buckets.shape != (batch_size, token_count):
            raise ValueError(
                "DiT token timesteps must match hidden token axes: "
                f"expected {(batch_size, token_count)}, "
                f"got {tuple(token_timestep_buckets.shape)}"
            )

        flat_time_embedding = self.model.timestep_encoder(
            token_timestep_buckets.reshape(-1)
        )
        token_time_embedding = flat_time_embedding.reshape(
            batch_size, token_count, -1
        )
        hidden_states = hidden_states.contiguous()
        vl_embs_list = [embedding.contiguous() for embedding in vl_embs_list]

        for index, block in enumerate(self.model.transformer_blocks):
            use_self_attention = (
                index % 2 == 1
                and self.model.config.interleave_self_attention
                and self.model.config.use_canonical_forward
            )
            block_encoder = None if use_self_attention else vl_embs_list[index]
            block_mask = None if use_self_attention else encoder_attention_mask
            hidden_states = self._block_forward_per_token(
                block,
                hidden_states,
                block_encoder,
                block_mask,
                token_time_embedding,
            )
        return hidden_states

    def _predict_velocity_per_token(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        action_timestep_buckets: torch.Tensor,
        base_timestep_buckets: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        """Predict velocity while exposing per-action flow timesteps to DiT."""

        batch_size = actions.shape[0]
        action_features = self._encode_actions_per_token(
            actions, action_timestep_buckets
        )
        if self.config.add_pos_embed:
            positions = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=actions.device,
            )
            action_features = action_features + self.position_embedding(
                positions
            ).unsqueeze(0)

        state_features = self.state_encoder(state) if state is not None else None
        future_features = self.future_tokens.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        prefix_features = (
            torch.cat([state_features, future_features], dim=1)
            if state_features is not None
            else future_features
        )
        hidden_states = torch.cat([prefix_features, action_features], dim=1)

        prefix_timestep_buckets = base_timestep_buckets[:, None].expand(
            -1, prefix_features.shape[1]
        )
        token_timestep_buckets = torch.cat(
            [prefix_timestep_buckets, action_timestep_buckets], dim=1
        )
        model_output = self._dit_forward_per_token(
            hidden_states,
            vl_embs_list,
            token_timestep_buckets,
            encoder_attention_mask=encoder_attention_mask,
        )
        decoded = self.action_decoder(model_output)
        return decoded[:, -self.action_horizon :]

    def _sample_training_delays(self, batch_size: int, device) -> torch.Tensor:
        """Sample delays uniformly from 0 through rtc_max_delay_steps."""

        return torch.randint(
            low=0,
            high=self.rtc_max_delay_steps + 1,
            size=(batch_size,),
            device=device,
        )

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        """Compute the training-time RTC prefix-conditioned flow loss."""

        batch_size, horizon, action_dim = actions.shape
        if horizon != self.action_horizon:
            raise ValueError(
                f"Expected action horizon {self.action_horizon}, got {horizon}"
            )

        noise = torch.randn_like(actions)
        base_time = self.sample_time(
            batch_size, device=actions.device, dtype=actions.dtype
        )
        delays = self._sample_training_delays(batch_size, actions.device)
        prefix_mask = (
            torch.arange(horizon, device=actions.device)[None, :]
            < delays[:, None]
        )

        action_time = torch.where(
            prefix_mask,
            torch.ones_like(base_time[:, None]),
            base_time[:, None],
        )
        noisy_actions = (
            action_time[:, :, None] * actions
            + (1 - action_time[:, :, None]) * noise
        )
        target_velocity = actions - noise

        action_timestep_buckets = torch.clamp(
            (action_time * self.num_timestep_buckets).long(),
            min=0,
            # Unlike ordinary flow samples, the clean RTC prefix is exactly
            # t=1.  The sinusoidal encoders accept this endpoint bucket.
            max=self.num_timestep_buckets,
        )
        base_timestep_buckets = torch.clamp(
            (base_time * self.num_timestep_buckets).long(),
            min=0,
            max=self.num_timestep_buckets - 1,
        )
        predicted_velocity = self._predict_velocity_per_token(
            vl_embs_list,
            noisy_actions,
            action_timestep_buckets,
            base_timestep_buckets,
            state=state,
            encoder_attention_mask=encoder_attention_mask,
        )

        postfix_mask = (~prefix_mask)[:, :, None]
        squared_error = (predicted_velocity - target_velocity).square()
        # Match Algorithm 1: sum over action dimensions, then normalize by
        # the number of postfix time steps (not by action_dim as well).
        denominator = postfix_mask.sum().clamp_min(1)
        return (squared_error * postfix_mask).sum() / denominator

    @torch.no_grad()
    def predict_action_realtime(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        prev_action_chunk: torch.Tensor = None,
        inference_delay: int = 1,
        mode: str = "simulated_delay",
        encoder_attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate a postfix with training-time hard-prefix conditioning."""

        if mode != "simulated_delay":
            return super().predict_action_realtime(
                vl_embs_list,
                state,
                prev_action_chunk=prev_action_chunk,
                inference_delay=inference_delay,
                mode=mode,
                **kwargs,
            )
        if prev_action_chunk is None or inference_delay <= 0:
            return self.predict_action(
                vl_embs_list,
                state,
                encoder_attention_mask=encoder_attention_mask,
            )
        if inference_delay > self.rtc_max_delay_steps:
            raise ValueError(
                f"inference_delay={inference_delay} exceeds trained maximum "
                f"{self.rtc_max_delay_steps}"
            )
        if inference_delay >= self.action_horizon:
            raise ValueError("inference_delay must be smaller than action_horizon")

        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        dtype = vl_embs_list[0].dtype
        previous = prev_action_chunk.to(device=device, dtype=dtype)
        if previous.shape[1] < self.action_horizon:
            padding = torch.zeros(
                batch_size,
                self.action_horizon - previous.shape[1],
                self.action_dim,
                device=device,
                dtype=dtype,
            )
            previous = torch.cat([previous, padding], dim=1)
        elif previous.shape[1] > self.action_horizon:
            previous = previous[:, : self.action_horizon]

        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        prefix_mask = torch.arange(
            self.action_horizon, device=device
        )[None, :, None] < inference_delay
        prefix_mask = prefix_mask.expand(batch_size, -1, -1)
        prefix_bucket = self.num_timestep_buckets
        integration_step = 1.0 / self.num_inference_timesteps

        for step in range(self.num_inference_timesteps):
            continuous_time = step / float(self.num_inference_timesteps)
            actions = torch.where(prefix_mask, previous, actions)
            postfix_bucket = min(
                int(continuous_time * self.num_timestep_buckets),
                self.num_timestep_buckets - 1,
            )
            action_timestep_buckets = torch.full(
                (batch_size, self.action_horizon),
                postfix_bucket,
                device=device,
                dtype=torch.long,
            )
            action_timestep_buckets[:, :inference_delay] = prefix_bucket
            base_timestep_buckets = torch.full(
                (batch_size,),
                postfix_bucket,
                device=device,
                dtype=torch.long,
            )
            velocity = self._predict_velocity_per_token(
                vl_embs_list,
                actions,
                action_timestep_buckets,
                base_timestep_buckets,
                state=state,
                encoder_attention_mask=encoder_attention_mask,
            )
            actions = actions + integration_step * velocity
            actions = torch.where(prefix_mask, previous, actions)
        return actions


def get_action_model(config=None):
    return TrainingRTCLayerwiseFlowmatchingActionHead(global_config=config)
