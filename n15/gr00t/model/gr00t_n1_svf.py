"""Opt-in N1.5 BC + SVF critic composition; existing BC/DEAS/IQL paths unchanged."""

from dataclasses import asdict, replace

import torch
from torch import nn

from .action_head.svf_critic import SVFCriticConfig, SVFCriticHead


class GR00TN15SVF(nn.Module):
    """Owns an existing BC policy and freezes it for critic-only training.

    The original DiT actor is retained as ``bc_policy.action_head``. This class
    deliberately does not implement the guided actor or reference BC-SDE yet.
    """

    def __init__(
        self,
        bc_policy,
        critic_config=None,
        *,
        time_embed_dim=None,
        outer_update=None,
        q_loss_type=None,
        v_loss_type=None,
        hl_gauss_min=None,
        hl_gauss_max=None,
    ):
        super().__init__()
        actor = bc_policy.action_head
        cfg = actor.config
        inferred = {
            "backbone_dim": cfg.backbone_embedding_dim,
            "state_dim": cfg.max_state_dim,
            "action_dim": cfg.action_dim,
            "action_horizon": cfg.action_horizon,
            "num_embodiments": cfg.max_num_embodiments,
        }
        if critic_config is None:
            critic_config = SVFCriticConfig(**inferred)
        overrides = {
            "time_embed_dim": time_embed_dim,
            "outer_update": outer_update,
            "q_loss_type": q_loss_type,
            "v_loss_type": v_loss_type,
            "hl_gauss_min": hl_gauss_min,
            "hl_gauss_max": hl_gauss_max,
        }
        critic_config = replace(
            critic_config, **{key: value for key, value in overrides.items() if value is not None}
        )
        for key, value in inferred.items():
            if getattr(critic_config, key) != value:
                raise ValueError(f"SVF {key} must match the BC checkpoint ({value})")
        self.bc_policy = bc_policy.requires_grad_(False).eval()
        self.critic_head = SVFCriticHead(critic_config, actor.vlln, actor.vl_self_attention)
        # Keep new trainable critic weights float32; copied frozen BC modules
        # preserve their original dtype. AMP can be enabled by the training loop.
        self.critic_head.to(device=next(bc_policy.parameters()).device)

    def train(self, mode=True):
        super().train(mode)
        self.bc_policy.eval()
        return self

    @classmethod
    def from_bc_checkpoint(
        cls,
        path,
        critic_config=None,
        *,
        time_embed_dim=None,
        outer_update=None,
        q_loss_type=None,
        v_loss_type=None,
        hl_gauss_min=None,
        hl_gauss_max=None,
        **load_kwargs,
    ):
        """Select Fourier width (e.g. 16 or 64); never forward it to the BC loader.

        An explicit width overrides critic_config; omitted preserves its value
        (16 when no config is supplied). Changing width requires new critics.
        """
        from .gr00t_n1 import GR00T_N1_5

        load_kwargs.update(tune_visual=False, tune_llm=False, tune_projector=False, tune_diffusion_model=False)
        return cls(
            GR00T_N1_5.from_pretrained(path, **load_kwargs),
            critic_config,
            time_embed_dim=time_embed_dim,
            outer_update=outer_update,
            q_loss_type=q_loss_type,
            v_loss_type=v_loss_type,
            hl_gauss_min=hl_gauss_min,
            hl_gauss_max=hl_gauss_max,
        )

    @torch.no_grad()
    def encode_observation(self, inputs):
        backbone_inputs, action_inputs = self.bc_policy.prepare_input(inputs)
        # Do not call actor.forward: it mutates backbone_features in this repo.
        backbone_output = self.bc_policy.backbone(backbone_inputs)
        return self.critic_head.encode_observation(
            backbone_output, action_inputs["state"], action_inputs["embodiment_id"]
        )

    def forward(
        self,
        inputs,
        *,
        actions,
        noisy_actions=None,
        times=None,
        q_targets=None,
        soft_targets=None,
        action_mask=None,
        next_inputs=None,
        chunk_returns=None,
        bootstrap_discounts=None,
    ):
        """TD mode takes explicit Q targets; IQL mode builds targets from transitions."""
        if self.critic_head.config.outer_update == "iql":
            if q_targets is not None:
                raise ValueError("IQL builds Q targets itself; do not supply q_targets")
            if next_inputs is None or chunk_returns is None or bootstrap_discounts is None:
                raise ValueError("IQL requires next_inputs, chunk_returns and bootstrap_discounts")
            observation = self.encode_observation(inputs)
            next_observation = self.encode_observation(next_inputs)
            return self.critic_head.iql_loss(
                observation,
                next_observation,
                actions,
                chunk_returns,
                bootstrap_discounts,
                noisy_actions=noisy_actions,
                times=times,
                soft_targets=soft_targets,
                action_mask=inputs.get("action_mask") if action_mask is None else action_mask,
            )
        if next_inputs is not None or chunk_returns is not None or bootstrap_discounts is not None:
            raise ValueError("Transition inputs require outer_update='iql'; TD mode takes q_targets")
        if noisy_actions is None or times is None:
            raise ValueError("TD forward requires noisy_actions and times")
        if (q_targets is None) != (soft_targets is None):
            raise ValueError("Supply both critic targets, or neither")
        observation = self.encode_observation(inputs)
        if action_mask is None:
            action_mask = inputs.get("action_mask")
        output = self.critic_head(observation, actions, noisy_actions, times, action_mask=action_mask)
        if q_targets is not None:
            output.update(self.critic_head.loss(output, q_targets, soft_targets))
        return output

    def critic_checkpoint(self):
        """Pair with the original BC checkpoint path AND optimizer state to resume."""
        return {
            "format_version": 1,
            "config": asdict(self.critic_head.config),
            "state_dict": self.critic_head.state_dict(),
        }

    def load_critic_checkpoint(self, checkpoint):
        if checkpoint["format_version"] != 1:
            raise ValueError("Unsupported SVF critic checkpoint version")
        if asdict(SVFCriticConfig(**checkpoint["config"])) != asdict(self.critic_head.config):
            raise ValueError("SVF critic checkpoint configuration mismatch")
        self.critic_head.load_state_dict(checkpoint["state_dict"], strict=True)
