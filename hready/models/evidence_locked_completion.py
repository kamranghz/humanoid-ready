"""Evidence-locked completion: conditional VAE over per-frame latents on the simulated evidence schema.

``forward(obs, rig)`` is the only inference path (evidence + mandatory rig) and decodes the
prior mean, i.e. a deterministic point estimate. ``sample(obs, rig, n, generator)`` draws from the
learned conditional prior. ``forward_train(obs, rig, targets)`` is used by the training loss only:
the posterior encoder reads GT targets there and nowhere else.
With ``generative=False`` the latent is fixed at zero (no posterior, no KL): deterministic ablation.

Pre-registered-comparison options (defaults reproduce the ablation-study model exactly): ``latent_mode="window"`` uses one latent per window
(prior and posterior read the time-mean of their tokens), ``logvar_min`` raises the log-variance
floor, ``free_bits`` > 0 floors the KL of each latent dimension (nats).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hready.models.ego_complete_motion import (
    NUM_BODY,
    NUM_CONTACT,
    NUM_KP,
    evidence_features,
)


def _encoder(
    d_model: int, n_heads: int, n_layers: int, ff_dim: int, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model,
        n_heads,
        dim_feedforward=ff_dim,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class EvidenceLockedCompletion(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dec_layers: int = 4,
        post_layers: int = 2,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 64,
        latent_dim: int = 32,
        generative: bool = True,
        latent_mode: str = "frame",
        logvar_min: float = -8.0,
        free_bits: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_mode not in ("frame", "window"):
            raise ValueError(
                f"latent_mode must be 'frame' or 'window', got {latent_mode!r}"
            )
        self.generative = generative
        self.latent_dim = latent_dim
        self.latent_mode = latent_mode
        self.logvar_min = float(logvar_min)
        self.free_bits = float(free_bits)
        in_dim = NUM_KP * 8 + 12
        self.frame_in = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.evidence = _encoder(d_model, n_heads, n_layers, ff_dim, dropout)
        self.prior = nn.Linear(d_model, 2 * latent_dim)
        # Posterior input per frame: GT joints relative to the head (66) + GT root 6D (6) + contact (4).
        self.post_in = nn.Linear(NUM_KP * 3 + 6 + NUM_CONTACT, d_model)
        self.posterior = _encoder(d_model, n_heads, post_layers, ff_dim, dropout)
        self.post_out = nn.Linear(d_model, 2 * latent_dim)
        self.z_in = nn.Linear(latent_dim, d_model)
        self.decoder = _encoder(d_model, n_heads, dec_layers, ff_dim, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head_transl = nn.Linear(d_model, 3)
        self.head_root = nn.Linear(d_model, 6)
        self.head_body = nn.Linear(d_model, NUM_BODY * 6)
        self.head_contact = nn.Linear(d_model, NUM_CONTACT)
        from hready.body.rotations import matrix_to_rotation_6d

        self.register_buffer(
            "rot6d_identity", matrix_to_rotation_6d(torch.eye(3)), persistent=False
        )
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

    def _encode(self, obs: dict[str, Tensor], rig: dict[str, Tensor]) -> Tensor:
        x = evidence_features(obs, rig)
        t = x.shape[1]
        if t > self.pos.shape[1]:
            raise ValueError(f"window {t} > max_len {self.pos.shape[1]}")
        return self.evidence(self.frame_in(x) + self.pos[:, :t])

    def _pool(self, x: Tensor) -> Tensor:
        """Window mode: one token per window (time mean); frame mode: unchanged."""
        return x.mean(dim=1, keepdim=True) if self.latent_mode == "window" else x

    def _prior(self, h: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = self.prior(self._pool(h)).chunk(2, dim=-1)
        return mu, logvar.clamp(self.logvar_min, 4.0)

    @staticmethod
    def _expand(z: Tensor, h: Tensor) -> Tensor:
        return z.expand(-1, h.shape[1], -1) if z.shape[1] != h.shape[1] else z

    def _decode(
        self, h: Tensor, z: Tensor, rig: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        y = self.norm(self.decoder(h + self.z_in(z)))
        b, t, _ = y.shape
        ident = self.rot6d_identity.to(y.dtype)
        return {
            "transl": rig["head_pos_world"].to(y.dtype) + self.head_transl(y),
            "root_rot_6d": self.head_root(y) + ident,
            "body_rot_6d": (
                self.head_body(y).reshape(b, t, NUM_BODY, 6) + ident
            ).reshape(b, t, NUM_BODY * 6),
            "contact_logits": self.head_contact(y),
        }

    def forward(
        self, obs: dict[str, Tensor], rig: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        h = self._encode(obs, rig)
        z = (
            self._expand(self._prior(h)[0], h)
            if self.generative
            else h.new_zeros(*h.shape[:2], self.latent_dim)
        )
        return self._decode(h, z, rig)

    @torch.no_grad()
    def sample(
        self,
        obs: dict[str, Tensor],
        rig: dict[str, Tensor],
        n: int,
        generator: torch.Generator | None = None,
    ) -> list[dict[str, Tensor]]:
        h = self._encode(obs, rig)
        if not self.generative:
            return [self.forward(obs, rig)] * n
        mu, logvar = self._prior(h)
        std = (0.5 * logvar).exp()
        outs = []
        for _ in range(n):
            eps = torch.randn(
                mu.shape, generator=generator, device=mu.device, dtype=mu.dtype
            )
            outs.append(self._decode(h, self._expand(mu + std * eps, h), rig))
        return outs

    def forward_train(
        self, obs: dict[str, Tensor], rig: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Training only: posterior sample -> decode; returns outputs and per-frame KL ``(B, T)``."""
        h = self._encode(obs, rig)
        if not self.generative:
            return self._decode(
                h, h.new_zeros(*h.shape[:2], self.latent_dim), rig
            ), h.new_zeros(h.shape[:2])
        from hready.body.rotations import matrix_to_rotation_6d

        head = rig["head_pos_world"].unsqueeze(2)
        rel = (targets["joints_gt_22"] - head).flatten(2)
        p_in = torch.cat(
            [rel, matrix_to_rotation_6d(targets["root_R"]), targets["contact"]], dim=-1
        )
        q = self._pool(self.posterior(self.post_in(p_in.to(h.dtype)) + h))
        mu_q, logvar_q = self.post_out(q).chunk(2, dim=-1)
        logvar_q = logvar_q.clamp(self.logvar_min, 4.0)
        mu_p, logvar_p = self._prior(h)
        z = mu_q + (0.5 * logvar_q).exp() * torch.randn_like(mu_q)
        if self.free_bits > 0:
            kl_d = 0.5 * (
                logvar_p
                - logvar_q
                + (logvar_q.exp() + (mu_q - mu_p).square()) / logvar_p.exp()
                - 1.0
            )
            kl = kl_d.clamp_min(self.free_bits).sum(dim=-1)
        else:
            kl = 0.5 * (
                logvar_p
                - logvar_q
                + (logvar_q.exp() + (mu_q - mu_p).square()) / logvar_p.exp()
                - 1.0
            ).sum(dim=-1)
        # Window mode: one KL per window, repeated per frame so the loss masks it like per-frame mode.
        return self._decode(h, self._expand(z, h), rig), self._expand(
            kl.unsqueeze(-1), h
        ).squeeze(-1)
