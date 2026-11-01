"""Attention-I: intra-sensor time-frequency dual-domain attention (Section 2.3).

Temporal branch: bidirectional selective SSM (Eqs. 8-11) followed by
attention with the adaptive Gaussian locality bias of Eqs. (12, 13),
truncated to |i - j| <= w_b.
Spectral branch: attention over harmonic-band tokens with the additive
harmonic-family prior lambda_h * Lambda of Eqs. (14, 15).
The two branches query each other (Eq. 16), are pooled by additive
attention (Eq. 17) and merged by a feature-wise gate (Eq. 18).
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .ssm import BiSSMBlock, TransformerBlock

NEG_INF = -1e9


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with an additive logit bias."""

    def __init__(self, d_model, heads, dropout=0.0):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.dk = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, xq, xkv, bias=None, key_mask=None, return_attn=False):
        """xq: (B, Nq, d); xkv: (B, Nk, d); bias broadcastable to (B, H, Nq, Nk);
        key_mask: (B, Nk) bool, True for valid keys."""
        b, nq, _ = xq.shape
        nk = xkv.shape[1]
        q = self.q(xq).view(b, nq, self.heads, self.dk).transpose(1, 2)
        k = self.k(xkv).view(b, nk, self.heads, self.dk).transpose(1, 2)
        v = self.v(xkv).view(b, nk, self.heads, self.dk).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if bias is not None:
            logits = logits + bias
        if key_mask is not None:
            logits = logits.masked_fill(~key_mask[:, None, None, :], NEG_INF)
        attn = torch.softmax(logits, dim=-1)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, nq, -1)
        out = self.o(out)
        return (out, attn) if return_attn else out


class FeedForward(nn.Module):
    def __init__(self, d_model, mult=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(mult * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransientTemporalAttention(nn.Module):
    """Pre-norm attention layer with the adaptive locality bias G (Eqs. 12, 13)."""

    def __init__(self, d_model, heads, sigma_min=1.0, sigma_max=16.0, gamma=4.0, band=48,
                 use_bias=True, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.sigma_min, self.sigma_max, self.gamma = sigma_min, sigma_max, gamma
        self.band = band
        self.use_bias = use_bias

    def locality_bias(self, y, mask):
        """Return G (B, 1, N, N) including the band truncation, and sigma (B, N)."""
        n = y.shape[1]
        e = (y[:, 1:] - y[:, :-1]).norm(dim=-1)
        e = torch.cat([e[:, :1], e], dim=1)
        m = mask.to(y.dtype)
        e_bar = (e * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
        sigma = self.sigma_max - (self.sigma_max - self.sigma_min) * torch.sigmoid(self.gamma * (e - e_bar))
        idx = torch.arange(n, device=y.device, dtype=y.dtype)
        dist = idx[:, None] - idx[None, :]
        if self.use_bias:
            g = -(dist ** 2)[None] / (2.0 * sigma.unsqueeze(-1) ** 2)
        else:
            g = torch.zeros(y.shape[0], n, n, device=y.device, dtype=y.dtype)
        g = g.masked_fill(dist.abs()[None] > self.band, NEG_INF)
        return g.unsqueeze(1), sigma

    def forward(self, x, mask):
        h = self.norm1(x)
        g, sigma = self.locality_bias(h, mask)
        x = x + self.attn(h, h, bias=g, key_mask=mask)
        x = x + self.ff(self.norm2(x))
        return x, sigma


def harmonic_prior(order, mask, chi=0.05):
    """Lambda of Eq. (15) for tokens with harmonic orders `order` (B, F)."""

    def kappa(r):
        return torch.exp(-((r - torch.round(r)) ** 2) / (2.0 * chi ** 2))

    k_single = kappa(order)
    hi = torch.maximum(order.unsqueeze(-1), order.unsqueeze(-2))
    lo = torch.minimum(order.unsqueeze(-1), order.unsqueeze(-2)).clamp_min(1e-3)
    lam = k_single.unsqueeze(-1) * k_single.unsqueeze(-2) * kappa(hi / lo)
    valid = (mask.unsqueeze(-1) & mask.unsqueeze(-2)) & (order > 0).unsqueeze(-1) & (order > 0).unsqueeze(-2)
    return lam * valid.to(lam.dtype)


class HarmonicSpectralAttention(nn.Module):
    """Pre-norm attention layer with the prior lambda_h * Lambda (Eq. 14)."""

    def __init__(self, d_model, heads, chi=0.05, lambda_init=1.0, use_prior=True, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.chi = chi
        self.lambda_h = nn.Parameter(torch.tensor(float(lambda_init)))
        self.use_prior = use_prior

    def forward(self, x, mask, order, prior_on):
        h = self.norm1(x)
        bias = None
        if self.use_prior:
            lam = harmonic_prior(order, mask, self.chi) * prior_on.view(-1, 1, 1)
            bias = (self.lambda_h * lam).unsqueeze(1)
        x = x + self.attn(h, h, bias=bias, key_mask=mask)
        x = x + self.ff(self.norm2(x))
        return x


class AttentionPool(nn.Module):
    """Additive attention pooling (Eq. 17)."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.w = nn.Linear(d_model, 1, bias=False)

    def forward(self, x, mask):
        score = self.w(torch.tanh(self.proj(x))).squeeze(-1)
        score = score.masked_fill(~mask, NEG_INF)
        a = torch.softmax(score, dim=-1)
        return torch.einsum("bn,bnd->bd", a, x)


class AttentionI(nn.Module):
    """Shared intra-sensor encoder; returns the fused embedding r^(m)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        heads = cfg["heads"]
        self.use_cross = cfg.get("use_cross_attention", True)
        self.use_checkpoint = cfg.get("checkpoint_encoder", False)
        if cfg.get("encoder", "mamba") == "mamba":
            self.encoder = nn.ModuleList(
                [BiSSMBlock(d, cfg["d_state"], cfg["expand"], cfg.get("use_cuda_kernel", False))
                 for _ in range(cfg["n_ssm_blocks"])]
            )
        else:
            self.encoder = nn.ModuleList([TransformerBlock(d, heads) for _ in range(cfg["n_ssm_blocks"])])
        self.temporal = nn.ModuleList(
            [
                TransientTemporalAttention(d, heads, cfg["sigma_min"], cfg["sigma_max"], cfg["gamma"],
                                           cfg["band"], cfg.get("use_transient_bias", True), cfg["dropout"])
                for _ in range(cfg["n_attn_layers"])
            ]
        )
        self.spectral = nn.ModuleList(
            [
                HarmonicSpectralAttention(d, heads, cfg["chi"], cfg["lambda_h"], cfg.get("use_harmonic_prior", True),
                                          cfg["dropout"])
                for _ in range(cfg["n_attn_layers"])
            ]
        )
        self.norm_t = nn.LayerNorm(d)
        self.norm_f = nn.LayerNorm(d)
        self.cross_tf = MultiHeadAttention(d, heads, cfg["dropout"])
        self.cross_ft = MultiHeadAttention(d, heads, cfg["dropout"])
        self.pool_t = AttentionPool(d)
        self.pool_f = AttentionPool(d)
        self.gate = nn.Linear(2 * d, d)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, ht, mask_t, hf, mask_f, order, prior_on):
        for blk in self.encoder:
            if self.use_checkpoint and self.training:
                ht = checkpoint(blk, ht, mask_t, use_reentrant=False)
            else:
                ht = blk(ht, mask_t)
        sigma = None
        for layer in self.temporal:
            ht, sigma = layer(ht, mask_t)
        for layer in self.spectral:
            hf = layer(hf, mask_f, order, prior_on)
        nt, nf = self.norm_t(ht), self.norm_f(hf)
        if self.use_cross:
            ht = nt + self.cross_tf(nt, nf, key_mask=mask_f)
            hf = nf + self.cross_ft(nf, nt, key_mask=mask_t)
        else:
            ht, hf = nt, nf
        bt = self.pool_t(ht, mask_t)
        bf = self.pool_f(hf, mask_f)
        g = torch.sigmoid(self.gate(torch.cat([bt, bf], dim=-1)))
        r = g * bt + (1.0 - g) * bf
        return self.out_norm(r), {"sigma": sigma, "gate": g}
