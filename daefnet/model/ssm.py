"""Selective state-space temporal encoder (Section 2.3.1).

Continuous system (Eq. 8):  s'(t) = A s(t) + B h(t),  y(t) = C s(t)
ZOH discretization (Eq. 9): A_bar = exp(dA),  B_bar = (dA)^-1 (exp(dA) - I) dB
Recurrence (Eq. 10):        s_i = A_bar_i s_{i-1} + B_bar_i h_i,  y_i = C_i s_i
Selectivity (Eq. 11):       d_i = softplus(W_d h_i),  B_i = W_B h_i,  C_i = W_C h_i

A is diagonal, so the ZOH input matrix reduces to (exp(d a) - 1) / a * B.
The scan below is a plain PyTorch reference implementation that runs in
time linear in the number of tokens. If the `mamba_ssm` package is installed
and `use_cuda_kernel=True`, its fused kernel is used instead; that kernel
discretizes B with d * B rather than the exact ZOH expression.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


def selective_scan(u, delta, a, b, c, d):
    """Reference selective scan.

    Args:
        u, delta: (batch, channels, length)
        a: (channels, state) with negative entries
        b, c: (batch, state, length)
        d: (channels,) skip connection
    Returns:
        y: (batch, channels, length)
    """
    bsz, ch, length = u.shape
    da = delta.unsqueeze(-1) * a.view(1, ch, 1, -1)  # (B, C, L, N)
    a_bar = torch.exp(da)
    b_scale = torch.where(da.abs() < 1e-4, delta.unsqueeze(-1), (a_bar - 1.0) / a.view(1, ch, 1, -1))
    bu = b_scale * b.transpose(1, 2).unsqueeze(1) * u.unsqueeze(-1)  # (B, C, L, N)
    c_t = c.transpose(1, 2)  # (B, L, N)
    s = u.new_zeros(bsz, ch, a.shape[-1])
    ys = []
    for i in range(length):
        s = a_bar[:, :, i] * s + bu[:, :, i]
        ys.append(torch.einsum("bcn,bn->bc", s, c_t[:, i]))
    y = torch.stack(ys, dim=-1)
    return y + u * d.view(1, -1, 1)


class SelectiveSSM(nn.Module):
    """One-directional selective state-space mixer."""

    def __init__(self, d_model, d_state=16, expand=2, d_conv=4, dt_min=1e-3, dt_max=1e-1,
                 use_cuda_kernel=False):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.use_cuda_kernel = use_cuda_kernel and selective_scan_fn is not None

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.d = nn.Parameter(torch.ones(d_inner))

    def forward(self, h, mask=None):
        """h: (B, L, d_model); mask: (B, L) bool, True for valid tokens."""
        length = h.shape[1]
        x, z = self.in_proj(h).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :length]
        x = F.silu(x)
        if mask is not None:
            x = x * mask.unsqueeze(1).to(x.dtype)
        dbc = self.x_proj(x.transpose(1, 2))
        dt, b, c = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt)).transpose(1, 2)  # (B, D, L)
        a = -torch.exp(self.a_log.float())
        b = b.transpose(1, 2).contiguous()
        c = c.transpose(1, 2).contiguous()
        if self.use_cuda_kernel and x.is_cuda:
            y = selective_scan_fn(x.contiguous(), delta.contiguous(), a, b, c, self.d.float())
        else:
            y = selective_scan(x, delta, a, b, c, self.d)
        y = y.transpose(1, 2) * F.silu(z)
        return self.out_proj(y)


class BiSSMBlock(nn.Module):
    """Pre-norm residual block with forward and backward scans summed."""

    def __init__(self, d_model, d_state=16, expand=2, use_cuda_kernel=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = SelectiveSSM(d_model, d_state, expand, use_cuda_kernel=use_cuda_kernel)
        self.bwd = SelectiveSSM(d_model, d_state, expand, use_cuda_kernel=use_cuda_kernel)

    def forward(self, x, mask):
        h = self.norm(x) * mask.unsqueeze(-1).to(x.dtype)
        y_f = self.fwd(h, mask)
        y_b = self.bwd(h.flip(1), mask.flip(1)).flip(1)
        return x + y_f + y_b


class TransformerBlock(nn.Module):
    """Full self-attention block, used for the encoder ablation of Table 9."""

    def __init__(self, d_model, heads=4, ff_mult=2, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model, heads, ff_mult * d_model, dropout,
                                                batch_first=True, norm_first=True)

    def forward(self, x, mask):
        return self.layer(x, src_key_padding_mask=~mask)
