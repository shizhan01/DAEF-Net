"""Attention-II: inter-sensor topology- and reliability-aware attention (Section 2.4).

Graph prior (Eq. 20):
    A_hat = mu exp(-d^2 / s^2) + (1 - mu) softmax(ReLU(E1 E2^T))
Reliability (Eq. 21):
    rho_m = sigmoid(w^T [r_m; q_m] + b)
Attention logits (Eqs. 22, 23):
    a_mn = (W_Q r_m)^T (W_K r_n) / sqrt(d_k) + beta log(A_hat_mn + eps) + log rho_n
Update (Eq. 24):
    z_m = LN(r_m + W_O ||_j sum_n pi_mn^(j) W_V^(j) r_n),  o_m = LN(z_m + FFN(z_m))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_i import FeedForward, MultiHeadAttention


class AttentionII(nn.Module):
    def __init__(self, d_model, n_sensors, heads=4, beta=0.8, mu=0.0, distance=None, emb_dim=16, eps=1e-6,
                 graph="physical", use_reliability=True, mode="attention", dropout=0.1):
        super().__init__()
        self.n = n_sensors
        self.beta, self.eps = beta, eps
        self.graph_mode = graph  # "physical" (Eq. 20) or "uniform"
        self.use_reliability = use_reliability
        self.mode = mode  # "attention" or "mean" (ablation)
        self.e1 = nn.Parameter(torch.randn(n_sensors, emb_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_sensors, emb_dim) * 0.1)
        if distance is not None and mu > 0:
            d = torch.as_tensor(distance, dtype=torch.float32)
            nz = d[d > 0]
            s = nz.median() if nz.numel() > 0 else torch.tensor(1.0)
            self.register_buffer("kernel", torch.exp(-(d ** 2) / s ** 2))
            self.mu = float(mu)
        else:
            self.register_buffer("kernel", torch.zeros(n_sensors, n_sensors))
            self.mu = 0.0
        self.rel = nn.Linear(d_model + 4, 1)
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)

    def adjacency(self):
        if self.graph_mode == "uniform":
            return torch.full((self.n, self.n), 1.0 / self.n, device=self.e1.device)
        adaptive = torch.softmax(F.relu(self.e1 @ self.e2.t()), dim=-1)
        return self.mu * self.kernel + (1.0 - self.mu) * adaptive

    def forward(self, r, q):
        """r: (B, M, d) sensor embeddings; q: (B, M, 4) quality descriptors.

        Returns updated embeddings o (B, M, d), reliability rho (B, M) and the
        attention weights (B, H, M, M) or None.
        """
        logit = self.rel(torch.cat([r, q], dim=-1)).squeeze(-1)
        if self.use_reliability:
            rho = torch.sigmoid(logit)
            log_rho = F.logsigmoid(logit)
        else:
            rho = torch.ones_like(logit)
            log_rho = torch.zeros_like(logit)

        if self.mode == "mean":
            pooled = r.mean(dim=1, keepdim=True).expand_as(r)
            return pooled, rho, None

        a_hat = self.adjacency()
        bias = self.beta * torch.log(a_hat + self.eps)[None, None] + log_rho[:, None, None, :]
        upd, attn = self.attn(r, r, bias=bias, return_attn=True)
        z = self.norm1(r + upd)
        o = self.norm2(z + self.ff(z))
        return o, rho, attn
