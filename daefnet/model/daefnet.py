"""DAEF-Net: physics-informed dual-attention evidential fusion network."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_i import AttentionI
from .attention_ii import AttentionII
from .evidential import fuse


class ContinuousTimeEncoding(nn.Module):
    """phi(tau) = [sin(nu_k tau), cos(nu_k tau)] with learnable nu_k (Eq. 5).

    tau is in seconds, so sensors with different rates share one time axis.
    """

    def __init__(self, d_model, f_min=1.0, f_max=2000.0):
        super().__init__()
        freqs = 2.0 * math.pi * torch.logspace(math.log10(f_min), math.log10(f_max), d_model // 2)
        self.log_nu = nn.Parameter(torch.log(freqs))

    def forward(self, tau):
        arg = tau.unsqueeze(-1) * torch.exp(self.log_nu)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


def _pad_tokens(x, n):
    if x.shape[1] == n:
        return x
    pad = x.new_zeros(x.shape[0], n - x.shape[1], *x.shape[2:])
    return torch.cat([x, pad], dim=1)


class DAEFNet(nn.Module):
    def __init__(self, specs, num_classes, cfg, distance=None):
        """
        Args:
            specs: list with one dict per sensor: kind id, dt_in, df_in, n_spec
                (see `MultiSensorDataset.specs`).
            num_classes: K.
            cfg: model configuration (see configs/*.yaml).
            distance: optional (M, M) electrical distance matrix of Eq. (19).
        """
        super().__init__()
        d = cfg["d_model"]
        self.num_sensors = len(specs)
        self.num_classes = num_classes
        self.head_type = cfg.get("head", "evidential")
        self.use_discounting = cfg.get("use_discounting", True)

        self.embed_t = nn.ModuleList([nn.Linear(s["dt_in"], d) for s in specs])
        self.embed_f = nn.ModuleList([nn.Linear(s["df_in"], d) for s in specs])
        self.kind_emb = nn.Embedding(4, d)
        self.band_emb = nn.Embedding(max(s["n_spec"] for s in specs), d)
        self.time_enc = ContinuousTimeEncoding(d, cfg.get("time_f_min", 1.0), cfg.get("time_f_max", 2000.0))
        self.register_buffer("kind_ids", torch.tensor([s["kind"] for s in specs], dtype=torch.long))

        self.encoder = AttentionI(cfg)
        self.fusion = AttentionII(
            d,
            self.num_sensors,
            heads=cfg["heads_ii"],
            beta=cfg["beta"],
            mu=cfg["mu"],
            distance=distance,
            graph=cfg.get("graph", "physical"),
            use_reliability=cfg.get("use_reliability", True),
            mode=cfg.get("attention_ii", "attention"),
            dropout=cfg["dropout"],
        )

        def mlp():
            return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(d, num_classes))

        if self.head_type == "evidential":
            self.heads = nn.ModuleList([mlp() for _ in range(self.num_sensors)])
        else:
            self.classifier = mlp()

    def encode(self, sensors, drop=None):
        """Embed and encode every sensor; returns r (B, M, d) and q (B, M, 4).

        drop: optional (B, M) bool. A dropped sensor is replaced by a flat zero
        signal, whose tokens are zero and whose quality vector is
        [0, 1, jitter, 0] (see daefnet.data.features).
        """
        hts, mts, hfs, mfs, orders, priors, qs = [], [], [], [], [], [], []
        for m, s in enumerate(sensors):
            tok_t, tok_f, q = s["tok_t"], s["tok_f"], s["q"]
            if drop is not None:
                dm = drop[:, m]
                keep = (~dm).to(tok_t.dtype)
                tok_t = tok_t * keep[:, None, None]
                tok_f = tok_f * keep[:, None, None]
                zero, one = torch.zeros_like(q[:, 0]), torch.ones_like(q[:, 0])
                q_zero = torch.stack([zero, one, q[:, 2], zero], dim=-1)
                q = torch.where(dm[:, None], q_zero, q)
            kind = self.kind_emb(self.kind_ids[m])
            band = torch.arange(tok_f.shape[1], device=tok_f.device)
            hts.append(self.embed_t[m](tok_t) + self.time_enc(s["tc"]) + kind)
            hfs.append(self.embed_f[m](tok_f) + self.band_emb(band) + kind)
            mts.append(s["mask_t"])
            mfs.append(s["mask_f"])
            orders.append(s["order_f"])
            priors.append(s["prior_on"])
            qs.append(q)

        b = hts[0].shape[0]
        n_t = max(h.shape[1] for h in hts)
        n_f = max(h.shape[1] for h in hfs)
        ht = torch.stack([_pad_tokens(h, n_t) for h in hts], dim=1).flatten(0, 1)
        mt = torch.stack([_pad_tokens(x, n_t) for x in mts], dim=1).flatten(0, 1)
        hf = torch.stack([_pad_tokens(h, n_f) for h in hfs], dim=1).flatten(0, 1)
        mf = torch.stack([_pad_tokens(x, n_f) for x in mfs], dim=1).flatten(0, 1)
        order = torch.stack([_pad_tokens(x, n_f) for x in orders], dim=1).flatten(0, 1)
        prior = torch.stack(priors, dim=1).flatten(0, 1)

        r, aux = self.encoder(ht, mt, hf, mf, order, prior)
        r = r.view(b, self.num_sensors, -1)
        return r, torch.stack(qs, dim=1), aux

    def forward(self, sensors, drop=None):
        r, q, aux = self.encode(sensors, drop)
        o, rho, attn = self.fusion(r, q)
        out = {"r": r, "rho": rho, "attn": attn, "sigma": aux["sigma"], "gate": aux["gate"]}
        if self.head_type == "evidential":
            alpha_m = torch.stack([F.softplus(head(o[:, m])) + 1.0 for m, head in enumerate(self.heads)], dim=1)
            alpha, belief, u = fuse(alpha_m, rho, self.use_discounting)
            out.update(alpha=alpha, alpha_m=alpha_m, belief=belief, u=u.squeeze(-1),
                       prob=alpha / alpha.sum(dim=-1, keepdim=True))
        else:
            logits = self.classifier(o.mean(dim=1))
            prob = torch.softmax(logits, dim=-1)
            out.update(logits=logits, prob=prob, u=1.0 - prob.max(dim=-1).values)
        return out
