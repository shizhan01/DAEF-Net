"""Training objective (Section 2.6).

L = L_ev(alpha, y) + 1/M sum_m L_ev(alpha^(m), y) + lambda_1 L_con + lambda_2 L_sd
"""

import torch
import torch.nn.functional as F


def kl_to_uniform_dirichlet(alpha):
    """KL( Dir(p | alpha) || Dir(p | 1) )."""
    k = alpha.shape[-1]
    s = alpha.sum(dim=-1)
    return (
        torch.lgamma(s)
        - torch.lgamma(alpha).sum(dim=-1)
        - torch.lgamma(torch.tensor(float(k), device=alpha.device))
        + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(s).unsqueeze(-1))).sum(dim=-1)
    )


def evidential_loss(alpha, y, lam_t):
    """Eq. (30): expected cross-entropy plus annealed KL on misleading evidence."""
    y1 = F.one_hot(y, alpha.shape[-1]).to(alpha.dtype)
    s = alpha.sum(dim=-1, keepdim=True)
    ce = (y1 * (torch.digamma(s) - torch.digamma(alpha))).sum(dim=-1)
    alpha_tilde = y1 + (1.0 - y1) * alpha
    return (ce + lam_t * kl_to_uniform_dirichlet(alpha_tilde)).mean()


def cross_sensor_infonce(r, temperature):
    """Eq. (31): embeddings of the same event agree across sensors."""
    b, m, _ = r.shape
    z = F.normalize(r, dim=-1)
    target = torch.arange(b, device=r.device)
    total, count = r.new_zeros(()), 0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            logits = z[:, i] @ z[:, j].t() / temperature
            total = total + F.cross_entropy(logits, target)
            count += 1
    return total / max(count, 1)


def sensor_dropout_loss(prob_full, prob_drop, eps=1e-8):
    """Eq. (32): KL( sg[p] || p_D )."""
    p = prob_full.detach()
    return (p * (torch.log(p + eps) - torch.log(prob_drop + eps))).sum(dim=-1).mean()


def sample_drop_mask(batch, n_sensors, p_drop, device):
    """Drop each sensor with probability p_drop, keeping at least one per event."""
    drop = torch.rand(batch, n_sensors, device=device) < p_drop
    all_dropped = drop.all(dim=1)
    if all_dropped.any():
        keep = torch.randint(n_sensors, (int(all_dropped.sum()),), device=device)
        rows = torch.nonzero(all_dropped).squeeze(1)
        drop[rows, keep] = False
    return drop
