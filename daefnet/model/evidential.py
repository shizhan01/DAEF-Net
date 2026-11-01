"""Reliability-discounted evidential decision fusion (Section 2.5).

Evidence and Dirichlet parameters (Eq. 25):  e = softplus(MLP(o)), alpha = e + 1
Subjective opinion (Eq. 26):                 b_k = e_k / S, u = K / S
Shafer discounting (Eq. 27):                 b_hat = rho b, u_hat = 1 - rho + rho u
Reduced Dempster rule (Eq. 28):
    b_k = (b1_k b2_k + b1_k u2 + b2_k u1) / (1 - C),  u = u1 u2 / (1 - C),
    C = sum_{i != j} b1_i b2_j
Fused Dirichlet (Eq. 29):                    alpha_k = K / u * b_k + 1

Proposition 1: with rho = 0 the discounted opinion is (0, 1), which is the
neutral element of the rule.
"""


def opinion(alpha):
    """Belief masses (B, K) and uncertainty (B, 1) of a Dirichlet."""
    k = alpha.shape[-1]
    s = alpha.sum(dim=-1, keepdim=True)
    return (alpha - 1.0) / s, k / s


def discount(b, u, rho):
    rho = rho.unsqueeze(-1)
    return rho * b, 1.0 - rho + rho * u


def combine(b1, u1, b2, u2, eps=1e-8):
    conflict = b1.sum(-1, keepdim=True) * b2.sum(-1, keepdim=True) - (b1 * b2).sum(-1, keepdim=True)
    scale = 1.0 / (1.0 - conflict).clamp_min(eps)
    b = (b1 * b2 + b1 * u2 + b2 * u1) * scale
    u = u1 * u2 * scale
    return b, u


def fuse(alphas, rho=None, use_discounting=True):
    """Fuse per-sensor Dirichlet parameters.

    Args:
        alphas: (B, M, K) per-sensor Dirichlet parameters.
        rho: (B, M) reliability scores, required when use_discounting.
    Returns:
        fused alpha (B, K), belief (B, K), uncertainty (B, 1).
    """
    k = alphas.shape[-1]
    b, u = None, None
    for m in range(alphas.shape[1]):
        bm, um = opinion(alphas[:, m])
        if use_discounting:
            bm, um = discount(bm, um, rho[:, m])
        if b is None:
            b, u = bm, um
        else:
            b, u = combine(b, u, bm, um)
    alpha = k / u.clamp_min(1e-8) * b + 1.0
    return alpha, b, u
