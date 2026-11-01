import torch

from daefnet.model.evidential import combine, discount, fuse, opinion


def random_alpha(b, k, scale=5.0):
    return 1.0 + scale * torch.rand(b, k, dtype=torch.float64)


def test_vacuous_opinion_is_neutral():
    """Proposition 1: a sensor with zero reliability does not change the fusion."""
    b2, u2 = opinion(random_alpha(8, 6))
    b1, u1 = discount(*opinion(random_alpha(8, 6)), torch.zeros(8, dtype=torch.float64))
    b, u = combine(b1, u1, b2, u2)
    torch.testing.assert_close(b, b2)
    torch.testing.assert_close(u, u2)


def test_masses_sum_to_one():
    b, u = combine(*opinion(random_alpha(8, 5)), *opinion(random_alpha(8, 5)))
    torch.testing.assert_close(b.sum(-1, keepdim=True) + u, torch.ones(8, 1, dtype=torch.float64))


def test_rule_is_commutative_and_associative():
    o = [opinion(random_alpha(4, 5)) for _ in range(3)]
    ab, ba = combine(*o[0], *o[1]), combine(*o[1], *o[0])
    torch.testing.assert_close(ab[0], ba[0])
    left = combine(*combine(*o[0], *o[1]), *o[2])
    right = combine(*o[0], *combine(*o[1], *o[2]))
    torch.testing.assert_close(left[0], right[0])
    torch.testing.assert_close(left[1], right[1])


def test_influence_vanishes_continuously():
    a = torch.stack([random_alpha(1, 4)[0], random_alpha(1, 4, 50.0)[0]]).unsqueeze(0)
    reference = fuse(a[:, 1:], torch.ones(1, 1, dtype=torch.float64))[1]
    previous = None
    for rho0 in [1e-1, 1e-2, 1e-3, 1e-4]:
        rho = torch.tensor([[rho0, 1.0]], dtype=torch.float64)
        gap = (fuse(a, rho)[1] - reference).abs().max().item()
        if previous is not None:
            assert gap < previous
        previous = gap
    assert previous < 1e-3


def test_single_sensor_keeps_prediction():
    a = random_alpha(16, 7)
    alpha, _, _ = fuse(a.unsqueeze(1), torch.full((16, 1), 0.3, dtype=torch.float64))
    assert torch.equal(alpha.argmax(-1), a.argmax(-1))
