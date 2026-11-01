import torch

from daefnet.config import load_config
from daefnet.data.dataset import MultiSensorDataset, collate
from daefnet.data.io import save_events
from daefnet.data.synthetic import generate_d2
from daefnet.engine import build_model, compute_loss


def test_forward_backward(tmp_path):
    classes = ("normal", "sag", "harmonics")
    meta, y, sensors = generate_d2(2, seed=0, classes=classes)
    path = tmp_path / "toy.npz"
    save_events(path, meta, y, sensors)
    data = MultiSensorDataset(path, train=True, noise_aug=True)
    batch = collate([data[i] for i in range(len(data))])

    cfg = load_config(None, ["model.d_model=32", "model.d_state=4", "model.n_ssm_blocks=1",
                             "model.n_attn_layers=1", "model.heads=2", "model.heads_ii=2",
                             "model.use_cuda_kernel=false", "model.checkpoint_encoder=false"])
    model = build_model(data.specs(), data.num_classes, cfg)
    loss, terms = compute_loss(model, batch["sensors"], batch["y"], cfg["train"], epoch=0)
    loss.backward()
    assert torch.isfinite(loss)
    assert {"main", "con", "sd"} <= set(terms)

    model.eval()
    with torch.no_grad():
        out = model(batch["sensors"])
    assert out["prob"].shape == (len(data), len(classes))
    assert out["rho"].shape == (len(data), 3)
    torch.testing.assert_close(out["prob"].sum(-1), torch.ones(len(data)))
