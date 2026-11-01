import numpy as np

from daefnet.data.features import quality_descriptors, tokenize_sensor
from daefnet.data.network import DEMO_METERED_BUSES, Feeder
from daefnet.data.synthetic import generate_d2


def test_d2_token_shapes():
    meta, y, sensors = generate_d2(1, seed=0, classes=("harmonics",))
    expected = {"S1": (159, 32), "S2": (159, 32), "S3": (19, 8)}
    for m, s in enumerate(sensors):
        x = s["x"][0].astype(float)
        fs = s["fs"][0]
        t = np.arange(x.shape[-1]) / fs
        tok = tokenize_sensor(x, t, fs, meta["f0"], meta["sensors"][m]["kind"], meta["window_cycles"])
        assert tok["tok_t"].shape == expected[meta["sensors"][m]["name"]]
        assert tok["tok_f"].shape[0] == 50


def test_harmonic_bands():
    meta, y, sensors = generate_d2(1, seed=1, classes=("harmonics",))
    x = sensors[0]["x"][0].astype(float)
    t = np.arange(x.shape[-1]) / 12800.0
    tok = tokenize_sensor(x, t, 12800.0, 50.0, "voltage", 10)
    energy = np.abs(tok["tok_f"]).sum(axis=1)
    strongest = set(tok["order_f"][np.argsort(energy)[-4:]].tolist())
    assert strongest == {1.0, 3.0, 5.0, 7.0}


def test_zero_signal_quality():
    t = np.arange(640) / 3200.0
    q = quality_descriptors(np.zeros((1, 640)), t, 3200.0, 50.0, "voltage")
    np.testing.assert_allclose(q, [0.0, 1.0, 0.0, 0.0], atol=1e-6)
    q = quality_descriptors(np.zeros((4, 20)), np.arange(1, 21) / 100.0, 100.0, 50.0, "phasor")
    np.testing.assert_allclose(q, [0.0, 1.0, 0.0, 0.0], atol=1e-6)


def test_electrical_distance():
    d = Feeder.demo().electrical_distance(DEMO_METERED_BUSES)
    np.testing.assert_allclose(d, d.T)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)
    assert (d[~np.eye(len(d), dtype=bool)] > 0).all()
