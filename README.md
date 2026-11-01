# DAEF-Net

PyTorch implementation of DAEF-Net, the physics-informed dual-attention evidential fusion network described in

> Z. Shi, "Physics-informed attention and evidential fusion enable trustworthy power quality disturbance recognition from heterogeneous sensors."

DAEF-Net classifies power quality disturbances observed by several sensors that differ in sampling rate, clock accuracy and reliability, and returns an uncertainty for every decision. Attention-I encodes each sensor with a selective state-space model, a transient-adaptive temporal attention and a harmonic-structured spectral attention. Attention-II exchanges information between sensors with graph cross-attention biased by the Thevenin electrical distance and a learned reliability score. The same score discounts each sensor's Dirichlet evidence before reduced Dempster combination.

## Repository layout

```
configs/                 per-dataset settings on top of the defaults of Table 5
daefnet/
  config.py              defaults and command-line overrides
  data/
    classes.py           class lists of D1 to D4
    signal_model.py      disturbance model, Eq. (2), parameter ranges of Table 4
    sensors.py           sensor emulation, Eq. (3)
    features.py          tokenization, Eqs. (4), (6), (7), and quality descriptors q
    network.py           feeder model and Thevenin electrical distance, Eq. (19)
    synthetic.py         generators for D2 and the D4-style multi-bus set
    io.py                event file format
    dataset.py           PyTorch dataset and collate function
  model/
    ssm.py               selective state-space encoder, Eqs. (8)-(11)
    attention_i.py       Attention-I, Eqs. (12)-(18)
    attention_ii.py      Attention-II, Eqs. (20)-(24)
    evidential.py        discounting and reduced Dempster fusion, Eqs. (25)-(29)
    daefnet.py           full network and continuous-time encoding, Eq. (5)
  losses.py              training objective, Eqs. (30)-(33)
  metrics.py             accuracy, macro-F1, kappa, ECE, NLL, Brier, AURC, OOD AUROC
  engine.py              training and inference loops
scripts/                 data preparation, training, evaluation and robustness tests
tests/                   unit tests, including a numerical check of Proposition 1
```

## Installation

```bash
git clone <repository-url> DAEF-Net
cd DAEF-Net
pip install -r requirements.txt
```

The selective scan in `daefnet/model/ssm.py` is written in plain PyTorch and runs on CPU or GPU. If the `mamba-ssm` package is installed, its fused CUDA kernel is used automatically, which is faster and uses much less memory. Set `model.use_cuda_kernel=false` to force the reference scan.

## Data

All sources are converted into one event format (`daefnet/data/io.py`): each split is an `.npz` file holding, for every sensor, the samples, the sampling or frame rate and optional timestamps, together with the labels and a JSON header with the nominal frequency, window length in cycles, class names, sensor kinds and an optional electrical distance matrix. Waveforms are in per unit. Phasor streams carry magnitude (pu), angle (rad), frequency deviation (Hz) and ROCOF (Hz/s).

### D1: SEED-PQD-v1

Download `SEED_PQD_v1.zip` from https://zenodo.org/records/11843312, unzip it and run

```bash
python scripts/prepare_d1.py --src SEED_PQD_v1 --out data/d1
```

The script reads the MATLAB file or the 17 CSV files and makes a stratified 70/10/20 split.

### D2: compound disturbances seen by three emulated sensors

```bash
python scripts/generate_d2.py --out data/d2
```

Events follow the signal model of Eq. (2) with the parameter ranges of Table 4 and the 29 classes listed in `daefnet/data/classes.py`. Each event is observed by S1 (12.8 kHz, 16 bit), S2 (3.2 kHz, 12 bit, second-order anti-aliasing filter) and S3 (phasors at 100 frames/s from a one-cycle DFT) with clock offsets drawn from U[-50, 50] us. Train, validation and test splits use disjoint seeds. Test sets for the asynchrony and frequency-deviation experiments are produced from the same seed:

```bash
python scripts/generate_d2.py --out data/d2_offset5ms --splits test --clock-offset-ms 5 --offset-sensors 1 2
python scripts/generate_d2.py --out data/d2_49p5hz --splits test --freq 49.5
```

### D3: Grid Event Signature Library

GESL records (https://gesl.ornl.gov) must be exported by the user into the event format, with three sensors per event: three-phase voltage point-on-wave (`kind: voltage`), three-phase current point-on-wave (`kind: current`) and the PMU stream (`kind: phasor`). Windows cover 20 cycles, 6 before and 14 after the trigger, and timestamps of all sensors are expressed in seconds from the window start.

```python
from daefnet.data.classes import D3_CLASSES
from daefnet.data.io import save_events

meta = {
    "name": "D3", "f0": 60.0, "window_cycles": 20, "classes": list(D3_CLASSES),
    "sensors": [{"name": "voltage", "kind": "voltage"},
                {"name": "current", "kind": "current"},
                {"name": "pmu", "kind": "phasor"}],
    "distance": None,
}
sensors = [
    {"x": voltage_windows, "fs": voltage_rates},   # lists of (3, L_i) arrays and rates
    {"x": current_windows, "fs": current_rates},
    {"x": pmu_windows, "fs": pmu_rates, "t": pmu_times},
]
save_events("data/d3/train.npz", meta, labels, sensors)
```

Lengths may differ between events because the point-on-wave rate depends on the provider. For the leave-one-provider-out protocol, write one directory per held-out provider.

### D4: multi-bus circuit with injected disturbances

```bash
python scripts/generate_d4.py --out data/d4
```

Without further arguments the generator uses a ten-bus demo feeder (`Feeder.demo()`) with eight metered buses and synthetic backgrounds. For the SoCal 28-bus data set (https://github.com/caltech-netlab/digital-twin-dataset), export the branch and load parameters to CSV, select the metered buses and supply background snapshots split chronologically; see the docstring of `scripts/generate_d4.py` for the expected formats. Disturbances are injected at a random metered bus and propagated with the transfer ratio Z_nk(f) / Z_kk(f) of the reduced nodal impedance matrix. The electrical distances of Eq. (19) between metered buses are stored with the data and used by Attention-II.

## Training

```bash
python scripts/train.py --config configs/d2.yaml --data data/d2 --out runs/d2_s0 --seed 0
```

The run directory receives the resolved configuration, a per-epoch log, the checkpoint with the best validation macro-F1 and the test metrics. Any setting can be changed from the command line, for example `--set train.batch_size=32 model.lambda_h=0.5`. `scripts/run_d2_d4.sh` runs data generation, five seeds and the robustness tests for D2 and D4.

## Evaluation

```bash
# accuracy, macro-F1, kappa, worst-class F1, ECE (15 bins), NLL, Brier score and AURC
python scripts/evaluate.py --ckpt runs/d2_s0/best.pt --data data/d2/test.npz

# test-time noise
python scripts/evaluate.py --ckpt runs/d2_s0/best.pt --data data/d2/test.npz --test-snr 20

# clock offset on S2 and S3
python scripts/evaluate.py --ckpt runs/d2_s0/best.pt --data data/d2_offset5ms/test.npz

# sensor failure (flat zero signal) and corruption (5 dB noise), averaged over sensors and sensor pairs
python scripts/robustness.py --ckpt runs/d4_s0/best.pt --data data/d4/test.npz

# out-of-distribution detection and the three-tier policy
python scripts/train.py --config configs/d3.yaml --data data/d3 --out runs/d3_ood \
    --exclude-classes incipient_arcing_fault oscillation
python scripts/evaluate.py --ckpt runs/d3_ood/best.pt --data data/d3/test.npz --ood --tiers 0.15 0.40

# few-shot adaptation from a pretrained checkpoint with 5 % of the labels
python scripts/train.py --config configs/d3.yaml --data data/d3 --out runs/d3_ft5 \
    --init runs/d2_s0/best.pt --train-frac 0.05
```

Fine-tuning with `--init` copies every tensor whose name and shape match. The shared encoder and Attention-II are transferred; sensor-specific token embeddings and evidential heads whose shapes differ are trained from scratch.

## Ablations

The variants of Table 9 are selected with overrides:

| Variant | Override |
|---|---|
| Without transient bias G | `model.use_transient_bias=false` |
| Selective SSM replaced by Transformer encoder | `model.encoder=transformer` |
| Without harmonic prior | `model.use_harmonic_prior=false` |
| Without time-frequency cross-attention | `model.use_cross_attention=false` |
| Without Attention-II (mean pooling) | `model.attention_ii=mean` |
| Uniform sensor graph | `model.graph=uniform` |
| Without reliability score | `model.use_reliability=false` |
| Without discounting | `model.use_discounting=false` |
| Softmax head instead of evidential fusion | `model.head=softmax` |
| Without sensor-dropout loss | `train.lambda_sd=0` |
| Without contrastive loss | `train.lambda_con=0` |

The sensitivity study of Table 10 uses `model.lambda_h`, `model.beta`, `train.p_drop`, `train.lambda_con` and `train.lambda_sd`.

## Implementation notes

Some details are not fixed by the equations of the paper; this code makes the following choices. Spectral tokens receive a learned band-index embedding in addition to the sensor-type embedding, so that the spectral attention knows which harmonic band each token represents. The SNR descriptor is computed from the residual of a cycle-wise least-squares fit of the fundamental and harmonics up to the seventh order. Phasor channels are scaled by (1, 1, 0.2, 0.002) and clipped to +/-5 before tokenization. Dropout of 0.1 is applied inside attention and feed-forward layers, gradients are clipped at norm 1.0 and encoder blocks use activation checkpointing during training. Sensor dropout in Eq. (32) is drawn independently for every event of a batch, with at least one sensor kept.

## Tests

```bash
pytest -q
```

`tests/test_fusion.py` checks numerically that a sensor with zero reliability is the neutral element of the fusion rule and that its influence vanishes continuously (Proposition 1), and that the rule is commutative and associative. `tests/test_model.py` runs a forward and backward pass of a small model on generated data.

## Citation

```bibtex
@article{shi_daefnet,
  author = {Shi, Zhan},
  title  = {Physics-informed attention and evidential fusion enable trustworthy power quality disturbance recognition from heterogeneous sensors},
  year   = {2026}
}
```

## License

MIT, see `LICENSE`. The data sets are distributed by their owners under their own licenses.
