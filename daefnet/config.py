"""Configuration handling. Defaults follow Table 5."""

import copy

import yaml

DEFAULTS = {
    "model": {
        "d_model": 128,
        "d_state": 16,
        "expand": 2,
        "n_ssm_blocks": 2,
        "n_attn_layers": 2,
        "heads": 4,
        "heads_ii": 4,
        "sigma_min": 1.0,
        "sigma_max": 16.0,
        "gamma": 4.0,
        "band": 48,
        "lambda_h": 1.0,
        "chi": 0.05,
        "beta": 0.8,
        "mu": 0.0,
        "dropout": 0.1,
        "encoder": "mamba",
        "use_transient_bias": True,
        "use_harmonic_prior": True,
        "use_cross_attention": True,
        "attention_ii": "attention",
        "graph": "physical",
        "use_reliability": True,
        "use_discounting": True,
        "head": "evidential",
        "use_cuda_kernel": True,
        "checkpoint_encoder": True,
    },
    "data": {
        "n_points": 32,
        "max_order": 25,
        "noise_aug": False,
        "aug_prob": 0.5,
        "snr_range": [20.0, 50.0],
    },
    "train": {
        "batch_size": 64,
        "epochs": 120,
        "warmup_epochs": 5,
        "lr": 1.0e-3,
        "weight_decay": 0.05,
        "patience": 15,
        "p_drop": 0.3,
        "lambda_con": 0.1,
        "lambda_sd": 0.5,
        "temperature": 0.1,
        "anneal_epochs": 20,
        "grad_clip": 1.0,
        "num_workers": 4,
    },
}


def _merge(base, update):
    out = copy.deepcopy(base)
    for k, v in (update or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides=()):
    """Load a YAML file on top of DEFAULTS and apply "section.key=value" overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as fh:
            cfg = _merge(cfg, yaml.safe_load(fh))
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def save_config(cfg, path):
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
