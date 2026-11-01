"""Training and inference loops."""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.dataset import MultiSensorDataset, collate
from .losses import (cross_sensor_infonce, evidential_loss, sample_drop_mask, sensor_dropout_loss)
from .metrics import calibration_metrics, classification_metrics
from .model.daefnet import DAEFNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    sensors = [{k: v.to(device, non_blocking=True) for k, v in s.items()} for s in batch["sensors"]]
    return sensors, batch["y"].to(device, non_blocking=True)


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0, drop_last=False)


def build_model(specs, num_classes, cfg, distance=None):
    return DAEFNet(specs, num_classes, cfg["model"], distance=distance)


def save_checkpoint(path, model, specs, classes, cfg, distance, epoch, score):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "specs": specs,
            "classes": classes,
            "config": cfg,
            "distance": None if distance is None else np.asarray(distance).tolist(),
            "epoch": epoch,
            "val_macro_f1": score,
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    dist = ckpt["distance"]
    model = build_model(ckpt["specs"], len(ckpt["classes"]), ckpt["config"],
                        None if dist is None else np.asarray(dist, dtype=np.float32))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def compute_loss(model, sensors, y, tcfg, epoch):
    out = model(sensors)
    terms = {}
    if model.head_type == "evidential":
        lam_t = min(1.0, (epoch + 1) / tcfg["anneal_epochs"])
        loss = evidential_loss(out["alpha"], y, lam_t)
        per_sensor = sum(evidential_loss(out["alpha_m"][:, m], y, lam_t) for m in range(model.num_sensors))
        loss = loss + per_sensor / model.num_sensors
    else:
        loss = F.cross_entropy(out["logits"], y)
    terms["main"] = float(loss.detach())

    if model.num_sensors > 1:
        if tcfg["lambda_con"] > 0:
            l_con = cross_sensor_infonce(out["r"], tcfg["temperature"])
            loss = loss + tcfg["lambda_con"] * l_con
            terms["con"] = float(l_con.detach())
        if tcfg["lambda_sd"] > 0:
            drop = sample_drop_mask(y.shape[0], model.num_sensors, tcfg["p_drop"], y.device)
            out_d = model(sensors, drop=drop)
            l_sd = sensor_dropout_loss(out["prob"], out_d["prob"])
            loss = loss + tcfg["lambda_sd"] * l_sd
            terms["sd"] = float(l_sd.detach())
    return loss, terms


def train_one_epoch(model, loader, optimizer, tcfg, epoch, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        sensors, y = to_device(batch, device)
        loss, _ = compute_loss(model, sensors, y, tcfg, epoch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if tcfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()
        total += float(loss.detach()) * y.shape[0]
        n += y.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, probs, us, rhos = [], [], [], []
    for batch in loader:
        sensors, y = to_device(batch, device)
        out = model(sensors)
        ys.append(y.cpu())
        probs.append(out["prob"].cpu())
        us.append(out["u"].cpu())
        rhos.append(out["rho"].cpu())
    return {
        "y": torch.cat(ys).numpy(),
        "prob": torch.cat(probs).double().numpy(),
        "u": torch.cat(us).double().numpy(),
        "rho": torch.cat(rhos).numpy(),
    }


def evaluate_predictions(pred):
    res = classification_metrics(pred["y"], pred["prob"])
    res.update(calibration_metrics(pred["y"], pred["prob"], pred["u"]))
    return res


def cosine_with_warmup(warmup, total):
    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def dataset_from_config(path, cfg, train=False, **kwargs):
    dcfg = cfg["data"]
    return MultiSensorDataset(
        path,
        train=train,
        noise_aug=train and dcfg["noise_aug"],
        aug_prob=dcfg["aug_prob"],
        snr_range=tuple(dcfg["snr_range"]),
        n_points=dcfg["n_points"],
        max_order=dcfg["max_order"],
        **kwargs,
    )
