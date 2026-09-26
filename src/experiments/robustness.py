from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
_PROJ_DIR = _SRC_DIR.parent
for _p in (_PROJ_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.data.dataloader import TACFDataModule
from src.models.tacf import TACF
from src.training import TrainerConfig, run_stage4_finetune, seed_everything
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_metrics


NOISE_STD_LEVELS = (0.0, 0.05, 0.1, 0.2)
MISSING_FRACS = (0.0, 0.1, 0.25, 0.5)
N_SEEDS = 5


def _apply_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return x + std * torch.randn_like(x)


def _apply_missing(x: torch.Tensor, frac: float) -> torch.Tensor:
    if frac <= 0:
        return x
    mask = torch.rand_like(x[..., 0:1]) > frac
    return x * mask


def _iter_noisy(loader, noise_std: float, missing_frac: float):
    batches: List[tuple] = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) >= 4:
                x, xs, y, ys = batch[0], batch[1], batch[2], batch[3]
            else:
                x, y = batch[0], batch[1]
                xs = torch.zeros(x.shape[0], x.shape[1], 0)
                ys = torch.zeros(y.shape[0], y.shape[1], 0)
            xn = _apply_noise(x, noise_std)
            xn = _apply_missing(xn, missing_frac)
            batches.append((xn, xs, y, ys))
    return batches


def _evaluate_iter(model, batches_iter, inverse_transform_fn, device):
    model.eval()
    preds, targets, sigmas = [], [], []
    with torch.no_grad():
        for batch in batches_iter:
            x, xs, y, ys = [t.to(device) for t in batch[:4]]
            out = model(x, xs)
            yh = inverse_transform_fn(out.y_hat.detach().cpu())
            sg = inverse_transform_fn(out.sigma.detach().cpu())
            yc = inverse_transform_fn(y.detach().cpu())
            preds.append(yh); sigmas.append(sg); targets.append(yc)
    return compute_metrics(preds, targets, sigmas).as_dict()


def _mean_std_dict(ds: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not ds:
        return {}
    keys = list(ds[0].keys())
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [d.get(k, float("nan")) for d in ds]
        arr = np.array(vals, dtype=float)
        out[k] = {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}
    return out


def main(argv=None) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser("TACF robustness tests: noise + missing")
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--root", default=str(_PROJ_DIR / "preprocess"))
    parser.add_argument("--log-dir", default=str(_PROJ_DIR / "logs/robustness"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=15)
    args = parser.parse_args(argv)

    results: Dict[str, Any] = {"noise": {}, "missing": {}}

    dm = TACFDataModule(
        root=args.root, dataset=args.dataset,
        seq_len=args.seq_len, pred_len=args.pred_len,
        batch_size=args.batch_size, num_workers=0,
    )
    seed_everything(0)
    model = TACF(
        in_channels=dm.n_features, out_channels=dm.n_features,
        seq_len=dm.seq_len, pred_len=dm.pred_len,
        specialist_d_model=256, specialist_n_layers=2, d_hidden=128,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = ExperimentLogger(log_dir=args.log_dir)
    logger.log_config({"args": vars(args)})
    cfg = TrainerConfig(max_epochs=args.max_epochs, early_stop=5, device=device, amp=False, lr=8e-4)
    run_stage4_finetune(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        test_loader=dm.test_dataloader(),
        cfg=cfg, logger=logger, dm=dm,
    )

    print("\n--- Noise sweep (train on clean, test with noise) ---")
    for std in NOISE_STD_LEVELS:
        per_seed: List[Dict[str, float]] = []
        for s in range(N_SEEDS):
            seed_everything(1000 + s)
            noisy_batches = _iter_noisy(dm.test_dataloader(), noise_std=std, missing_frac=0.0)
            metrics = _evaluate_iter(model, noisy_batches, dm.inverse_transform, device)
            per_seed.append(metrics)
        results["noise"][f"std={std}"] = _mean_std_dict(per_seed)
        print(f"  noise std={std}: {results['noise'][f'std={std}']}")

    print("\n--- Missing-value sweep (train on clean, test with missing) ---")
    for frac in MISSING_FRACS:
        per_seed: List[Dict[str, float]] = []
        for s in range(N_SEEDS):
            seed_everything(2000 + s)
            missing_batches = _iter_noisy(dm.test_dataloader(), noise_std=0.0, missing_frac=frac)
            metrics = _evaluate_iter(model, missing_batches, dm.inverse_transform, device)
            per_seed.append(metrics)
        results["missing"][f"frac={frac}"] = _mean_std_dict(per_seed)
        print(f"  missing frac={frac}: {results['missing'][f'frac={frac}']}")

    summary_path = Path(logger.log_dir) / "robustness.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {summary_path}")
    return results


if __name__ == "__main__":
    main()
