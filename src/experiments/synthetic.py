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


CONTAM_RATES = (0.0, 0.05, 0.1, 0.25)
OUTLIER_MAG = 10.0


def _inject_outliers(y: torch.Tensor, frac: float, mag: float = OUTLIER_MAG) -> torch.Tensor:
    if frac <= 0:
        return y
    mask = torch.rand_like(y) < frac
    signs = torch.randint(0, 2, size=y.shape, dtype=y.dtype, device=y.device) * 2 - 1
    y_out = y.clone()
    y_out[mask] = y_out[mask] + mag * signs[mask] * y_out[mask].std()
    return y_out, mask


def main(argv=None) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser("Synthetic-contamination reject-signal validation")
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--root", default=str(_PROJ_DIR / "preprocess"))
    parser.add_argument("--log-dir", default=str(_PROJ_DIR / "logs/synthetic"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=15)
    args = parser.parse_args(argv)
    seed_everything(42)
    dm = TACFDataModule(
        root=args.root, dataset=args.dataset, seq_len=args.seq_len,
        pred_len=args.pred_len, batch_size=args.batch_size, num_workers=0,
    )
    model = TACF(
        in_channels=dm.n_features, out_channels=dm.n_features,
        seq_len=dm.seq_len, pred_len=dm.pred_len,
        specialist_d_model=256, specialist_n_layers=2, d_hidden=256,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = ExperimentLogger(log_dir=args.log_dir)
    logger.log_config({"args": vars(args)})
    cfg = TrainerConfig(max_epochs=args.max_epochs, early_stop=5, device=device, amp=False, lr=8e-4)
    run_stage4_finetune(
        model=model, train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(), test_loader=dm.test_dataloader(),
        cfg=cfg, logger=logger, dm=dm,
    )
    # Evaluate reject signals under synthetic contamination of test batches.
    results: Dict[str, Any] = {}
    for frac in CONTAM_RATES:
        alpha_list, reject_list, eff_list = [], [], []
        with torch.no_grad():
            for batch in dm.test_dataloader():
                x, xs, y, ys = [t.to(device) for t in batch[:4]]
                yc, _ = _inject_outliers(y, frac=frac)
                out = model(x, xs)
                alpha_list.append(out.alpha.cpu().numpy())
                reject_list.append(out.reject.cpu().numpy())
                eff_list.append(out.effective_weights.cpu().numpy())
        alpha = np.concatenate(alpha_list, axis=0)
        reject = np.concatenate(reject_list, axis=0)
        eff = np.concatenate(eff_list, axis=0)
        results[f"frac={frac}"] = {
            "alpha_mean": alpha.mean(axis=0).tolist(),
            "reject_mean": reject.mean(axis=0).tolist(),
            "eff_weights_mean": eff.mean(axis=0).tolist(),
        }
    with open(logger.log_dir / "synthetic.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
