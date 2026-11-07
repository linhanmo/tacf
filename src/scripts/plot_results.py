from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJ_DIR = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJ_DIR / "src"
for _p in (_PROJ_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch

from src.utils.visualization import (
    available as plot_available,
    plot_agent_weights_heatmap,
    plot_components,
    plot_forecast,
)


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def _load_best_ckpt(ckpt_path: Path, model_cls, **model_kwargs):
    state = torch.load(ckpt_path, map_location="cpu")
    model = model_cls(**model_kwargs)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Plot results for a completed TACF run")
    parser.add_argument("--log-dir", type=Path, required=True,
                        help="Experiment log dir created by ExperimentLogger")
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--root", type=Path, default=_PROJ_DIR / "preprocess")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=168)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--feature-idx", type=int, default=0)
    parser.add_argument("--best-name", type=Path, default=Path("checkpoints/best.pt"))
    args = parser.parse_args(argv)

    if not plot_available():
        print("matplotlib/seaborn not available; skipping plotting.")
        return 0

    log_dir: Path = args.log_dir
    out_dir = log_dir / "figures"
    out_dir.mkdir(exist_ok=True, parents=True)

    results_path = log_dir / "results.json"
    if results_path.exists():
        data = _load_json(results_path)
        with open(out_dir / "metrics_summary.txt", "w") as f:
            f.write(json.dumps(data, indent=2, default=str))

    # Build model + run one sample visualisation if a checkpoint exists
    ckpt = args.best_name if args.best_name.is_absolute() else log_dir / args.best_name
    if not ckpt.exists():
        print(f"[warn] checkpoint not found at {ckpt}; skipping qualitative plots")
        return 0

    from src.data.dataloader import TACFDataModule
    from src.models.tacf import TACF

    dm = TACFDataModule(
        root=str(args.root), dataset=args.dataset,
        seq_len=args.seq_len, pred_len=args.pred_len,
        label_len=args.pred_len, batch_size=1, num_workers=0,
    )
    model = _load_best_ckpt(
        ckpt, TACF,
        in_channels=dm.n_features, out_channels=dm.n_features,
        seq_len=dm.seq_len, pred_len=dm.pred_len,
        specialist_d_model=512, specialist_n_layers=4, aggregator_d_model=256,
    )
    test_loader = dm.test_dataloader()
    batch = next(iter(test_loader))
    x, xs, y, ys = batch
    with torch.no_grad():
        out = model(x, xs)
    plot_forecast(
        history=x.cpu().numpy(), pred=out.y_hat.cpu().numpy(),
        target=y.cpu().numpy(), sigma=out.sigma.cpu().numpy(),
        feature_idx=args.feature_idx, sample_idx=args.sample_idx,
        out_dir=out_dir,
    )
    plot_components(
        {
            "trend": out.decomposer_out.trend.cpu().numpy(),
            "cycle": out.decomposer_out.cycle.cpu().numpy(),
            "local": out.decomposer_out.local.cpu().numpy(),
        },
        feature_idx=args.feature_idx, sample_idx=args.sample_idx,
        out_dir=out_dir,
    )
    plot_agent_weights_heatmap(
        out.consensus_out.comm_weights.cpu().numpy(),
        out_dir=out_dir, name="consensus_attn.png",
        title="Consensus: inter-agent attention (mean over batch)",
    )
    eff_w = out.effective_weights.cpu().numpy().reshape(-1, 1, 3)
    plot_agent_weights_heatmap(
        eff_w if eff_w.ndim == 3 else eff_w[:, None, :],
        out_dir=out_dir, name="agg_effective_weights.png",
        title="Aggregator: effective per-agent weights",
    )
    print(f"Saved plots to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
