#!/usr/bin/env python3
"""CLI entry-point for training TACF MOA on a preprocessed time-series dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tacf import ExperimentConfig
from tacf.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TACF MOA for time series forecasting.")
    p.add_argument("--dataset", type=str, default="ETTh1",
                   choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather",
                            "electricity", "traffic", "exchange_rate"])
    p.add_argument("--seq-len", type=int, default=336)
    p.add_argument("--pred-len", type=int, default=168,
                   help="prediction horizon (e.g., 96/192/336/720).")
    p.add_argument("--label-len", type=int, default=0,
                   help="start token length; 0 = auto seq_len//2.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-nll", type=float, default=1.0)
    p.add_argument("--lambda-mse", type=float, default=1.0)
    p.add_argument("--lambda-consensus", type=float, default=0.1)
    p.add_argument("--lambda-reject", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=512,
                   help="specialist Bi-Mamba hidden dim.")
    p.add_argument("--n-layers", type=int, default=4,
                   help="Bi-Mamba layers per specialist agent.")
    p.add_argument("--device", type=str, default=None,
                   help="'cuda' or 'cpu'. Falls back to train config.")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig.for_dataset(args.dataset, seq_len=args.seq_len, pred_len=args.pred_len)
    if args.label_len > 0:
        cfg.data.label_len = args.label_len
        cfg.moa.label_len = args.label_len
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers
    cfg.train.max_epochs = args.max_epochs
    cfg.train.early_stop_patience = args.early_stop
    cfg.train.lr = args.lr
    cfg.train.seed = args.seed
    cfg.train.use_amp = not args.no_amp
    cfg.train.lambda_nll = args.lambda_nll
    cfg.train.lambda_mse = args.lambda_mse
    cfg.train.lambda_consensus = args.lambda_consensus
    cfg.train.lambda_reject = args.lambda_reject
    cfg.moa.specialist.d_model = args.d_model
    cfg.moa.specialist.n_layers = args.n_layers

    name = (
        args.experiment_name
        or f"tacf_moa_{args.dataset}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}"
    )
    cfg.train.experiment_name = name

    print("Running with config:")
    print(json.dumps(cfg.to_dict(), indent=2, default=str)[:1500])
    print("...\n")

    trainer = Trainer(cfg=cfg, device=args.device)
    result = trainer.fit()
    print()
    print("=" * 70)
    print(result.summary())
    print("=" * 70)
    if result.best_checkpoint_path:
        print(f"Checkpoint saved: {result.best_checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
