from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
_PROJ_DIR = _SRC_DIR.parent
for _p in (_PROJ_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.data.dataloader import TACFDataModule
from src.models.tacf import TACF
from src.losses.calibrator import build_total_loss
from src.training import (
    Trainer,
    TrainerConfig,
    seed_everything,
    run_stage4_finetune,
)
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_metrics


ABLATIONS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("full", {}),
    ("no_consensus", dict(consensus_n_rounds=0)),
    ("no_ivw", dict(consensus_use_ivw=False)),
    ("only_trend", dict(only_one_specialist="trend")),
    ("no_decomp", dict(skip_decomposer=True)),
)


def _build_ablation_model(dm, variant_override: Dict[str, Any]) -> TACF:
    direct_kwargs: Dict[str, Any] = {
        "in_channels": dm.n_features,
        "out_channels": dm.n_features,
        "seq_len": dm.seq_len,
        "pred_len": dm.pred_len,
    }
    model_kwargs = {k: v for k, v in variant_override.items()
                    if k not in ("skip_decomposer", "only_one_specialist")}
    model = TACF(**direct_kwargs, **model_kwargs)
    if variant_override.get("skip_decomposer"):
        model.decomposer.requires_grad_(False)
        for p in model.decomposer.parameters():
            p.data.zero_()
    only = variant_override.get("only_one_specialist")
    if only in ("trend", "cycle", "local"):
        for name, agent in model.specialists.agents.items():
            if name != only:
                agent.requires_grad_(False)
                for p in agent.parameters():
                    p.data.zero_()
    return model


def run_one(
    name: str,
    variant_override: Dict[str, Any],
    dm: TACFDataModule,
    logger: ExperimentLogger,
    cfg: TrainerConfig,
) -> Dict[str, Any]:
    seed_everything(cfg.__dict__.get("seed", 42))
    model = _build_ablation_model(dm, variant_override=variant_override)
    print(f"\n=== Ablation: {name}  params={model.n_params:,}")
    sub_logger = ExperimentLogger(log_dir=str(Path(logger.log_dir) / name))
    sub_logger.log_config({"variant": variant_override})
    result = run_stage4_finetune(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        test_loader=dm.test_dataloader(),
        cfg=cfg,
        logger=sub_logger,
        dm=dm,
    )
    return {k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in result.items() if isinstance(v, (int, float, str, dict, list))}


def main(argv=None) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser("TACF ablation study")
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--root", type=str, default=str(_PROJ_DIR / "preprocess"))
    parser.add_argument("--log-dir", type=str, default=str(_PROJ_DIR / "logs/ablation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    dm = TACFDataModule(
        root=args.root, dataset=args.dataset,
        seq_len=args.seq_len, pred_len=args.pred_len, label_len=args.seq_len // 2,
        batch_size=args.batch_size, num_workers=0,
    )
    print(dm)
    cfg = TrainerConfig(
        max_epochs=args.max_epochs,
        early_stop=5, lr=args.lr, amp=False,
        lambda_agent=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    logger = ExperimentLogger(log_dir=args.log_dir)
    logger.log_config({"args": vars(args)})

    results: Dict[str, Any] = {}
    for name, var in ABLATIONS:
        results[name] = run_one(name=name, variant_override=dict(var),
                                dm=dm, logger=logger, cfg=cfg)

    with open(Path(logger.log_dir) / "ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    main()
