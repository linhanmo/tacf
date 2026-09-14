from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure project root on path
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
_PROJ_DIR = _SRC_DIR.parent
for _p in (_PROJ_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.data.dataloader import TACFDataModule
from src.models.tacf import TACF
from src.training import (
    TrainerConfig,
    run_stage1_pretrain,
    run_stage2_comm,
    run_stage3_aggregator,
    run_stage4_finetune,
    seed_everything,
)
from src.utils.logger import ExperimentLogger


def build_model_from_dm(dm: TACFDataModule, **override: Any) -> TACF:
    return TACF(
        in_channels=dm.n_features,
        out_channels=dm.n_features,
        seq_len=dm.seq_len,
        pred_len=dm.pred_len,
        **override,
    )


def main(argv=None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="TACF main 4-stage experiment runner")
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=168)
    parser.add_argument("--label-len", type=int, default=168)
    parser.add_argument("--root", type=str, default=str(_PROJ_DIR / "preprocess"))
    parser.add_argument("--log-dir", type=str, default=str(_PROJ_DIR / "logs"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs-stage1", type=int, default=30)
    parser.add_argument("--max-epochs-stage2", type=int, default=10)
    parser.add_argument("--max-epochs-stage3", type=int, default=10)
    parser.add_argument("--max-epochs-stage4", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=512, dest="d_model")
    parser.add_argument("--agg-d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--grad-accum", "--accum-steps", type=int, default=1, dest="grad_accum",
                        help="Gradient accumulation micro-batch steps.  Effective batch = --batch-size * grad_accum."
                             "  E.g. --batch-size 14 --grad-accum 4 gives eff BS=56 with ~1/4 peak VRAM of BS=56.")
    # ----- Stage4 calibration + best-stage selection (per 8-run smoke analysis findings)
    parser.add_argument("--stage4-lr-mult", type=float, default=0.03,
                        help="Stage-4 global LR multiplier. Analysis recommends 0.03 (was 0.10).")
    parser.add_argument("--stage4-early-stop", type=int, default=5,
                        help="Stage-4 separate early-stop patience. 5 recommended.")
    parser.add_argument("--no-best-stage-select", action="store_true",
                        help="Do not auto-pick the best-val stage and skip copying final_best.pt.")
    parser.add_argument("--copy-best-stage-metrics-only", action="store_true",
                        help="Copy final_best.pt from best-ckpt + overwrite results.json final with best stage metrics.")
    args = parser.parse_args(argv)

    seed_everything(args.seed)

    dm = TACFDataModule(
        root=args.root,
        dataset=args.dataset,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        label_len=args.label_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        use_time_features=True,
    )
    print(dm)
    train_loader, val_loader, test_loader = dm.dataloaders()

    # ---- sigma_global_multiplier_init is pure TACF-init hyperparam (NOT Trainer).
    sigma_mult_init = float(getattr(args, "sigma_global_multiplier_init", 3.0))

    model_override = dict(
        specialist_d_model=args.d_model,
        specialist_n_layers=args.n_layers,
        aggregator_d_model=args.agg_d_model,
        sigma_global_multiplier_init=sigma_mult_init,
    )
    model = build_model_from_dm(dm, **model_override)
    print(model)

    cfg_common = dict(
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        amp=bool(args.amp and torch.cuda.is_available()),
        lr=args.lr,
        grad_clip=1.0,
        monitor="val_mse",
        mode="min",
        lambda_agent=0.05,
        # ---- Gradient accumulation.  TrainerConfig field is called ``accum_steps``.
        # Effective batch = DataLoader batch_size * accum_steps.  VRAM ∝ dataloader BS.
        accum_steps=int(args.grad_accum),
        # ---- σ calibration upgrades per 8-run smoke analysis ----
        lambda_nll=2.0,             # was 1.0; raise weight of NLL so σ becomes honest
        lambda_cov_penalty=0.25,    # per-train step direct coverage penalty
        target_q=0.95,              # desired q95 coverage
    )

    logger = ExperimentLogger(log_dir=args.log_dir)
    logger.log_config({"args": vars(args), "stats": asdict(dm.stats)})

    results: Dict[str, Any] = {}
    stages = [
        ("stage1", run_stage1_pretrain, TrainerConfig(max_epochs=args.max_epochs_stage1, early_stop=args.early_stop, **cfg_common)),
        ("stage2", run_stage2_comm, TrainerConfig(max_epochs=args.max_epochs_stage2, early_stop=max(3, args.early_stop//2), **cfg_common)),
        ("stage3", run_stage3_aggregator, TrainerConfig(max_epochs=args.max_epochs_stage3, early_stop=max(3, args.early_stop//2), **cfg_common)),
        ("stage4", run_stage4_finetune, TrainerConfig(max_epochs=args.max_epochs_stage4, early_stop=args.stage4_early_stop, **cfg_common)),
    ]
    stage_ckpt_paths: Dict[str, Optional[Path]] = {tag: None for tag, _, _ in stages}
    for tag, runner, trainer_cfg in stages:
        print(f"\n===== Running {tag} =====")
        runner_kwargs = dict(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            cfg=trainer_cfg,
            logger=logger,
            dm=dm,
        )
        if tag == "stage4":
            runner_kwargs["lr_mult"] = float(args.stage4_lr_mult)
            runner_kwargs["early_stop_override"] = int(args.stage4_early_stop)
        res = runner(**runner_kwargs)
        ckpt_path = res.get("best_checkpoint")
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)
        stage_ckpt_paths[tag] = ckpt_path
        results[tag] = {k: (str(v) if isinstance(v, Path) else v) for k, v in res.items()}
        best = res.get("best")
        if best is not None:
            print(f"  best_epoch={res.get('best_epoch')}  tag={tag}")
        print(f"  best_monitor={res.get('best_monitor')}  best_checkpoint={res.get('best_checkpoint')}")

    # ============= New: BEST-STAGE SELECTION =============
    # Pick the stage with lowest val monitor value (usually val_mse).
    # This fixes the 50%-regress problem in S4 without weakening any run.
    if not args.no_best_stage_select:
        monitor_pairs: List[Tuple[str, float]] = []
        for tag in stage_ckpt_paths.keys():
            val = results.get(tag, {}).get("best_monitor")
            if isinstance(val, (int, float)) and not (isinstance(val, float) and val != val):
                monitor_pairs.append((tag, float(val)))
        if monitor_pairs:
            best_tag, best_val = min(monitor_pairs, key=lambda kv: kv[1])
            print(f"\n===== Best-stage selection: {best_tag} (val_monitor={best_val:.5f}) =====")
            src_ckpt = stage_ckpt_paths.get(best_tag)
            if src_ckpt is not None and Path(src_ckpt).exists():
                dst_dir = logger.log_dir / "checkpoints"
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / "final_best.pt"
                import shutil
                shutil.copy2(src_ckpt, dst)
                print(f"  copied {src_ckpt} -> {dst}")
                results["final_best"] = {
                    "stage": best_tag,
                    "best_monitor": best_val,
                    "best_epoch": results[best_tag].get("best_epoch"),
                    "checkpoint": str(dst),
                }
                # Overwrite all "final" downstream references (best/test metrics from best-tag stage)
                if results[best_tag].get("best") is not None:
                    results["final_best"]["best"] = results[best_tag]["best"]
            else:
                results["final_best"] = {"stage": best_tag, "best_monitor": best_val, "error": "no ckpt found"}

    with open(logger.log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
