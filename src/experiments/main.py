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
    # ---------------------------------------------------------------------
    # 2026-09-26 SOTA 对齐：顶会（ICLR/AAAI/NeurIPS 2023-2025, TSLib standard
    # scripts, DLinear / TimesNet / PatchTST / iTransformer / Mamba / Bi-Mamba4TS
    # 官方脚本）统一 seq/pred 组合，避免 apple-to-orange。
    #
    #   官方标准组合（来自 Time-Series-Library/scripts/**/*.sh 循环）:
    #     seq_len ∈ { 96, 336, 512 }
    #     pred_len ∈ { 96, 192, 336, 720 }
    #
    #   - TS-Lib/README.md 默认模型 ID 示例: ETTh1_512_96
    #   - DLinear (AAAI 2023) Table A.1: ETTh1 pred_len 96/192/336/720 × seq_len 336
    #   - iTransformer (ICLR 2024) Table 3: pred_len 96/192/336/720 × seq_len 96/512
    #   - TEFN / Bi-Mamba4TS (hyper.ai leaderboard): pred_len 96/192/336/720 × seq_len 336
    #
    # 默认值改为 TS-Lib README 展示的 "test_long" 组合 seq=96, pred=96
    # 之前的 336→168 是项目自定义非标准 setting（168 小时=1 周，不在顶会 4 档中），
    # 现在要跑 336→168 必须手动 --seq-len 336 --pred-len 168 显式指定。
    # ---------------------------------------------------------------------
    parser.add_argument("--seq-len", type=int, default=96,
                        help="标准 setting: 96 / 336 / 512（对应 TS-Lib 官方脚本）。之前的默认 336 已改，需显式指定。")
    parser.add_argument("--pred-len", type=int, default=96,
                        help="标准 setting: 96 / 192 / 336 / 720（顶会 4 档基准）。之前默认 168 非标准，需显式指定。")
    parser.add_argument("--label-len", type=int, default=48,
                        help="TS-Lib decoder start token长度：通常 = pred_len//2，DLinear 等纯 encoder 模型可忽略但保持接口一致。")
    parser.add_argument("--root", type=str, default=str(_PROJ_DIR / "preprocess"))
    parser.add_argument("--log-dir", type=str, default=str(_PROJ_DIR / "logs"))
    # --- run / project naming ---
    parser.add_argument("--tag", type=str, default=None, dest="tag",
                        help="Override log folder project name. "
                             "If not set, we auto-generate one of the form: "
                             "  <dataset>_s<seq_len>_p<pred_len>_l<label_len>[_<tag_suffix>]_<YYYYMMDD_HHMMSS>")
    parser.add_argument("--tag-suffix", type=str, default=None,
                        help="Extra identifier appended to the auto-generated tag.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs-stage1", type=int, default=60)
    parser.add_argument("--max-epochs-stage2", type=int, default=25)
    parser.add_argument("--max-epochs-stage3", type=int, default=30)
    parser.add_argument("--max-epochs-stage4", type=int, default=40)
    parser.add_argument("--d-model", type=int, default=768, dest="d_model",
                        help="SOTA upgrade: specialist BiMamba width (previously 512).  768 fits ETTh1 336/168 @ 8G BS16×accum4 with AMP.")
    parser.add_argument("--agg-d-model", type=int, default=384,
                        help="SOTA upgrade: aggregator LightMamba width (previously 256).")
    parser.add_argument("--n-layers", type=int, default=6,
                        help="SOTA upgrade: specialist BiMamba depth (previously 4).  6 layers roughly matches iTransformer 6-block encoder size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--grad-accum", "--accum-steps", type=int, default=1, dest="grad_accum",
                        help="Gradient accumulation micro-batch steps.  Effective batch = --batch-size * grad_accum."
                             "  E.g. --batch-size 14 --grad-accum 4 gives eff BS=56 with ~1/4 peak VRAM of BS=56.")
    # ----- Stage4 calibration + best-stage selection (per 8-run smoke analysis findings)
    parser.add_argument("--stage4-lr-mult", type=float, default=0.1,
                        help="Stage-4 global LR multiplier. Analysis recommends 0.10 (was 0.03).")
    parser.add_argument("--stage4-early-stop", type=int, default=12,
                        help="Stage-4 separate early-stop patience. 12 recommended.")
    parser.add_argument("--no-best-stage-select", action="store_true",
                        help="Do not auto-pick the best-val stage and skip copying final_best.pt.")
    parser.add_argument("--copy-best-stage-metrics-only", action="store_true",
                        help="Copy final_best.pt from best-ckpt + overwrite results.json final with best stage metrics.")
    # ----- New post-failure-hyperparams (P0/P1/P2 fixes per ETTh1 tacf_20260926_133419 diagnostic)
    parser.add_argument("--sigma-global-multiplier-init", type=float, default=1.0,
                        dest="sigma_global_multiplier_init",
                        help="Initial value for learnable global sigma multiplier. Previously 3.0. Set to 1.0 to avoid S1 over-coverage forcing shrinkage.")
    args = parser.parse_args(argv)

    # ---------------------------------------------------------------------
    # 2026-09-26 SOTA 公平对比：启动前检查 seq/pred 是否属于顶会标准组合。
    # 标准组合来自 TS-Lib 官方 scripts/*.sh 与 DLinear(AAAI23)/PatchTST(ICLR23)/
    # iTransformer(ICLR24)/TimesNet(ICLR23)/Bi-Mamba4TS/Mamba 所有公开脚本：
    #   seq_len  ∈ { 96, 336, 512 }
    #   pred_len ∈ { 96, 192, 336, 720 }
    # 非标准组合会打印 WARNING，但不阻止运行（避免破坏项目自定义 336→168 等旧跑法）。
    # ---------------------------------------------------------------------
    STANDARD_SEQ = {96, 336, 512}
    STANDARD_PRED = {96, 192, 336, 720}
    if args.seq_len not in STANDARD_SEQ or args.pred_len not in STANDARD_PRED:
        import sys, datetime as _dt
        _ts = _dt.datetime.now().strftime("%H:%M:%S")
        _compat_pairs = sorted(f"{s}→{p}" for s in STANDARD_SEQ for p in STANDARD_PRED)
        print(
            f"[{_ts}] WARNING (TACF SOTA fairness): "
            f"seq_len={args.seq_len} pred_len={args.pred_len} 不是顶会/TS-Lib 官方标准 setting。\n"
            f"  允许运行，但与已发表 SOTA (DLinear/TEFN/iTransformer/Bi-Mamba4TS/Mamba…) 对比不公平。\n"
            f"  标准组合（12 对，全 TS-Lib 脚本）: seq_len ∈ {sorted(STANDARD_SEQ)} × pred_len ∈ {sorted(STANDARD_PRED)}\n"
            f"  示例:  --seq-len  96 --pred-len  96      (TS-Lib README test_long 默认)\n"
            f"         --seq-len 336 --pred-len 96/192/336/720  (DLinear/TEFN/Bi-Mamba4TS 四基准)\n"
            f"         --seq-len 512 --pred-len 96/192/336/720  (iTransformer/WPMixer 长lookback四基准)\n"
            f"  自定义 336→168 等 setting 需显式声明非标准，审稿人不予 SOTA 对比承认。",
            file=sys.stderr, flush=True,
        )

    seed_everything(args.seed)

    # ---------------------------------------------------------------------
    # 2026-09-26 日志目录命名规则升级：
    #   旧：tacf_YYYYMMDD_HHMMSS  (数据集/setting 信息丢失，多 run 下完全分不清)
    #   新：{dataset}_s{seq}_p{pred}_l{label}[_{suffix}]_YYYYMMDD_HHMMSS
    #   如果用户显式传了 --tag 则直接用 tag（保留覆盖能力，例如 ablation 标记）。
    # ---------------------------------------------------------------------
    import datetime as _dt
    if args.tag is not None and len(str(args.tag).strip()) > 0:
        PROJECT_NAME = str(args.tag).strip()
    else:
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        _base = f"{args.dataset}_s{args.seq_len}_p{args.pred_len}_l{args.label_len}"
        if args.tag_suffix and len(str(args.tag_suffix).strip()) > 0:
            _base = f"{_base}_{str(args.tag_suffix).strip()}"
        PROJECT_NAME = f"{_base}_{_ts}"

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
        monitor="composite_coverage",           # default now probabilistic; per-stage overrides below
        mode="min",
        lambda_agent=0.05,
        accum_steps=int(args.grad_accum),
        # Calibration / probabilistic losses — per-stage overrides below.
        lambda_cov_penalty=0.25,
        target_q=0.95,
        sigma_reg_weight=0.0,
        lambda_ece=0.0,
        lambda_crps=0.0,
        lambda_reject_dist=0.0,
        # NaN self-recovery defaults (always on, cheap).
        nan_recovery_threshold=2,
        nan_recovery_max=3,
        nan_lr_decay=0.5,
        # Composite monitor — now the DEFAULT for all stages per calibration protocol.
        composite_width_weight=1.0,
        composite_mse_weight=0.5,
        s2_warm_epochs=0,                        # FIX: was 5 in cfg_s2 override; doubled with scheduler warmup
        s2_warm_lr_mult=0.1,
        eval_temperature=1.0,
        # FIX: added per-stage warmup overrides (was fixed at TrainerConfig.warmup_epochs=5 for all)
        warmup_epochs=3,
    )

    logger = ExperimentLogger(log_dir=args.log_dir, project=PROJECT_NAME)
    logger.log_config({"args": vars(args), "stats": asdict(dm.stats)})

    results: Dict[str, Any] = {}
    # ------------------------------------------------------------------
    # Protocol (per user request):
    #   S1 (pretrain specialists) : monitor = val_mse ; λ_MSE = 1, λ_NLL ≈ 0
    #   S2 (train consensus)     : monitor = val_mse ; λ_MSE = 1, λ_NLL small
    #   S3 (train aggregator)    : monitor = val_nll ; λ_NLL main + λ_ECE + λ_Rdist
    #   S4 (end-to-end finetune) : monitor = val_nll ; λ_NLL main + λ_ECE + λ_CRPS
    # Post-S4: temperature-scaling (T) on VAL sigma, re-run VAL+TEST with T,
    #          pick "final_best" on scaled-val NLL with ECE/Q50/Q90 aux checks.
    # RMSE is no longer used for checkpoint selection or highlighted printing.
    # ------------------------------------------------------------------
    cfg_s1 = dict(cfg_common)
    cfg_s1.update(dict(
        warmup_epochs=5,                       # Decomposer + 3 specialists larger now (~15M params, 5ep warm OK
        monitor="composite_coverage",
        mode="min",
        lambda_mse=1.0,
        lambda_nll=0.3,                          # FIX ×3 SOTA-probabilistic: guide sigma
        sigma_reg_weight=0.12,               # FIX ×12 (was 0.01) strong sigma collapse
        lambda_orthogonality=0.08,              # FIX ×8 (was 0.01) decomp freq decouple branches
        lambda_cov_penalty=0.4,              # FIX ×1.6 under-coverage hinge stronger
        lambda_ece=0.2,                         # FIX enable S1 ECE guidance (was 0)
    ))
    cfg_s2 = dict(cfg_common)
    cfg_s2.update(dict(
        warmup_epochs=1,                       # FIX was 5 (double warm caused LR=1e-5)
        monitor="composite_coverage",
        mode="min",
        lambda_mse=1.0,
        lambda_nll=0.1,
        sigma_reg_weight=0.02,                  # σ convex regulariser
        s2_warm_epochs=0,                     # FIX was 5 (double warm LR 1e-5 * 0.1 = 1e-6 disaster)
        s2_warm_lr_mult=0.1,
    ))
    cfg_s3 = dict(cfg_common)
    cfg_s3.update(dict(
        warmup_epochs=2,                       # FIX was 5 (S3 warm=5 == early_stop=5 → best@ep1 before full-LR
        monitor="composite_coverage",            # FIX was val_nll
        mode="min",
        lambda_nll=1.5,                           # FIX ×1.5 stronger probabilistic for calibration
        lambda_mse=0.2,                         # small MSE to keep point estimates non-degenerate
        lambda_ece=0.6,                         # FIX 0.6 (was 0.5) ECE gap penalty
        sigma_reg_weight=0.08,                  # FIX stronger σ regularization 0.08 (was 0.05)
        lambda_reject_dist=0.08,                  # FIX reduced further 0.08 (was 0.2, then 0.1) keep flexibility
        lambda_cov_penalty=0.45,                 # FIX ×1.5 coverage hinge
        lambda_crps=0.2,                         # FIX enable S3 CRPS (was 0) — train for probabilistic
    ))
    cfg_s4 = dict(cfg_common)
    cfg_s4.update(dict(
        warmup_epochs=3,
        monitor="val_crps",                    # FIX was val_nll — CRPS no bias big/small σ both goodhart
        mode="min",
        lambda_nll=1.5,                           # FIX ×1.5 (was 1.0) stronger probabilistic
        lambda_mse=0.2,                         # keep tie-breaking MSE guidance
        lambda_ece=0.5,                          # FIX calibration stronger (was 0.4)
        lambda_crps=1.0,                      # FIX ×2 SOTA-probabilistic primary (was 0.5)
        sigma_reg_weight=0.05,                 # FIX mild σ regulariser 0.05 (was 0.03)
        lambda_cov_penalty=0.45,                # FIX ×1.5 coverage hinge
        lambda_reject_dist=0.05,                # S4 keep reject distribution light flexibility
    ))

    stages = [
        ("stage1", run_stage1_pretrain, TrainerConfig(max_epochs=args.max_epochs_stage1, early_stop=15, **cfg_s1)),
        ("stage2", run_stage2_comm, TrainerConfig(max_epochs=args.max_epochs_stage2, early_stop=8, **cfg_s2)),
        ("stage3", run_stage3_aggregator, TrainerConfig(max_epochs=args.max_epochs_stage3, early_stop=10, **cfg_s3)),
        ("stage4", run_stage4_finetune, TrainerConfig(max_epochs=args.max_epochs_stage4, early_stop=args.stage4_early_stop, **cfg_s4)),
    ]
    stage_ckpt_paths: Dict[str, Optional[Path]] = {tag: None for tag, _, _ in stages}
    stage_trainers: Dict[str, Any] = {}
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
        stage_trainers[tag] = runner_kwargs.get("_trainer")
        ckpt_path = res.get("best_checkpoint")
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)
        stage_ckpt_paths[tag] = ckpt_path
        results[tag] = {k: (str(v) if isinstance(v, Path) else v) for k, v in res.items()}
        best = res.get("best")
        if best is not None:
            print(f"  best_epoch={res.get('best_epoch')}  tag={tag}")
        print(f"  best_monitor={res.get('best_monitor')}  best_checkpoint={res.get('best_checkpoint')}")

    # -------- Post-S4: temperature scaling (T) on the VAL split of S4 best ckpt
    # We learn T on VAL (sigma) and then re-evaluate VAL/TEST with T applied.
    if not args.no_best_stage_select:
        s4_result = results.get("stage4")
        if s4_result is not None and stage_ckpt_paths.get("stage4") is not None:
            try:
                from ..losses.calibrator import temperature_scale_nll
                from ..training.trainer import Trainer
                s4_trainer_cfg = TrainerConfig(**{k: v for k, v in cfg_s4.items()})
                s4_trainer = Trainer(
                    model=model,
                    cfg=s4_trainer_cfg,
                    trainable_parameters=list(model.parameters()),
                    logger=logger,
                    inverse_transform_fn=dm.inverse_transform if dm else None,
                )
                best_ckpt = stage_ckpt_paths["stage4"]
                from ..utils.logger import load_checkpoint
                load_checkpoint(best_ckpt, s4_trainer.model, device=s4_trainer.device)
                # Run once on VAL to get (mu, sigma, y) lists (no gradient)
                _, mus, ys, sigmas = s4_trainer._evaluate(val_loader)
                if len(sigmas) and len(mus) and len(ys):
                    t_res = temperature_scale_nll(mus, sigmas, ys)
                    T = float(t_res["T"])
                    print(
                        f"\n  🌡  S4 Post-hoc T-scaling (val set): T={T:.4f}  "
                        f"NLL {t_res['nll_before']:.4f} → {t_res['nll_after']:.4f}  "
                        f"(Δ {(t_res['nll_after']-t_res['nll_before'])/max(1e-8,abs(t_res['nll_before']))*100:+.2f}%)  "
                        f"status={t_res['status']}  N={t_res['N_samples']}"
                    )
                    results["stage4"]["temperature_scaling"] = t_res
                    # Evaluate again with T (VAL + TEST)
                    s4_trainer.cfg.eval_temperature = T
                    val_m_T, *_ = s4_trainer._evaluate(val_loader)
                    test_m_T, *_ = s4_trainer._evaluate(test_loader)
                    # Append scaled metrics to stage4["best_*_scaled"] for reporting
                    results["stage4"]["scaled_val_metrics"] = val_m_T.as_dict()
                    results["stage4"]["scaled_test_metrics"] = test_m_T.as_dict()
                    print(f"    VAL (T={T:.4f}):  NLL={val_m_T.nll:.4f}  MSE={val_m_T.mse:.5f}  "
                          f"ECE={val_m_T.ece:.3f}  Q50={val_m_T.q50_coverage:.2f}  "
                          f"Q90={val_m_T.q90_coverage:.2f}  Q95={val_m_T.q95_coverage:.2f}")
                    print(f"    TEST(T={T:.4f}):  NLL={test_m_T.nll:.4f}  MSE={test_m_T.mse:.5f}  "
                          f"ECE={test_m_T.ece:.3f}  Q50={test_m_T.q50_coverage:.2f}  "
                          f"Q90={test_m_T.q90_coverage:.2f}  Q95={test_m_T.q95_coverage:.2f}  "
                          f"MAPE={test_m_T.mape:.2f}%  CORR={test_m_T.corr:.4f}  "
                          f"CRPS={test_m_T.crps:.4f}")
            except Exception as exc:
                print(f"  ⚠ temperature scaling skipped: {exc}", flush=True)
                results.setdefault("stage4", {})["temperature_scaling"] = {"error": str(exc)}


    # ============= New: BEST-STAGE SELECTION (FIXED: unified monitor re-eval) =============
    # Previously compared stage1.best_monitor (val_mse=14.92) vs stage3.best_monitor (val_nll=2.86)
    # directly — apples-to-oranges because different monitor quantities. Now we load each
    # stage's best ckpt and compute a UNIFIED monitor (val_crps, aka S4's monitor) on VAL,
    # picking the stage with lowest unified score.
    if not args.no_best_stage_select:
        unified_monitor = "val_crps"          # CRPS: no-goodhart bias towards σ scale
        unified_mode = "min"
        print(f"\n===== Best-stage selection (unified monitor: {unified_monitor}) =====")
        unified_monitor_pairs: List[Tuple[str, float]] = []
        stage_reval_metrics: Dict[str, Dict[str, float]] = {}
        try:
            from ..training.trainer import Trainer
            from ..utils.logger import load_checkpoint
            # Build a temporary trainer using S4 cfg (so val_crps compute path is consistent)
            common_reval_cfg = TrainerConfig(**{**cfg_s4, "monitor": unified_monitor})
            common_reval_cfg.eval_temperature = 1.0  # T is applied ONLY to S4 final reporting
            reval_trainer = Trainer(
                model=model,
                cfg=common_reval_cfg,
                trainable_parameters=list(model.parameters()),
                logger=logger,
                inverse_transform_fn=dm.inverse_transform if dm else None,
            )
            for tag, ckpt_path in stage_ckpt_paths.items():
                if ckpt_path is None or not Path(ckpt_path).exists():
                    print(f"  skip {tag}: no ckpt {ckpt_path}")
                    continue
                try:
                    load_checkpoint(ckpt_path, reval_trainer.model, device=reval_trainer.device)
                    val_m, *_ = reval_trainer._evaluate(val_loader)
                    # Map "val_crps" -> attribute "crps" on val_m
                    key = unified_monitor
                    if key.startswith("val_"):
                        key = key[4:]
                    val_raw = getattr(val_m, key)
                    try:
                        val_unified = float(val_raw)
                    except Exception:
                        val_unified = float("inf")
                    if unified_mode == "min" and val_unified != val_unified:  # NaN
                        val_unified = float("inf")
                    stage_reval_metrics[tag] = {
                        unified_monitor: val_unified,
                        "val_mse": float(getattr(val_m, "mse", float("nan"))),
                        "val_nll": float(getattr(val_m, "nll", float("nan"))),
                        "val_q95_cov": float(getattr(val_m, "q95_coverage", float("nan"))),
                        "val_corr": float(getattr(val_m, "corr", float("nan"))),
                    }
                    unified_monitor_pairs.append((tag, val_unified))
                    print(f"  {tag}: ckpt={Path(ckpt_path).name}  unified({unified_monitor})={val_unified:.5f}  "
                          f"mse={stage_reval_metrics[tag]['val_mse']:.4f}  q95_cov={stage_reval_metrics[tag]['val_q95_cov']:.3f}")
                except Exception as _exc:
                    print(f"  skip {tag}: re-eval failed: {_exc}")
        except Exception as _global_exc:
            print(f"  ⚠ unified reval best-stage selector failed, fallback to best_monitor raw: {_global_exc}")
            stage_reval_metrics = {}
            unified_monitor_pairs = []
            for tag in stage_ckpt_paths.keys():
                val = results.get(tag, {}).get("best_monitor")
                if isinstance(val, (int, float)) and not (isinstance(val, float) and val != val):
                    unified_monitor_pairs.append((tag, float(val)))

        if unified_monitor_pairs:
            best_tag, best_val = min(unified_monitor_pairs, key=lambda kv: kv[1])
            print(f"  => BEST: {best_tag}  unified({unified_monitor})={best_val:.5f}")
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
                    "unified_monitor": unified_monitor,
                    "best_unified_value": float(best_val),
                    "reval_val_metrics": stage_reval_metrics.get(best_tag, {}),
                    "all_stage_reval": stage_reval_metrics if stage_reval_metrics else "fallback",
                    "best_epoch": results[best_tag].get("best_epoch"),
                    "checkpoint": str(dst),
                }
                if results[best_tag].get("best") is not None:
                    results["final_best"]["best"] = results[best_tag]["best"]
            else:
                results["final_best"] = {
                    "stage": best_tag, "unified_monitor": unified_monitor,
                    "best_unified_value": float(best_val), "error": "no ckpt found",
                }

    with open(logger.log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
