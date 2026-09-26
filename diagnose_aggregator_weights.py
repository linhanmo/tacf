#!/usr/bin/env python3
"""
Diagnostic for aggregator alpha / reject (r) statistics over the TEST split.

Iterates over all ``logs/tacf_*/`` experiment directories; for each run it:
  1. reads dataset + seq/pred/label from ``config.json``
  2. rebuilds the matching ``TACFDataModule`` and ``TACF`` model
  3. loads ``checkpoints/final_best.pt`` (or ``checkpoints/best.pt`` as fallback)
  4. runs the full test_loader in eval mode, collecting per-sample
        alpha ∈ R^{B×3},  reject ∈ R^{B×3},  eff_w ∈ R^{B×3},
        y_hat, sigma, y
  5. prints a summary markdown table plus a machine-readable JSON dump.

Usage:
    python diagnose_aggregator_weights.py
    python diagnose_aggregator_weights.py --runs logs/tacf_exchange_rate_20260910_091234
    python diagnose_aggregator_weights.py --max-batches 30   # quick smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import TACFDataModule
from src.experiments.main import build_model_from_dm
from src.utils.metrics import quantile_coverage_probability


AGENT_ORDER = ["trend", "cycle", "local"]


@dataclass
class RunStats:
    run_dir: str
    dataset: str
    D: int
    seq_len: int
    pred_len: int
    batch_size_used: int
    ckpt_used: str
    sigma_global_multiplier_value: float
    test_samples: int = 0
    test_batches: int = 0
    # Alpha
    alpha_mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    alpha_std: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    alpha_entropy_mean: float = 0.0
    alpha_kl_uniform_mean: float = 0.0
    alpha_any_dominant_pct: float = 0.0  # % samples where any alpha_k > 0.6
    # Reject r
    reject_mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    reject_std: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    reject_gt_05_pct: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    reject_any_gt_05_pct: float = 0.0
    reject_all_low_pct: float = 0.0  # all r_k <= 0.1
    # Effective weights eff = alpha * (1 - r)
    eff_mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Prediction diagnostics (sanity check calibration after sigma-mult fix)
    q95_coverage: float = 0.0
    q95_width_mean: float = 0.0
    test_mse: float = 0.0
    test_nll: float = 0.0
    test_corr: float = 0.0


def _load_state_dict_into_model(model: torch.nn.Module, ckpt: Path):
    """Try common state-dict layouts."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    for top in ("state_dict", "model_state_dict", "model"):
        if isinstance(sd, dict) and top in sd:
            sd = sd[top]
            break
    # strip distributed DDP prefix
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def _entropy(p: torch.Tensor, dim=-1, eps=1e-8) -> torch.Tensor:
    # p: (*, K) over K-simplex
    return -(p * (p + eps).log()).sum(dim=dim)


def _kl_uniform(p: torch.Tensor, dim=-1, eps=1e-8) -> torch.Tensor:
    K = p.shape[dim]
    u = torch.full_like(p, 1.0 / K)
    return (p * ((p + eps).log() - (u + eps).log())).sum(dim=dim)


def _gaussian_nll_batch(mu, sigma, y, eps=1e-6):
    sigma = sigma.clamp(min=eps)
    var = sigma.pow(2)
    return (0.5 * (torch.log(2 * math.pi * var) + (mu - y).pow(2) / var)).mean().item()


def _corr(x, y):
    xm = x - x.mean()
    ym = y - y.mean()
    n = (xm * ym).sum().item()
    d = math.sqrt((xm.pow(2).sum().item() * ym.pow(2).sum().item())) + 1e-12
    return n / d


def diagnose_one(
    run_dir: Path,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> RunStats:
    cfg = json.load((run_dir / "config.json").open())
    args = cfg["args"]
    stats_meta = cfg["stats"]
    dataset = args["dataset"]

    # --- rebuild DM
    dm = TACFDataModule(
        root=args["root"],
        dataset=dataset,
        seq_len=int(args["seq_len"]),
        pred_len=int(args["pred_len"]),
        label_len=int(args.get("label_len") or int(args["pred_len"])),
        batch_size=int(args.get("batch_size") or 32),
        num_workers=0,
        pin_memory=False,
        use_time_features=True,
    )
    _, _, test_loader = dm.dataloaders()

    # --- rebuild model (use d_model override = args params)
    override: Dict[str, Any] = {}
    for k in ("d_model", "n_layers", "agg_d_model"):
        if k in args and args[k] is not None:
            if k == "n_layers":
                override["specialist_n_layers"] = int(args[k])
            elif k == "d_model":
                override["specialist_d_model"] = int(args[k])
            elif k == "agg_d_model":
                override["aggregator_d_model"] = int(args[k])
    if "sigma_global_multiplier_init" in args and args["sigma_global_multiplier_init"] is not None:
        override["sigma_global_multiplier_init"] = float(args["sigma_global_multiplier_init"])
    model = build_model_from_dm(dm, **override)
    ckpt = run_dir / "checkpoints" / "final_best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "checkpoints" / "best.pt"
    missing, unexpected = _load_state_dict_into_model(model, ckpt)
    # sigma multiplier value (learnable scalar)
    mult_val = float(
        getattr(model, "sigma_global_multiplier", torch.tensor(1.0)).detach().cpu().item()
    )
    model = model.to(device).eval()

    stats = RunStats(
        run_dir=str(run_dir), dataset=dataset,
        D=stats_meta["n_features"],
        seq_len=dm.seq_len, pred_len=dm.pred_len,
        batch_size_used=test_loader.batch_size or 0,
        ckpt_used=str(ckpt),
        sigma_global_multiplier_value=mult_val,
    )

    # Accumulators
    N = 0
    alpha_sum = torch.zeros(3, dtype=torch.float64)
    alpha_sq_sum = torch.zeros(3, dtype=torch.float64)
    alpha_H_sum = 0.0
    alpha_KL_sum = 0.0
    alpha_dom = 0
    rej_sum = torch.zeros(3, dtype=torch.float64)
    rej_sq_sum = torch.zeros(3, dtype=torch.float64)
    rej_gt_05 = torch.zeros(3, dtype=torch.float64)
    rej_any_gt_05 = 0
    rej_all_low = 0
    eff_sum = torch.zeros(3, dtype=torch.float64)
    # metrics
    sum_sq = 0.0
    sum_abs = 0.0
    sum_nll = 0.0
    cov_q95_sum = 0.0
    width_q95_sum = 0.0
    yh_all = []
    y_all = []

    z95 = math.sqrt(2.0) * torch.special.erfinv(torch.tensor(0.95, dtype=torch.float64)).item()

    with torch.no_grad():
        for bi, batch in enumerate(test_loader, 1):
            if max_batches is not None and bi > max_batches:
                break
            x, x_stamp, y, y_stamp = [t.to(device) for t in batch[:4]]
            out = model(x, x_stamp)
            # ---- aggregator stats
            al = out.alpha.double().cpu()    # (B, 3)
            rj = out.reject.double().cpu()   # (B, 3)
            ef = (al * (1.0 - rj))
            ef = ef / (ef.sum(-1, keepdim=True) + 1e-8)
            B = al.shape[0]
            N += B
            alpha_sum += al.sum(dim=0)
            alpha_sq_sum += (al.pow(2)).sum(dim=0)
            alpha_H_sum += float(_entropy(al, dim=-1).sum().item())
            alpha_KL_sum += float(_kl_uniform(al, dim=-1).sum().item())
            alpha_dom += int((al.max(dim=-1).values > 0.6).sum().item())
            rej_sum += rj.sum(dim=0)
            rej_sq_sum += (rj.pow(2)).sum(dim=0)
            rej_gt_05 += (rj > 0.5).sum(dim=0)
            rej_any_gt_05 += int((rj > 0.5).any(dim=-1).sum().item())
            rej_all_low += int(((rj <= 0.1).all(dim=-1)).sum().item())
            eff_sum += ef.sum(dim=0)
            # ---- forecasts metrics
            yh = out.y_hat.detach()
            sg = out.sigma.detach().clamp(min=1e-6)
            yh_all.append(yh.cpu().double())
            y_all.append(y.cpu().double())
            se = (yh - y).pow(2).mean().item()
            ae = (yh - y).abs().mean().item()
            nll = _gaussian_nll_batch(yh, sg, y)
            w = (sg * z95 * 2.0).mean().item()
            # empirical coverage with true sigma interval ± z·σ
            covered = ((y >= yh - z95 * sg) & (y <= yh + z95 * sg)).float().mean().item()
            sum_sq += se * B
            sum_abs += ae * B
            sum_nll += nll * B
            cov_q95_sum += covered * B
            width_q95_sum += w * B
            stats.test_batches = bi

    Nf = float(max(1, N))
    stats.test_samples = N
    # alpha mean/std
    am = alpha_sum / Nf
    av = (alpha_sq_sum / Nf) - am.pow(2)
    av = av.clamp(min=0.0)
    stats.alpha_mean = [float(am[i]) for i in range(3)]
    stats.alpha_std = [float(av[i].sqrt()) for i in range(3)]
    stats.alpha_entropy_mean = alpha_H_sum / Nf
    stats.alpha_kl_uniform_mean = alpha_KL_sum / Nf
    stats.alpha_any_dominant_pct = 100.0 * alpha_dom / Nf
    # reject
    rm = rej_sum / Nf
    rv = (rej_sq_sum / Nf - rm.pow(2)).clamp(min=0.0)
    stats.reject_mean = [float(rm[i]) for i in range(3)]
    stats.reject_std = [float(rv[i].sqrt()) for i in range(3)]
    stats.reject_gt_05_pct = [100.0 * float(rej_gt_05[i]) / Nf for i in range(3)]
    stats.reject_any_gt_05_pct = 100.0 * rej_any_gt_05 / Nf
    stats.reject_all_low_pct = 100.0 * rej_all_low / Nf
    # eff
    em = eff_sum / Nf
    stats.eff_mean = [float(em[i]) for i in range(3)]
    # prediction metrics
    stats.test_mse = sum_sq / Nf
    stats.test_nll = sum_nll / Nf
    stats.q95_coverage = 100.0 * cov_q95_sum / Nf
    stats.q95_width_mean = width_q95_sum / Nf
    yh_cat = torch.cat(yh_all, dim=0).reshape(-1)
    y_cat = torch.cat(y_all, dim=0).reshape(-1)
    stats.test_corr = _corr(yh_cat, y_cat)
    return stats


def print_tables(runs: List[RunStats]):
    # --- Alpha table
    print("\n## α (softmax agent weights): per-agent mean ± std")
    print("| dataset | ckpt | σ_mult | N(test) | α_trend | α_cycle | α_local | "
          "⟨H(α)⟩ | KL(α‖U) | %any α_k>0.6 |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in runs:
        a = [f"{m:.3f}±{d:.3f}" for m, d in zip(s.alpha_mean, s.alpha_std)]
        print(f"| {s.dataset} | {Path(s.ckpt_used).name} | {s.sigma_global_multiplier_value:.3f} | "
              f"{s.test_samples} | {a[0]} | {a[1]} | {a[2]} | "
              f"{s.alpha_entropy_mean:.3f} | {s.alpha_kl_uniform_mean:.4f} | {s.alpha_any_dominant_pct:.1f}% |")

    # --- Reject r table
    print("\n## r (reject sigmoid): per-agent mean ± std, r_k>0.5 rate")
    print("| dataset | r_trend | r_cycle | r_local | any r>0.5% | all r≤0.1% |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for s in runs:
        r_cols = [
            f"{m:.3f}±{d:.3f} ({p:.1f}%)"
            for m, d, p in zip(s.reject_mean, s.reject_std, s.reject_gt_05_pct)
        ]
        print(f"| {s.dataset} | {r_cols[0]} | {r_cols[1]} | {r_cols[2]} | "
              f"{s.reject_any_gt_05_pct:.1f}% | {s.reject_all_low_pct:.1f}% |")

    # --- Effective weights (α⊙(1−r)) table
    print("\n## Effective w = α ⊙ (1 − r): per agent contribution after arbitration")
    print("| dataset | w_trend | w_cycle | w_local | ∑w≡1 check | comment |")
    print("| --- | ---: | ---: | ---: | :---: | --- |")
    for s in runs:
        total = sum(s.eff_mean)
        comment = "✅ well balanced"
        dom = max(s.eff_mean)
        if dom > 0.6:
            k = s.eff_mean.index(dom)
            comment = f"⚠️ dominated by {AGENT_ORDER[k]} ({dom:.1%})"
        elif dom < 0.4 and min(s.eff_mean) > 0.2:
            comment = "✅ truly mixed experts"
        if total < 0.98 or total > 1.02:
            comment += f"   ⚠ sum={total:.3f}≠1"
        print(f"| {s.dataset} | {s.eff_mean[0]:.3f} | {s.eff_mean[1]:.3f} | {s.eff_mean[2]:.3f} | "
              f"{'✅' if 0.98 <= total <= 1.02 else '⚠'} | {comment} |")

    # --- Calibration sanity after sigma multiplier + coverage_loss upgrades
    print("\n## Prediction / calibration sanity (on final_best or best ckpt)")
    print("| dataset | MSE | Corr | NLL | Q95 cov% vs 95% | Q95 width (2·z·σ) | gap |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in runs:
        gap = s.q95_coverage - 95.0
        if abs(gap) < 5.0:
            status = "✅"
        elif gap < 0:
            status = f"⚠️ under {gap:.1f}pp"
        else:
            status = f"ℹ️ over {gap:.1f}pp"
        print(f"| {s.dataset} | {s.test_mse:.4g} | {s.test_corr:.4f} | {s.test_nll:.2f} | "
              f"{s.q95_coverage:.1f}% | {s.q95_width_mean:.3f} | {status} |")


def main(argv=None) -> List[RunStats]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[],
                        help="Specific run dirs to analyze (default: all logs/tacf_*/)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit batches per run (for smoke/debug)")
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.runs:
        run_dirs = [Path(r).resolve() for r in args.runs]
    else:
        run_dirs = sorted([p.resolve() for p in (PROJECT_ROOT / "logs").glob("tacf_*") if p.is_dir()])

    all_stats: List[RunStats] = []
    for rd in run_dirs:
        if not (rd / "config.json").exists():
            continue
        if not any(((rd / "checkpoints" / f).exists()) for f in ("final_best.pt", "best.pt")):
            continue
        print(f"\n==== {rd.name} ====", flush=True)
        try:
            st = diagnose_one(rd, device, max_batches=args.max_batches)
            all_stats.append(st)
        except Exception as exc:
            print(f"  !! FAILED: {type(exc).__name__}: {exc}", flush=True)
            import traceback
            traceback.print_exc()

    if all_stats:
        print_tables(all_stats)
        # Machine readable dumps
        out_json = PROJECT_ROOT / "logs" / "aggregator_diagnostics.json"
        out_csv = PROJECT_ROOT / "logs" / "aggregator_diagnostics.csv"
        out_json.write_text(json.dumps([asdict(s) for s in all_stats], indent=2, default=str))
        with out_csv.open("w", newline="") as f:
            base = list(asdict(all_stats[0]).keys())
            w = csv.DictWriter(f, fieldnames=base)
            w.writeheader()
            for s in all_stats:
                row = {}
                for k, v in asdict(s).items():
                    if isinstance(v, list):
                        for i, vi in enumerate(v):
                            row[f"{k}_{AGENT_ORDER[i]}"] = vi
                    else:
                        row[k] = v
                # Drop original list-valued fields (already expanded)
                row = {k: v for k, v in row.items() if k in base or isinstance(v, (int, float, str, bool))}
                # Writer needs original flat fieldnames; just write asdict expanded flatten
                flat = {}
                for k, v in asdict(s).items():
                    if isinstance(v, list):
                        for i, vi in enumerate(v):
                            flat[f"{k}_{AGENT_ORDER[i]}"] = vi
                    else:
                        flat[k] = v
                # re-build fieldnames dynamically on first row
                pass
        # (Re)write with dynamic flat fieldnames for correctness
        fieldnames = []
        flat_rows = []
        for s in all_stats:
            flat = {}
            for k, v in asdict(s).items():
                if isinstance(v, list):
                    for i, vi in enumerate(v):
                        flat[f"{k}_{AGENT_ORDER[i]}"] = vi
                else:
                    flat[k] = v
            flat_rows.append(flat)
            fieldnames = list(flat.keys())
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in flat_rows:
                w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"\n💾 Diagnostic data written to:")
        print(f"   JSON -> {out_json}")
        print(f"   CSV  -> {out_csv}")
    return all_stats


if __name__ == "__main__":
    main()
