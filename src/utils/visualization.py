from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    _HAS_PLOT = True
except Exception:  # pragma: no cover - optional dep
    _HAS_PLOT = False


__all__ = [
    "plot_forecast",
    "plot_components",
    "plot_agent_weights_heatmap",
    "available",
]


def available() -> bool:
    return _HAS_PLOT


def _save(fig, out_dir: str | os.PathLike, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_forecast(
    history: np.ndarray | torch.Tensor,
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    sigma: Optional[np.ndarray | torch.Tensor] = None,
    feature_idx: int = 0,
    sample_idx: int = 0,
    out_dir: str | os.PathLike = "figures",
    name: str = "forecast.png",
) -> Optional[Path]:
    """Plot a single sample of ``history ‖ pred vs target`` with 95% CI band."""
    if not _HAS_PLOT:
        return None
    h = np.asarray(history[sample_idx, ..., feature_idx]).reshape(-1).astype(float)
    p = np.asarray(pred[sample_idx, ..., feature_idx]).reshape(-1).astype(float)
    t = np.asarray(target[sample_idx, ..., feature_idx]).reshape(-1).astype(float)
    t_hist = np.arange(len(h))
    t_fut = np.arange(len(h), len(h) + len(p))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_hist, h, color="k", lw=1.2, label="History")
    ax.plot(t_fut, t, color="tab:blue", lw=1.8, label="Target")
    ax.plot(t_fut, p, color="tab:red", lw=1.8, ls="--", label="Predicted")
    if sigma is not None:
        s = np.asarray(sigma[sample_idx, ..., feature_idx]).reshape(-1).astype(float)
        lo = p - 1.96 * s
        hi = p + 1.96 * s
        ax.fill_between(t_fut, lo, hi, alpha=0.2, color="tab:red", label="95% CI")
    ax.axvline(len(h) - 0.5, color="grey", ls=":", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Feature {feature_idx}")
    ax.set_title("TACF: history → forecast")
    ax.legend()
    ax.grid(alpha=0.2)
    return _save(fig, out_dir, name)


def plot_components(
    components: Dict[str, np.ndarray | torch.Tensor],
    feature_idx: int = 0,
    sample_idx: int = 0,
    out_dir: str | os.PathLike = "figures",
    name: str = "components.png",
) -> Optional[Path]:
    """Plot the three LearnableDecomposer outputs for a single feature/sample."""
    if not _HAS_PLOT:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, (k, v) in enumerate(components.items()):
        arr = np.asarray(v[sample_idx, ..., feature_idx]).reshape(-1).astype(float)
        axes[i].plot(arr, color=f"C{i}")
        axes[i].set_title(k)
        axes[i].grid(alpha=0.2)
    axes[-1].set_xlabel("Step")
    fig.suptitle("LearnableDecomposer outputs")
    return _save(fig, out_dir, name)


def plot_agent_weights_heatmap(
    weight_matrix: np.ndarray | torch.Tensor,
    agent_names: Sequence[str] = ("trend", "cycle", "local"),
    out_dir: str | os.PathLike = "figures",
    name: str = "weight_heatmap.png",
    title: str = "Consensus attention / Aggregator weights",
) -> Optional[Path]:
    if not _HAS_PLOT:
        return None
    w = np.asarray(weight_matrix).astype(float)
    if w.ndim == 3:
        w = w.mean(axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        sns.heatmap(
            w,
            ax=ax,
            xticklabels=list(agent_names),
            yticklabels=list(agent_names),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            square=True,
            cbar=True,
        )
    except Exception:
        ax.imshow(w, cmap="Blues")
    ax.set_title(title)
    return _save(fig, out_dir, name)
