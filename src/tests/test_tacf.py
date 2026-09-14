from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_PROJ_DIR = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJ_DIR / "src"
for _p in (_PROJ_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.models.tacf import TACF  # noqa: E402
from src.losses.calibrator import build_total_loss  # noqa: E402
from src.losses.agent_losses import build_agent_heterogeneous_losses  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402


def _small_tacf(D=7, T=96, P=48, d_hidden=128, d_model=128, n_layers=1) -> TACF:
    return TACF(
        in_channels=D, out_channels=D, seq_len=T, pred_len=P, d_hidden=d_hidden,
        specialist_d_model=d_model, specialist_n_layers=n_layers,
        aggregator_d_model=128, aggregator_n_layers=1,
        specialist_dropout=0.0, aggregator_dropout=0.0,
        consensus_n_rounds=1,
    )


def test_topology_matches_architecture_diagram():
    """Every tensor shape promised by Architecture.md must match."""
    B, T, D, P = 2, 96, 7, 48
    model = _small_tacf(D=D, T=T, P=P, d_hidden=64, d_model=64, n_layers=1)
    x = torch.randn(B, T, D)
    xs = torch.randn(B, T, 4)
    y = torch.randn(B, P, D)
    out = model(x, xs, y)
    # Decomposer 3 components (Architecture.md)
    assert out.decomposer_out.trend.shape == (B, T, D)
    assert out.decomposer_out.cycle.shape == (B, T, D)
    assert out.decomposer_out.local.shape == (B, T, D)
    # 3 specialists → each (mu, sigma, h)
    for name in ("trend", "cycle", "local"):
        spec = out.specialists_out[name]
        assert spec.mu.shape == (B, P, D)
        assert spec.sigma.shape == (B, P, D)
        assert spec.h.shape == (B, 64)
    # Consensus output shapes
    assert out.consensus_out.mu.shape == (B, 3, P, D)
    assert out.consensus_out.sigma.shape == (B, 3, P, D)
    assert out.consensus_out.h.shape == (B, 3, 64)
    # Aggregator outputs: ŷ, σ, α, r
    assert out.y_hat.shape == (B, P, D)
    assert out.sigma.shape == (B, P, D)
    assert out.alpha.shape == (B, 3)
    assert out.reject.shape == (B, 3)
    assert ((out.alpha.sum(dim=-1) - 1.0).abs() < 1e-5).all()
    assert (out.sigma >= 1e-4).all()
    assert torch.isfinite(out.y_hat).all()


def test_heterogeneous_losses_all_finite_and_sum():
    B, T, D, P = 2, 96, 5, 48
    model = _small_tacf(D=D, T=T, P=P, d_hidden=32, d_model=32, n_layers=1)
    x = torch.randn(B, T, D)
    y = torch.randn(B, P, D)
    out = model(x, None, y)
    hetero = build_agent_heterogeneous_losses(
        {k: out.specialists_out[k].mu for k in ("trend", "cycle", "local")}, y
    )
    for field in ("tv_trend", "seasonal_cycle", "sparse_local", "total"):
        v = getattr(hetero, field)
        assert torch.is_tensor(v) and torch.isfinite(v).all() and v.dim() == 0


def test_build_total_loss_end_to_end_gradients_finite():
    B, T, D, P = 2, 48, 3, 24
    torch.manual_seed(3)
    model = _small_tacf(D=D, T=T, P=P, d_hidden=32, d_model=32, n_layers=1)
    x = torch.randn(B, T, D)
    y = torch.randn(B, P, D)
    out = model(x, None, y)
    hetero = build_agent_heterogeneous_losses(
        {k: out.specialists_out[k].mu for k in ("trend", "cycle", "local")}, y
    )
    loss_obj = build_total_loss(
        out, y,
        lambda_nll=1.0, lambda_mse=1.0, lambda_consensus=0.1,
        lambda_reject=0.01, lambda_orthogonality=0.01, lambda_agent=0.05,
        agent_hetero=hetero.total,
    )
    loss_obj.total.backward()
    total_norm2 = 0.0
    n_grads = 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            total_norm2 += p.grad.detach().pow(2).sum().item()
            n_grads += 1
    assert total_norm2 > 0 and n_grads > 0, "no trainable parameter gradients"


def test_compute_metrics_runs_after_forward():
    B, T, D, P = 2, 48, 3, 24
    model = _small_tacf(D=D, T=T, P=P, d_hidden=32, d_model=32, n_layers=1)
    torch.manual_seed(4)
    model.eval()
    xs_list, ys_list, ss_list = [], [], []
    for _ in range(3):
        x = torch.randn(B, T, D)
        y = torch.randn(B, P, D)
        out = model(x)
        xs_list.append(out.y_hat.detach())
        ys_list.append(y)
        ss_list.append(out.sigma.detach())
    result = compute_metrics(xs_list, ys_list, ss_list)
    for field in ("mse", "mae", "rmse", "corr", "rse"):
        v = getattr(result, field)
        assert isinstance(v, float) and (v == v), f"NaN in metric {field}={v}"


def test_model_save_load_roundtrip(tmp_path: Path):
    B, T, D, P = 1, 24, 4, 12
    torch.manual_seed(5)
    m1 = _small_tacf(D=D, T=T, P=P, d_hidden=32, d_model=32, n_layers=1).eval()
    x = torch.randn(B, T, D)
    y1 = m1(x).y_hat.detach()
    ckpt = tmp_path / "roundtrip.pt"
    torch.save({"model_state_dict": m1.state_dict()}, ckpt)
    m2 = _small_tacf(D=D, T=T, P=P, d_hidden=32, d_model=32, n_layers=1).eval()
    state = torch.load(ckpt, map_location="cpu")
    m2.load_state_dict(state["model_state_dict"])
    y2 = m2(x).y_hat.detach()
    assert torch.allclose(y1, y2, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
