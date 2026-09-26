from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PROJ_DIR = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJ_DIR / "src"
for _p in (_PROJ_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.models.agent import AgentGroup, IndependentAgent  # noqa: E402


def _make_input(B=2, T=96, D=7, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(B, T, D, device=device)
    xs = torch.randn(B, T, 4, device=device)
    return x, xs


def test_independent_agent_shapes_cpu():
    torch.manual_seed(0)
    B, T, D, P, d = 2, 96, 7, 48, 128
    agent = IndependentAgent(
        in_channels=D, out_channels=D, seq_len=T, pred_len=P, d_hidden=d,
        d_model=128, n_layers=1, dropout=0.0,
    )
    agent.eval()
    with torch.no_grad():
        x, xs = _make_input(B, T, D)
        out = agent(x, xs)
    assert out.mu.shape == (B, P, D)
    assert out.sigma.shape == (B, P, D)
    assert (out.sigma >= 1e-4).all()
    assert out.h.shape == (B, d)
    assert out.h_seq.shape == (B, T, 128)
    assert torch.isfinite(out.mu).all() and torch.isfinite(out.h).all()


def test_agent_group_three_outputs():
    torch.manual_seed(1)
    B, T, D, P, d = 2, 336, 7, 168, 256
    grp = AgentGroup(
        in_channels=D, out_channels=D, seq_len=T, pred_len=P, d_hidden=d,
        d_model=128, n_layers=2, dropout=0.0,
    )
    x_trend = torch.randn(B, T, D)
    x_cycle = torch.randn(B, T, D)
    x_local = torch.randn(B, T, D)
    out = grp({"trend": x_trend, "cycle": x_cycle, "local": x_local})
    for name in ("trend", "cycle", "local"):
        spec = out[name]
        assert spec.mu.shape == (B, P, D)
        assert spec.sigma.shape == (B, P, D)
    stack_mu = out.stack_mu()
    assert stack_mu.shape == (B, 3, P, D)
    assert out.stack_h().shape == (B, 3, d)


def test_agent_backward_finite():
    torch.manual_seed(2)
    B, T, D, P, d = 2, 48, 3, 24, 64
    agent = IndependentAgent(
        in_channels=D, out_channels=D, seq_len=T, pred_len=P, d_hidden=d,
        d_model=64, n_layers=1, dropout=0.0,
    )
    x, _ = _make_input(B, T, D)
    y = torch.randn_like(agent(x).mu)
    out = agent(x)
    loss = (out.mu - y).pow(2).mean() + out.sigma.log().mean()
    loss.backward()
    total_norm = 0.0
    for p in agent.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().pow(2).sum().item()
            assert torch.isfinite(p.grad).all(), "NaNs in gradient"
    assert total_norm > 0
