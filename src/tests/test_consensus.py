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

from src.models.consensus import ConsensusLayer, MultiHeadAgentAttention  # noqa: E402


def _make_triplet(B=2, K=3, P=24, D=7, d=128):
    mu = torch.randn(B, K, P, D).abs() * 0.1
    sigma = torch.rand(B, K, P, D).clamp(min=0.05) + 0.05
    h = torch.randn(B, K, d)
    return mu, sigma, h


def test_multi_head_agent_attention_shapes():
    B, K, d = 2, 3, 64
    attn = MultiHeadAgentAttention(d=d, d_attn=128, num_heads=4, dropout=0.0)
    h = torch.randn(B, K, d)
    h_new, w = attn(h)
    assert h_new.shape == (B, K, d)
    assert w.shape == (B, K, K)
    assert ((w >= 0) & torch.isfinite(w)).all()
    row_sum = w.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5)


def test_consensus_layer_io_invariant_shapes():
    B, K, P, D, d = 2, 3, 24, 7, 128
    consensus = ConsensusLayer(
        d_hidden=d, out_channels=D, pred_len=P,
        d_attn=128, num_heads=4, n_rounds=2,
    )
    mu, sigma, h = _make_triplet(B, K, P, D, d)
    out = consensus(mu, sigma, h)
    assert out.mu.shape == mu.shape
    assert out.sigma.shape == sigma.shape
    assert (out.sigma >= 1e-4).all()
    assert out.h.shape == h.shape
    assert out.comm_weights.shape == (B, K, K)
    assert torch.isfinite(out.mu).all()
    assert torch.isfinite(out.sigma).all()


def test_consensus_ivw_monotonic_on_synthetic():
    """If one agent reports very low uncertainty it should dominate the fusion."""
    B, K, P, D, d = 1, 3, 4, 2, 16
    torch.manual_seed(0)
    mu = torch.zeros(B, K, P, D)
    mu[:, 0, :, :] = 5.0
    mu[:, 1, :, :] = 10.0
    mu[:, 2, :, :] = 0.0
    sigma = torch.full((B, K, P, D), 1.0)
    sigma[:, 0, :, :] = 1e-4
    h = torch.randn(B, K, d)
    consensus = ConsensusLayer(
        d_hidden=d, out_channels=D, pred_len=P,
        d_attn=64, num_heads=2, n_rounds=2,
    )
    out = consensus(mu, sigma, h)
    diff0 = (out.mu[:, 0, :, :] - 5.0).abs().mean().item()
    diff_others = (out.mu[:, 1:, :, :] - 5.0).abs().mean().item()
    assert diff0 < 3.0, f"dominating agent drifted too much: {diff0:.4f}"
    assert diff_others < 6.0, f"non-dominating agent didn't fuse: {diff_others:.4f}"


def test_consensus_backward():
    B, K, P, D, d = 2, 3, 24, 3, 32
    torch.manual_seed(0)
    cons = ConsensusLayer(d_hidden=d, out_channels=D, pred_len=P,
                          d_attn=64, num_heads=2, n_rounds=1)
    mu, sigma, h = _make_triplet(B, K, P, D, d)
    mu.requires_grad_(True)
    sigma.requires_grad_(True)
    h.requires_grad_(True)
    out = cons(mu, sigma, h)
    loss = out.mu.pow(2).mean() + out.sigma.log().mean() + out.h.pow(2).mean()
    loss.backward()
    for name, t in (("mu", mu), ("sigma", sigma), ("h", h)):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all(), f"NaNs in grad of {name}"
