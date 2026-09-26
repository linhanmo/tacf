from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bimamba import BiMambaStack
from ..data.dataset import AGENT_NAMES
from ..utils._layers import DataEmbedding, RMSNorm


__all__ = [
    "AgentOutput",
    "AgentsOutput",
    "IndependentAgent",
    "AgentGroup",
]


@dataclass
class AgentOutput:
    """Prediction output of a single specialist."""

    mu: torch.Tensor
    sigma: torch.Tensor
    h: torch.Tensor
    h_seq: torch.Tensor


@dataclass
class AgentsOutput:
    outputs: Dict[str, AgentOutput]

    def __getitem__(self, key: str) -> AgentOutput:
        return self.outputs[key]

    @property
    def names(self) -> List[str]:
        return list(self.outputs.keys())

    def stack_mu(self) -> torch.Tensor:
        return torch.stack([self.outputs[n].mu for n in self.names], dim=1)

    def stack_sigma(self) -> torch.Tensor:
        return torch.stack([self.outputs[n].sigma for n in self.names], dim=1)

    def stack_h(self) -> torch.Tensor:
        return torch.stack([self.outputs[n].h for n in self.names], dim=1)

    def stack_h_seq(self) -> torch.Tensor:
        return torch.stack([self.outputs[n].h_seq for n in self.names], dim=1)


class IndependentAgent(nn.Module):
    """A single independent specialist agent with its own Bi-Mamba backbone.

    Pipeline
    --------
    X_k ∈ R^(B, T, D_in)
        -> DataEmbedding (value conv + sinusoial pos + token emb + optional stamp proj)
        -> BiMambaStack (n_layers of pre-norm BiMamba + SwiGLU FFN)
        -> Temporal downsampling Conv1D -> AdaptiveAvgPool1D(P) -> depthwise P-step mix
        -> head_mu (1×1 Conv) -> μ_k ∈ R^(B, P, D_out)
        -> head_sigma (1×1 Conv + Softplus + eps) -> σ_k ∈ R^(B, P, D_out)
        -> AvgPool(T) + MLP -> h_k ∈ R^(B, d_hidden)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        pred_len: int,
        d_hidden: int,
        d_model: int = 512,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        norm_eps: float = 1e-5,
        name: str = "agent",
    ) -> None:
        super().__init__()
        self.name = str(name)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d_hidden = int(d_hidden)
        self.d_model = int(d_model)

        self.embed = DataEmbedding(
            c_in=in_channels,
            d_model=d_model,
            dropout=dropout,
            max_len=max(5000, 2 * seq_len),
        )
        self.backbone = BiMambaStack(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bidirectional=bidirectional,
            norm_eps=norm_eps,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        d_mid = min(d_model, 256)
        self.time_down = nn.Sequential(
            nn.Conv1d(
                d_model, d_mid, kernel_size=3, padding=1, padding_mode="replicate"
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(self.pred_len),
        )
        self.time_proj = nn.Sequential(
            nn.Conv1d(
                self.pred_len,
                self.pred_len,
                kernel_size=3,
                padding=1,
                groups=self.pred_len,
            ),
            RMSNorm(d_mid, eps=norm_eps),
            nn.GELU(),
        )
        self.head_mu = nn.Conv1d(d_mid, out_channels, kernel_size=1)
        self.head_sigma = nn.Conv1d(d_mid, out_channels, kernel_size=1)
        self.head_h = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.min_sigma = 1e-4

    def forward(
        self,
        x: torch.Tensor,
        x_stamp: Optional[torch.Tensor] = None,
    ) -> AgentOutput:
        """
        x : (B, T, D_in)       component input window
        x_stamp : (B, T, S)    optional continuous time features

        Returns
        -------
        mu, sigma : (B, P, D_out)
        h : (B, d_hidden)
        h_seq : (B, T, d_model)   raw backbone sequence (used for auxiliary losses)
        """
        B, T, D = x.shape
        h_seq = self.embed(x, x_stamp)
        h_seq = self.backbone(h_seq)

        h_time = h_seq.transpose(1, 2).contiguous()
        down = self.time_down(h_time)
        proj = self.time_proj(down.transpose(1, 2)).transpose(1, 2)

        mu = self.head_mu(proj).transpose(1, 2).contiguous()
        sigma = (
            F.softplus(self.head_sigma(proj)).transpose(1, 2).contiguous()
            + self.min_sigma
        )

        pooled = self.pool(h_time).squeeze(-1)
        h = self.head_h(pooled)

        return AgentOutput(mu=mu, sigma=sigma, h=h, h_seq=h_seq)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AgentGroup(nn.Module):
    """Thin module-wrapper around the three independent specialists."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        pred_len: int,
        d_hidden: int,
        d_model: int = 512,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        norm_eps: float = 1e-5,
        share_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.agent_names = tuple(AGENT_NAMES)
        self.agents = nn.ModuleDict(
            {
                name: IndependentAgent(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    d_hidden=d_hidden,
                    d_model=d_model,
                    n_layers=n_layers,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    norm_eps=norm_eps,
                    name=name,
                )
                for name in self.agent_names
            }
        )
        if share_embeddings:
            for name in self.agent_names[1:]:
                self.agents[name].embed = self.agents[self.agent_names[0]].embed

    @property
    def n_agents(self) -> int:
        return len(self.agent_names)

    def forward(
        self,
        components: Dict[str, torch.Tensor],
        x_stamp: Optional[torch.Tensor] = None,
    ) -> AgentsOutput:
        out: Dict[str, AgentOutput] = {}
        for name in self.agent_names:
            out[name] = self.agents[name](components[name], x_stamp)
        return AgentsOutput(outputs=out)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def params_per_agent(self) -> Dict[str, int]:
        return {name: self.agents[name].n_params for name in self.agent_names}


SpecialistAgent = IndependentAgent
SpecialistGroup = AgentGroup
SpecialistOutput = AgentOutput
SpecialistsOutput = AgentsOutput
