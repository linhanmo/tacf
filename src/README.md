# TACF Mixture-of-Agents (MOA) — `src` Layout

This folder implements the modular TACF model described in the repository-level
`/home/lin/tacf/Architecture.md` and follows the layout codified in
`/home/lin/tacf/code.md`.

## Directory tree

```
src/
├── README.md                       (this file)
├── requirements.txt                (PyTorch 2.4 + mamba_ssm 2.2.2 + deps)
├── configs/
│   ├── default.yaml                default experiment hyper-parameters
│   ├── ett.yaml                    ETT-family dataset overrides
│   └── electricity.yaml            electricity dataset overrides
├── data/
│   ├── dataset.py                  sliding-window TACFDataset
│   ├── dataloader.py               TACFDataModule + build_dataloaders()
│   └── freq_decomp.py              LearnableDecomposer + 3-branch Conv1D
├── models/
│   ├── __init__.py                 public re-exports for all models
│   ├── bimamba.py                  BiDirectionalMamba + BiMambaStack + CPU fallback
│   ├── agent.py                    IndependentAgent + AgentGroup (×3 specialists)
│   ├── consensus.py                2-round MHA + inverse-variance correction
│   ├── aggregator.py               LightMamba aggregator with α/r heads
│   └── tacf.py                     top-level TACF model assembly
├── losses/
│   ├── __init__.py
│   ├── agent_losses.py             TV / seasonal-Fourier / sparse-L1
│   └── calibrator.py               GaussianNLL / MixtureNLL / ECE + build_total_loss
├── training/
│   ├── __init__.py
│   ├── trainer.py                  Trainer + optimizer/scheduler/seed utils
│   ├── stage1_pretrain.py          Decomposer + 3 specialists (standalone)
│   ├── stage2_comm.py              Consensus (freeze rest)
│   ├── stage3_aggregator.py        Aggregator (freeze rest)
│   └── stage4_finetune.py          End-to-end fine-tune with low LR
├── experiments/
│   ├── configs/                    per-experiment YAML overrides
│   ├── main.py                     4-stage standard experiment runner
│   ├── ablation.py                 ablation sweep (no_consensus / no_ivw ...)
│   ├── robustness.py               noise + missing robustness sweep
│   └── synthetic.py                synthetic contamination reject-signal test
├── baselines/
│   ├── __init__.py                 4 baseline wrappers with common I/O
│   ├── mafs.py                     MAFS (TIP 2025) interface
│   ├── time_moe.py                 Time-MoE (ICLR 2025) interface
│   ├── m2fmoe.py                   M²FMoE interface
│   └── patchtst.py                 PatchTST (ICLR 2023) interface
├── utils/
│   ├── _layers.py                  RMSNorm / Conv1dSame / FFN / DataEmbedding (private)
│   ├── metrics.py                  MSE/MAE/RMSE/MAPE/CORR/RSE/Q95/NLL + aggregate
│   ├── logger.py                   ExperimentLogger + CSV/TB + ckpt helpers
│   └── visualization.py            forecast / components / heatmap plots
├── tests/
│   ├── test_agent.py               IndependentAgent + AgentGroup unit tests
│   ├── test_consensus.py           MHA attention + consensus gradient tests
│   └── test_tacf.py                end-to-end topology + loss + metric tests
└── scripts/
    ├── run_experiment.sh           bash entry for `python -m src.experiments.main`
    └── plot_results.py             forecast/components/heatmap post-run plots
```

## Architecture mapping to `Architecture.md`

Every block in the ASCII diagram is implemented one-to-one:

| Diagram block                           | Implementation                                                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Input `X ∈ R^(T×D)`                     | `src.data.dataset.TACFDataset` returns sliding-window batches.                                                            |
| **LearnableDecomposer** (3 Conv1D k=64/32/16) | `src.data.freq_decomp.LearnableDecomposer`  –  3 parallel `_ComponentBranch`  with DW Conv + SE gate, orthogonality loss. |
| 3 × Agent-Trend/Cycle/Local (Bi-Mamba θ) | `src.models.agent.AgentGroup` wraps 3 `IndependentAgent` each using `BiMambaStack` from `src.models.bimamba`.             |
| Output `{μ_k, σ_k, h_k}` × 3            | `AgentOutput.mu / .sigma / .h`.  Stacked via `AgentsOutput.stack_{mu,sigma,h}()`.                                          |
| **ConsensusLayer** (2-round + IVW)       | `src.models.consensus.ConsensusLayer` — `MultiHeadAgentAttention` on K=3 agent tokens → IVW-fused μ/σ, sigmoid-gated delta.  |
| **AggregatorAgent** LightMamba (≈0.5M)   | `src.models.aggregator.AggregatorAgent` — LightMamba on agent dim → α (softmax), r (sigmoid), fusion-scale → precision fusion. |
| Outputs `ŷ, σ, {α_k}, {r_k}`            | `TACFOutput.y_hat / .sigma / .alpha / .reject` plus `.effective_weights = α⊙(1−r)/‖α⊙(1−r)‖₁`.                               |

## Training stages (see also `/home/lin/tacf/training.md`)

1. **stage1_pretrain** → train `decomposer + specialists` only. Per-agent NLL + MSE
   + TV/seasonal-Fourier/sparse-L1 heterogeneous losses + decomposer orthogonality.
2. **stage2_comm** → freeze stage1, train `consensus` only. Consensus μ/σ NLL + MSE +
   communication-weight diversity.
3. **stage3_aggregator** → freeze 1+2, train `aggregator` only. Global NLL + MSE +
   reject regularisation.
4. **stage4_finetune** → unfreeze everything with 0.1× learning rate; optimise the
   full combined objective from `losses.calibrator.build_total_loss`.

Run the standard experiment with:

```bash
bash src/scripts/run_experiment.sh
```

or directly:

```bash
python -m src.experiments.main --dataset ETTh1 --seq-len 336 --pred-len 168
```

## Baselines

The `src.baselines.*` modules provide drop-in wrappers with the same
`forward(x, x_stamp=None) -> output.y_hat / output.sigma` signature so they can be
benchmarked against TACF with the same dataloaders and evaluation code.
Local lightweight approximations are used when the upstream reference package is not
installed in the current environment so the shape and test pipeline always runs.
