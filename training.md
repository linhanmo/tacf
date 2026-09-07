# TACF 训练方法学 (四阶段渐进训练)

本文件描述 TACF 混合专家模型的完整训练流程。所有训练逻辑的代码位于
`src/training/`，包含通用 `Trainer` (`trainer.py`) 以及分阶段封装
`stage1_pretrain.py` / `stage2_comm.py` / `stage3_aggregator.py` /
`stage4_finetune.py`。主实验入口 `src/experiments/main.py` 会按顺序自动
调用这 4 个阶段。

---

## 0. 架构数据流回顾

参见 `Architecture.md` 的 ASCII 图。数据链路：

```
X ∈ R^(B,T,D)
  └─► LearnableDecomposer  (kernel 64/32/16 的 3 个 DWConv+SE+1x1 分支)
        └─► X_trend, X_cycle, X_local  ∈ R^(B,T,D) × 3
              └─► 3 × IndependentAgent (Bi-Mamba backbone + Conv-down heads)
                    └─► {μ_k, σ_k, h_k}_{k=t,c,l}    (每支 (B,P,D) / (B,P,D) / (B,d))
                          └─► ConsensusLayer (n_rounds × 广播 + 注意力通信 + IVW)
                                └─► {μ^R_k, σ^R_k, h^R_k}_{k=t,c,l}
                                      └─► AggregatorAgent (LightMamba ~0.5M)
                                            └─► α ∈ Δ³, r ∈ [0,1]^K, ŷ ∈ R^(B,P,D), σ ∈ R^(B,P,D)
```

每个阶段对其中一个子模块单独收敛，然后在最后全图一起微调，这样可以避免
早期的 α/r 门控信号把尚未训练好的专家完全压制，也能避免协商层的通信
梯度把分解器拉偏。

---

## 1. 损失函数总览

所有损失统一在 `src/losses/calibrator.py: build_total_loss()` 中组装，
支持任意子集项的 λ 加权求和。代码模块：

| 符号 | 代码路径 | 含义 |
| --- | --- | --- |
| L_nll | `gaussian_nll` / `mixture_nll` | 像素级 Gaussian（或 K=3 混合）负对数似然 |
| L_mse | `F.mse_loss` | 均值的 MSE（保证 NLL 不稳定时还有一个确定性锚） |
| L_consensus | `consensus_regularization` | 协商前后 ‖Δμ‖² + ‖Δσ‖² − 注意力熵(H) |
| L_reject | `reject_regularization` | r 软 BCE + 使用率罚项，目标使用率 = `target_usage` |
| L_ortho | `LearnableDecomposer.orthogonality_loss` | 三个分量 flattened 的两两余弦平方 |
| L_agent | `build_agent_heterogeneous_losses` | 专家异构正则项的加权和（见下表） |
| L_total | `build_total_loss` | Σ λ_i · L_i |

异构专家正则（`agent_losses.py`）：

| Agent | 正则项 | 实现 |
| --- | --- | --- |
| Trend | Total Variation (TV) | `total_variation_loss` = Σ ‖μ[:,t+1] − μ[:,t]‖₁ |
| Cycle | Seasonal Fourier-low-bin L1 | `seasonal_fourier_l1_loss` = Σ |FFT(μ_cycle)[:,:K_low] − FFT(y)[:,:K_low]|₁ |
| Local | 稀疏 L1 残差 | `local_sparsity_l1` = ‖μ_local − y‖₁ |

---

## 2. 推荐 λ 超参表

以下为默认值（已写入 `src/configs/default.yaml`），在 8 个数据集上表现稳健；
可按数据集大小（例如 electricity / traffic 的 D 很大）将 λ_agent 上调至
0.10 ~ 0.15：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| λ_nll | 1.00 | 主导损失；保证分布校准 |
| λ_mse | 0.50 | MSE 辅助；对早期不稳定 σ 有兜底 |
| λ_consensus | 0.10 | 只在 S2 / S4 启用；0 < H bonus ≤ 0.05 |
| λ_reject | 0.05 | 只在 S3 / S4 启用；`target_usage=0.95` |
| λ_orthogonality | 0.02 | S1 / S4 启用；cos² ∈ [0, 0.3] 时饱和 |
| λ_agent | 0.05 | S1 / S4 启用；TV / seasonal / sparse 内部均权 1.0 |
| λ_ece | 0.01 | S4 可选启用；ECE bin=10，惩罚过自信尾部 |

---

## 3. 四阶段细节

### Stage 1：预训练 — 可学习分解器 + 三个独立专家 Agent

**文件**：`src/training/stage1_pretrain.py`

- **训练哪些参数**：`trainable_names = ("decomposer.", "specialists.")`
- **冻结哪些参数**：`consensus`, `aggregator` — 整体作为 Identity pass-through
- **损失构成**：
  - 每个 agent 单独算 `gaussian_nll(μ_k, σ_k, y) + 0.5·MSE(μ_k, y)`（三项之和）
  - `λ_orthogonality · L_ortho`（分解器）
  - `λ_agent · L_agent_heterogeneous_total`（TV + seasonal + sparse）
- **典型 epoch**：`40 ~ 60`
- **典型 LR**：`8e-4`（AdamW；warmup=5, min_lr_factor=0.05）
- **早停监控**：`val/loss_stage1`
- **收敛判据**：分解器 orthogonality ≤ 0.10 且三 agent 的 val/MSE 都不再下降

### Stage 2：通信协商层单独训练

**文件**：`src/training/stage2_comm.py`

- **训练哪些参数**：`trainable_names = ("consensus.",)`
- **冻结哪些参数**：`decomposer`, `specialists`, `aggregator` — 全部冻结
- **输入**：Stage 1 专家输出的 3×{μ,σ,h} 作为 frozen 上游
- **损失构成**：
  - `gaussian_nll(μ^R_k, σ^R_k, y) + 0.5·MSE(μ^R_k, y)`（post-consensus）
  - `λ_consensus · L_consensus`（Δμ²+Δσ² − H(attn) bonus）
- **典型 epoch**：`15 ~ 25`
- **典型 LR**：`1e-3`（S2 参数量小，可以略大一些；梯度裁剪=1.0）
- **早停监控**：`val/loss_stage2`
- **收敛判据**：后验 μ^R_k 的 MSE 明显优于 Stage 1 的先验，且注意力权重熵
  `H(attn_w)` ≥ 0.7·log₂(K)（避免一轮就坍缩到某一个 agent）

### Stage 3：聚合器 LightMamba 单独训练

**文件**：`src/training/stage3_aggregator.py`

- **训练哪些参数**：`trainable_names = ("aggregator.",)`
- **冻结哪些参数**：`decomposer`, `specialists`, `consensus`
- **输入**：Stage 2 后的 3×{μ^R, σ^R, h^R} frozen
- **损失构成**：
  - `gaussian_nll(ŷ, σ, y) + 0.5·MSE(ŷ, y)`（主项）
  - `λ_reject · L_reject`；推荐 `target_usage=0.95`，保证平均 r ≤ 0.05
  - 可选 `use_mixture_nll=True`（`mixture_nll(K=3)`，以 α 为 mixture 权重）
- **典型 epoch**：`20 ~ 30`
- **典型 LR**：`8e-4`
- **早停监控**：`val/loss_stage3` 或 `val/mse`
- **收敛判据**：`val/mse` 优于任何单 agent（体现聚合带来的收益），且 α 分布
  没有出现 <1% 的塌陷（`min_k E[α_k]` ≥ 0.05 即可）

### Stage 4：端到端微调

**文件**：`src/training/stage4_finetune.py`

- **训练哪些参数**：全部 unfreeze
- **学习率缩放**：`base_lr × 0.10`（避免破坏 Stage 1~3 的好收敛点）
- **损失构成**：`build_total_loss(..., λ_nll=1.0, λ_mse=0.5, λ_consensus=0.1,
  λ_reject=0.05, λ_ortho=0.02, λ_agent=0.05, λ_ece=0.01, use_mixture_nll=True,
  agent_heterogeneous_outputs=hetero)` — **所有 λ 全部启用**
- **典型 epoch**：`30 ~ 50`
- **典型 LR**：`8e-5`（= stage1 的 1/10）
- **早停监控**：`val/mse` 或 `val/nll`（以 `--monitor` 参数选择）
- **收敛判据**：`val/mse` 在 patience=8 内无改进；加载 best checkpoint 做最终
  test 评估

---

## 4. 通用训练器 (Trainer) 机制

位于 `src/training/trainer.py`。核心特性：

- **可复现性**：`seed_everything(seed, cuda_deterministic=True)`
- **参数过滤**：`_named_filter(model, trainable_names, frozen_names)` 允许按前缀
  精确控制哪些参数参与 `optimizer.step()`，是 freeze/unfreeze 策略的基石
- **混合精度**：`GradScaler` 只在 device.type == "cuda" 时启用；CPU 下无警告
- **梯度裁剪**：`clip_grad_norm_` 默认值 = `1.0`（可调 `TrainerConfig.grad_clip`）
- **梯度累积**：`accumulate_grad_batches=N` 以支持 ETTh 大 batch 的显存不足
- **调度器**：`SequentialLR(LinearLR warmup=5, CosineAnnealingLR to η_min=lr·min_lr_factor)`
- **Early Stop + Best restore**：当 `early_stop` 触发后自动 `load_state_dict(best)`
- **AMP + CUDA/CPU fallback**：所有 tensor 设备统一从 first-batch 推断，无需手动 `.cuda()`

建议的常用参数（命令行）：

```bash
python -m src.experiments.main \
    --dataset ETTh1 --seq-len 336 --pred-len 168 \
    --batch-size 32 --max-epochs 80 --lr 8e-4 \
    --monitor val/mse --early-stop 10 \
    --seed 42
```

---

## 5. 监控指标清单

每个阶段通过 `ExperimentLogger`（`src/utils/logger.py`）记录：

| 指标 key | 含义 | 在哪个阶段启用 |
| --- | --- | --- |
| `train/loss_*`, `val/loss_*` | 阶段总损失标量 | S1~S4 |
| `{train,val}/nll`, `{train,val}/mse`, `{train,val}/mae` | 预测精度 | S1~S4 |
| `val/decomp/orthogonality` | 三分解分量两两 cos² 和 | S1, S4 |
| `val/consensus/delta_mu`, `val/consensus/attn_entropy` | 协商稳定性 | S2, S4 |
| `val/agg/alpha_{trend,cycle,local}_mean` | 聚合器权重均值 | S3, S4 |
| `val/agg/reject_{trend,cycle,local}_mean` | 拒绝信号均值 | S3, S4 |
| `val/agent/{tv,seasonal,sparse}` | 异构专家正则当前值 | S1, S4 |
| `test/{mse,mae,rmse,mape,corr,rse,q95_cov}` | test 汇总 | S4 结束时 |
| `grad_norm/total` | 优化 step 前总梯度范数 | 全部（用于诊断梯度爆炸） |

所有指标同时写入：`events.out.tfevents.*`（TensorBoard）、`metrics.csv`、
`history.json`。Best checkpoint 存为 `best.pt`，`final.pt` 为 last epoch。

---

## 6. 常见失败模式与排障

| 症状 | 原因 | 建议修复 |
| --- | --- | --- |
| Stage 1 三 agent val/MSE 几乎一样，且 `orthogonality ≈ 1.0` | 分解器没学会分开；λ_ortho 太小 | 调大 `λ_ortho=0.05 ~ 0.1`，或加 S1 epoch 到 80 |
| Stage 2 前后 Δμ ≈ 0（consensus 成了 Identity） | LR 太小，或 λ_consensus 中的 entropy bonus 没启用 | 把 λ_consensus 提到 0.2，且确保 H bonus 代码未被注释 |
| Stage 3 的 α 长期为 [1, 0, 0]（坍缩到 trend） | Stage 1 的 trend 明显比其他专家强 | 给三个专家用单独的 LR group，或加大 S1 中 cycle/local 的正则强度上限 |
| Stage 3 的 r_k → 1（全拒绝，`effective_weights` 近 0） | λ_reject 太大，或 `target_usage` 太低 | 降低 λ_reject ≤ 0.03，`target_usage=0.95` |
| Stage 4 一 unfreeze 立刻 val/mse 飙升 | S4 LR 过大（常见于 > 1e-4） | 严格使用 base_lr × 0.1，即 8e-5，且 grad_clip=0.5 |
| CPU 环境训练速度极慢（每 epoch > 5 min） | Bi-Mamba fallback 走了 `_CpuGatedConvMixer` | 调小 `d_model` / `n_layers`，或切换到 CUDA 机器运行 |

---

## 7. 命令行速查

### 快速复现（推荐默认）

```bash
# ETT 小时级 336→168（对应 ett.yaml）
uv run python -m src.experiments.main --dataset ETTh1 \
    --seq-len 336 --pred-len 168 --batch-size 32 --max-epochs 80

# Electricity 96→24（对应 electricity.yaml）
uv run python -m src.experiments.main --dataset electricity \
    --seq-len 96 --pred-len 24 --batch-size 16 --max-epochs 80 \
    --weight-decay 5e-4 --grad-clip 2.0
```

### 消融 / 鲁棒性 / 合成污染实验

```bash
uv run python -m src.experiments.ablation   --dataset ETTh1 --seq-len 96 --pred-len 24
uv run python -m src.experiments.robustness --dataset ETTh1 --seq-len 96 --pred-len 24
uv run python -m src.experiments.synthetic  --dataset ETTh1 --seq-len 96 --pred-len 24
```

### Bash 一键启动（封装所有超参）

```bash
bash src/scripts/run_experiment.sh --dataset ETTh1 --seq-len 336 --pred-len 168
```

完成 4 个阶段后，使用 `src/scripts/plot_results.py` 加载 `best.pt` 绘制预测曲线、
分量分解热力图、以及 α/r 的时序图，用于撰写论文 Figure。
