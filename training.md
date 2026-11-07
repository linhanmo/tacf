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
支持任意子集项的 λ 加权求和。除基础 NLL/MSE/Consensus 外新增的校准 &
正则项必须显式开启（`lambda_* > 0` 才真正加进 `total`）。代码模块：

| 符号 | 代码路径 | 含义 |
| --- | --- | --- |
| L_nll | `gaussian_nll` / `mixture_nll` | 像素级 Gaussian（或 K=3 混合）负对数似然 |
| L_mse | `F.mse_loss` | 均值的 MSE（保证 NLL 不稳定时还有一个确定性锚） |
| L_consensus | `consensus_regularization` | 协商前后 ‖Δμ‖² + ‖Δσ‖² − 注意力熵(H) |
| L_reject | `reject_regularization` | r 软 BCE + 使用率罚项，目标使用率 = `target_usage` |
| L_ortho | `LearnableDecomposer.orthogonality_loss` | 三个分量 flattened 的两两余弦平方 |
| L_agent | `build_agent_heterogeneous_losses` | 专家异构正则项的加权和（见下表） |
| **L_ece** | `ece_loss(σ, μ, y, q_target=0.95)` | ECE gap 形式 = `|覆盖率 − q_target|`，校准罚项；S3/S4 启用 |
| **L_crps** | `gaussian_crps(μ, σ, y)` 闭式解 | 连续排序概率得分（proper scoring rule），**S4 完全替代 RMSE 作为 probabilistic secondary** |
| **L_r_dist** | `reject_distribution_penalty(r, μ_t=0.1, σ_t=0.05)` | r_k 分布惩罚：(mean−0.1)² + (std−0.05)² + (1 − H_normed(hist))；S3 启用 |
| **L_σ_reg** | `gaussian_nll(... sigma_reg_weight=λ)` 内联 | `λ · (mean[σ] + mean[log σ_clamp_floor(σ, 1e-5)])` — σ 凸正则，**极小 σ 的 log 梯度不切断** |
| L_total | `build_total_loss` | Σ λ_i · L_i |

> **关于 RMSE**：代码仍保留 `val_rmse / test_rmse` 两个字段兼容历史 CSV，但
> ① 所有 `trainer.py` epoch 打印、② best-epoch 选择决策、③ 论文/汇报的辅指标
> 全部 **不再引用 RMSE**。S4 的 probabilistic secondary 指标位由 `CRPS` 占据。

异构专家正则（`agent_losses.py`）：

| Agent | 正则项 | 实现 |
| --- | --- | --- |
| Trend | Total Variation (TV) | `total_variation_loss` = Σ ‖μ[:,t+1] − μ[:,t]‖₁ |
| Cycle | Seasonal Fourier-low-bin L1 | `seasonal_fourier_l1_loss` = Σ |FFT(μ_cycle)[:,:K_low] − FFT(y)[:,:K_low]|₁ |
| Local | 稀疏 L1 残差 | `local_sparsity_l1` = ‖μ_local − y‖₁ |

---

## 2. 四阶段 λ 超参与监控协议（来自用户 request 11 的正式规范）

> 以下 λ 已硬编码到 `src/experiments/main.py` 的 `cfg_s1 / cfg_s2 / cfg_s3 / cfg_s4` 中。
> 监控指标和 best-epoch 选择策略 **严格按 stage 区分**，不再使用 composite coverage 选最佳。

| Stage | monitor (best epoch) | λ_MSE | λ_NLL | λ_ECE | λ_CRPS | λ_Rdist | σ_reg | 其它关键设置 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **S1 Pretrain** | `val_mse` (min) | **1.00** (主) | 0.05 (仅 σ 非零用) | 0 | 0 | 0 | 0.01 | λ_ortho=0.02, λ_agent=0.05, λ_cov_penalty=0.25, warmup=5ep |
| **S2 Comm** | `val_mse` (min) | **1.00** (主) | 0.10 | 0 | 0 | 0 | 0.02 | **前 5 ep LR×0.1 warm start** (防止零初始化 consensus 一次冲坏) |
| **S3 Aggregator** | `val_nll` (min) | 0.20 (辅) | **1.00** (主) | **0.5** | 0 | **0.2** | 0.05 | λ_reject=1.0, λ_cov_penalty=0.3, target_q=0.95 |
| **S4 Finetune** | `val_nll` (min) → **post-T-scaled val_nll (final)** | 0.20 (辅) | **1.00** (主) | **0.3** | **0.5** | 0 | 0.03 | **后处理：σ × T 最小化 val NLL 作为 final best**；ECE, Q50/Q90/Q95 覆盖辅助 |

后处理流程仅在 S4 结束后自动执行（代码位于 `main.py` S4 runner 之后）：
1. 加载 S4 best.pt → 在 val 上重新 `_evaluate(val_loader)` 拿到 `(μs, σs, ys)`
2. `temperature_scale_nll(μs, σs, ys)`：`[0.1, 5.0]` 41 点 grid 粗扫 → L-BFGS on logT 精修
3. `cfg.eval_temperature = T` → 再跑一次 val/test `_evaluate`，得到 post-T metrics
4. 写 `results["stage4"]["temperature_scaling"]` + `scaled_val/test_metrics`

### 2.1 σ 保护与数值细节

- **σ 数值 floor**：三处统一 `eps=1e-3` —— `gaussian_nll.sigma_numerical` / `metrics.compute_metrics.eps_nll` / `temperature_scale_nll.eps`
- **σ_reg 梯度保护（关键）**：`gaussian_nll` 的 σ 凸正则**不与数值 floor 共享同一个 clamp**：
  - 数值保护（1/σ², log σ 公式里）→ `sigma_numerical = clamp(σ, min=1e-3)`
  - σ_reg 项里 → `log_arg = clamp(σ, min=1e-5)`（极小 floor 仅保证 log 有限）
  - 结果：σ=1e-5 时 `dL/dσ` 仍有 +1.25 的正梯度推 σ ↑，而不是被 clamp 梯度切断
- **避免 `float + CUDA f16 tensor` 触发的 `CUDAGuardImpl.h:28` 崩溃**：4 个 stage
  runner 的 `_loss_fn` 全部以 `torch.zeros((), device=y.device, dtype=torch.float32)`
  初始化累加器，loss 在 f32 下累加完成后末步 `.to(dtype=y.dtype)` 转一次模型 dtype，
  中间不出现 `0.0 + CUDA tensor` 这类危险混合。

---

## 3. 四阶段细节

### Stage 1：预训练 — 可学习分解器 + 三个独立专家 Agent

**文件**：`src/training/stage1_pretrain.py`

- **训练哪些参数**：`trainable_names = ("decomposer.", "specialists.")`
- **冻结哪些参数**：`consensus`, `aggregator` — 整体作为 Identity pass-through
- **损失构成（按 S1 协议 = 主 MSE）**：
  - 每个 agent 单独算 `0.05·gaussian_nll(μ_k, σ_k, y) + 1.0·MSE(μ_k, y)` 三项之和
  - **`sigma_reg_weight=0.01`**（σ + log σ 的凸正则，避免 σ 被 NLL log 项单调推到 0）
  - `λ_orthogonality · L_ortho`（分解器）
  - `λ_agent · L_agent_heterogeneous_total`（TV + seasonal + sparse）
- **典型 epoch**：`40 ~ 60`
- **典型 LR**：`8e-4`（AdamW；warmup=5, min_lr_factor=0.05）
- **早停监控 = **`val_mse`**（按新协议）**；monitor_mode=min
- **收敛判据**：val_mse 不再下降，分解器 orthogonality ≤ 0.10 且三 agent 的 val_MSE 都不再下降

### Stage 2：通信协商层单独训练

**文件**：`src/training/stage2_comm.py`

- **训练哪些参数**：`trainable_names = ("consensus.",)`
- **冻结哪些参数**：`decomposer`, `specialists`, `aggregator` — 全部冻结
- **输入**：Stage 1 专家输出的 3×{μ,σ,h} 作为 frozen 上游
- **损失构成（按 S2 协议 = 主 MSE + 前 5 ep LR warm）**：
  - `0.1·gaussian_nll(μ^R_k, σ^R_k, y) + 1.0·MSE(μ^R_k, y)`（post-consensus，σ_reg_weight=0.02）
  - `λ_consensus · L_consensus`（Δμ²+Δσ² − H(attn) bonus）
- **典型 epoch**：`15 ~ 25`
- **典型 LR**：`1e-3`（S2 参数量小，可以略大一些；梯度裁剪=1.0）
- **LR warm start**：前 **5 个 epoch scheduler step 结果再 × 0.1**（防止零初始化 consensus 一次冲坏专家输出，对应历史数据 S2 MSE 倒退 76% 的根因修复）
- **早停监控 = **`val_mse`**（按新协议）**
- **收敛判据**：后验 μ^R_k 的 MSE 明显优于 Stage 1 的先验，且注意力权重熵
  `H(attn_w)` ≥ 0.7·log₂(K)（避免一轮就坍缩到某一个 agent）

### Stage 3：聚合器 LightMamba 单独训练

**文件**：`src/training/stage3_aggregator.py`

- **训练哪些参数**：`trainable_names = ("aggregator.",)`
- **冻结哪些参数**：`decomposer`, `specialists`, `consensus`
- **输入**：Stage 2 后的 3×{μ^R, σ^R, h^R} frozen
- **损失构成（按 S3 协议 = 主 NLL + ECE/r_k 分布辅助）**：
  - **主项 1.0·gaussian_nll(ŷ, σ, y)**（sigma_reg_weight=0.05 强 σ 保护）
  - 辅项 0.2·MSE(ŷ, y)（保持 μ 锚）
  - `λ_reject · L_reject`；推荐 `target_usage=0.95`，保证平均 r ≤ 0.05
  - **λ_ECE=0.5 · ECE_gap(coverage_95 − 0.95)**（训练中直接约束校准，而非靠 monitor 挑）
  - **λ_Rdist=0.2 · reject_distribution_penalty(r, μ_t=0.1, σ_t=0.05)**（鼓励 r_k 分布在 10% 左右且直方图不平坦）
  - 可选 `use_mixture_nll=True`（`mixture_nll(K=3)`，以 α 为 mixture 权重）
- **典型 epoch**：`20 ~ 30`
- **典型 LR**：`8e-4`
- **早停监控 = **`val_nll`**（按新协议）**；S3 不再用 composite score 选 snapshot
- **收敛判据**：val_nll 最小 + ECE<0.2 + Q95 覆盖 > 0.7；α 分布没有出现 <1% 的塌陷（`min_k E[α_k]` ≥ 0.05 即可）

### Stage 4：端到端微调

**文件**：`src/training/stage4_finetune.py` + `main.py` S4 runner 之后的温度缩放块

- **训练哪些参数**：全部 unfreeze
- **学习率缩放**：`base_lr × 0.10`（避免破坏 Stage 1~3 的好收敛点）
- **损失构成（按 S4 协议 = 主 NLL + CRPS probabilistic secondary）**：
  `build_total_loss(..., λ_nll=1.0, λ_mse=0.2, λ_consensus=0.1,
  λ_reject=1.0, λ_ortho=0.02, λ_agent=0.05,
  λ_ece=0.3, λ_crps=0.5, sigma_reg_weight=0.03, λ_cov_penalty=0.3,
  use_mixture_nll=True, agent_heterogeneous_outputs=hetero)` — **除 λ_Rdist 外全部 λ 启用**
- **典型 epoch**：`30 ~ 50`
- **典型 LR**：`8e-5`（= stage1 的 1/10）
- **早停监控 = **`val_nll`**（训练过程）**；patience=8
- **Final Best 判定 = 温度缩放后的 val_nll（见下）**

### Stage 4 后处理：Post-hoc 温度缩放

**位置**：`main.py` S4 `runner(**runner_kwargs)` 执行完毕、best-stage selection 之前。

**流程**
1. 临时构造一个 TrainerConfig = cfg_s4 + `weight_path = stage4_best_checkpoint_path` 的实例，加载 best.pt 权重；
2. 在 val 上跑 `_evaluate(val_loader)` 得到 `(val_metrics, val_mus, val_ys, val_sigmas)`；
3. 调用 `temperature_scale_nll(val_mus, val_sigmas, val_ys)` 解 T；
4. `trainer.cfg.eval_temperature = T_final` → `_evaluate(val_loader)` 得到 **post-T val metrics**，再 `_evaluate(test_loader)` 得到 **post-T test metrics**；
5. 写 `results["stage4"]["temperature_scaling"] = {T, nll_before, nll_after, status, N_samples}`，`results["stage4"]["best_post_T"] = {val: ... , test: ...}`；
6. 控制台打印 `🌡 S4 Post-hoc T-scaling: T=2.987 NLL 4.17→1.31` + post-T VAL/TEST 的 NLL/CRPS/ECE/Q50/Q90/Q95。

**失败的快速判定**（一眼看 T 是否合理）：
- **正常**：T ∈ [0.8, 2.5]；说明模型自身 σ 量级就对，仅 ±3× 校准
- **警告**：T=0.10 或 5.0（被夹住）→ 说明 σ 塌缩或 σ 放大 5× 仍然不够，**T-scaling 救不了**，需要回到 S1/S3 加强 σ_reg
- **严重错误**：`temperature_scale_nll.status` 返回 `grid_all_inf` 或 `nan_after` → σ 已数值炸掉，必须先回 S1 调 σ_reg_weight 再做

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
- **Best-epoch monitor key 映射修复**：`cfg.monitor`（常写为 `val_nll / val_mse`）
  会自动在 `MetricsResult` 的 plain 名（`nll / mse` / `crps / ece`）之间
  映射命中，**不会因为前缀 miss 而 fallback 到 train_loss**（修复 2026-09-19
  那轮 best.json = train_loss=−1.68 的 bug）
- **Trainer NaN 自恢复**：`train_loss` 连续 2 ep finite→NaN 时自动回滚 best.pt，
  所有 param_group LR × 0.5，scheduler 重 init（剩余 epochs，warmup 50%），
  最多重试 3 次；避免一次 GradScaler overflow 永久失去训练能力
- **S2 Warm-start**：前 `cfg.s2_warm_epochs=5` 个 epoch 所有 param_group 的
  scheduler 学习率结果再 `× cfg.s2_warm_lr_mult=0.1`（零初始化 consensus 防冲坏）

建议的常用参数（命令行）：

```bash
python -m src.experiments.main \
    --dataset ETTh1 --seq-len 336 --pred-len 168 \
    --batch-size 32 --max-epochs 80 --lr 8e-4 \
    --seed 42
```

---

## 5. 监控指标清单

每个阶段通过 `ExperimentLogger`（`src/utils/logger.py`）记录：

| 指标 key | 含义 | 在哪个阶段启用 |
| --- | --- | --- |
| `train/loss_*`, `val/loss_*` | 阶段总损失标量 | S1~S4 |
| `{train,val}/nll`, `{train,val}/mse`, `{train,val}/mae` | 预测精度 | S1~S4 |
| **`{val,test}/crps`** | Gaussian CRPS（闭式解），S4 用其取代 RMSE 作为 probabilistic secondary | S3/S4 + final best |
| **`{val,test}/ece`** | ECE gap 形式 = `|Q95 覆盖 − 0.95|` | S3/S4 + final best |
| **`{val,test}/q50_cov / q90_cov / q95_cov`** | 三分位数实际覆盖率 | S3/S4 + final best |
| **`{val,test}/q50_width / q90_width / q95_width`** | 对应区间的平均预测宽度（辅助校准宽窄判断） | S3/S4 + final best |
| `val/decomp/orthogonality` | 三分解分量两两 cos² 和 | S1, S4 |
| `val/consensus/delta_mu`, `val/consensus/attn_entropy` | 协商稳定性 | S2, S4 |
| `val/agg/alpha_{trend,cycle,local}_mean` | 聚合器权重均值 | S3, S4 |
| `val/agg/reject_{trend,cycle,local}_mean` | 拒绝信号均值 | S3, S4 |
| `val/agent/{tv,seasonal,sparse}` | 异构专家正则当前值 | S1, S4 |
| `test/{mse,mae,mape,corr,rse}` | test 汇总（确定性） | S4 结束时 |
| `test_rmse / val_rmse` | **软弃用列**：CSV 中仍保留兼容，但 epoch 打印 / best 决策 / 汇报不引用 | 兼容保留 |
| `grad_norm/total` | 优化 step 前总梯度范数 | 全部（用于诊断梯度爆炸） |

> 每 epoch 结束时 VAL/TEST 打印的格式（无 RMSE）：
> `val:  mse=xx mae=xx corr=xx  │  nll=xx crps=xx ece=xx  │  Q50=xx Q90=xx Q95=xx  W50/90/95=xx/xx/xx`
> `★ new best / ⏹ early stop` 标志同步显示。

所有指标同时写入：`events.out.tfevents.*`（TensorBoard）、`history.csv`、
`history.json`。Best checkpoint 存为 `checkpoints/best.pt` + `best.json`
（`monitor_value` = 对应 monitor 名的实际最小/最大值，与 epoch 记录一致），
`final.pt` 为 last epoch。

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
| **`Q95_cov < 0.2` 且 val_nll > 20 持续上升** | **σ 塌缩到 1e-3 以下，Q95 区间覆盖不了任何数据**；NLL 的 log σ 项把 σ 推 0 的同时 σ_reg 梯度被 clamp 切断 | 已修复：把 σ_reg 与数值 floor 的 clamp 分离（见 §2.1）。如果仍复发，把 cfg_s1/s3/s4 的 `sigma_reg_weight` 都 ×2 |
| **CUDAGuardImpl.h:28 崩溃（CUDA INTERNAL ASSERT FAILED）** | stage runner `_loss_fn` 里 `float + CUDA f16 tensor` 混合加法触发 PyTorch 内部标量提升 bug | 已修复：4 个 runner 全部改用 `torch.zeros((), device=y, dtype=torch.float32)` 累加 + 末步 `.to(y.dtype)`；避免把 Python float 与 CUDA tensor 做二元运算 |
| **best.json monitor_value 明显与 val_nll/mse 对不上，等于 train_loss** | `cfg.monitor="val_nll"` 但 `MetricsResult` attr 是 plain `nll`，miss 后 fallback 到 train_loss | 已修复：Trainer 的 `_compute_monitor` 自动尝试 `val_*/plain/val_+plain` 三种 key；检查 best.json 里 `monitor_value` 应等于 best epoch 的对应 val_* 列，且比历史最小 |
| **温度缩放 T=5.0 夹上限或 T=0.1 夹下限** | σ 量级差了 5× 以上（σ 塌缩或 σ 膨胀），单参数 T-scaling 救不回来 | 回 S1/S3 调 σ_reg，目标是 S4 训练后的 σ 自身量级在 0.5~2 之间，温度缩放仅承担 ±3× 微调 |

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
