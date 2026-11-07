src/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml              # 默认配置
│   ├── ett.yaml                  # ETT数据集配置
│   └── electricity.yaml          # Electricity数据集配置
│
├── preprocess/                   #   TSLib 标准预处理 (项目内独立可执行)
│   ├── __init__.py               #   导出 DATASET_CONFIG / process_single_dataset / main
│   └── preprocess_tslib.py       #   CLI: python -m src.preprocess.preprocess_tslib --all
│                                   #   输入根: <PROJECT_ROOT>/dataset/   (原始 CSV)
│                                   #   输出根: <PROJECT_ROOT>/preprocess/ (train/val/test.npz + scaler.pkl + meta.json)
│
├── data/
│   ├── dataset.py                # 数据加载与预处理
│   ├── dataloader.py             # DataLoader封装
│   └── freq_decomp.py            # 可学习频率分解器 (核心)
│
├── models/
│   ├── __init__.py
│   ├── agent.py                  # IndependentAgent (独立智能体)
│   ├── bimamba.py                # BiDirectionalMamba 实现
│   ├── consensus.py              # ConsensusLayer (通信层)
│   ├── aggregator.py             # AggregatorAgent (聚合器)
│   └── tacf.py                   # TACF 完整模型组装
│
├── losses/
│   ├── __init__.py
│   ├── agent_losses.py           # 各智能体异构损失 (TV/季节性/L1)
│   │                                - seasonal_fourier_l1_loss: FFT 前 float16 → float32 提升（避 cuFFT half 2^n ASSERT）
│   └── calibrator.py             # GaussianNLL + 校准指标 （详见下）
│                                  -- gaussian_nll(mu,sigma,y,sigma_reg_weight)
│                                     · σ 数值保护 clamp(σ, 1e-3) 用于 1/σ² 与 logσ 公式
│                                     · σ_reg (λ·(mean σ + mean log(σ_clamp(σ, 1e-5)))) — **与数值 clamp 分两段**，
│                                       保证 σ=1e-5 时 σ_reg 的正梯度仍把 σ 推大（修复 2026-09-19 σ 塌缩 bug）
│                                  -- ece_loss: ECE gap = |coverage(q_target) − q_target|
│                                  -- gaussian_crps(μ,σ,y) 闭式解 = σ·(z(2Φ(z)−1) + 2φ(z) − 1/√π)
│                                  -- reject_distribution_penalty(r, μ_t=0.1, σ_t=0.05)
│                                     · (mean−0.1)² + |std−0.05|² + entropy_weight·(1 − H_normed(r_hist))
│                                  -- temperature_scale_nll(mus,sigmas,ys) 单参数 σ'=Tσ 解
│                                     · 粗扫 T∈[0.1,5.0] 41 点 → 精修 L-BFGS on logT
│                                     · 输出 {T, nll_before, nll_after, status, N_samples}
│                                  -- build_total_loss(..., lambda_ece, lambda_crps, lambda_reject_dist, sigma_reg_weight)
│                                     · 缺省 lambda=0 不生效；所有 λ>0 项通过 float(λ) 乘法（防 cfg.lambda 是 CPU tensor）

├── training/
│   ├── __init__.py
│   ├── stage1_pretrain.py        # 四阶段训练实现（重要 dtype 安全策略，见下）
│   ├── stage2_comm.py
│   ├── stage3_aggregator.py       · _loss_fn 内联 ece_loss + reject_distribution_penalty 对齐 build_total_loss
│   ├── stage4_finetune.py         · build_total_loss 调用显式传入 sigma_reg_weight + lambda_ece/crps/reject_dist
│   └── trainer.py                 # 通用训练循环
│                                  -- TrainerConfig lambda_ece / lambda_crps / lambda_reject_dist / eval_temperature 新字段
│                                  -- _compute_monitor: cfg.monitor="val_nll" 自动映射到 MetricsResult plain "nll" attr，
│                                     三种 key 形式（带 val_/plain/拼接 val_+plain）顺序尝试，miss 才 fallback；
│                                     修复 2026-09-19 best.json = train_loss=−1.682 (monitor 本该是 val_nll)
│                                  -- 每 epoch VAL/TEST 打印：mse/mae/corr │ nll/crps/ece │ Q50/Q90/Q95 + Width
│                                     （**不再显示 RMSE**，字段仍兼容 CSV 列）
│                                  -- NaN 自恢复：train_loss 连续 nan_recovery_threshold=2 ep finite→NaN，
│                                     回滚 best.pt，LR×nan_lr_decay=0.5，scheduler 重 init，上限 3 次
│                                  -- S2 Warm-start：前 s2_warm_epochs 个 ep，scheduler 输出再 × s2_warm_lr_mult=0.1
│                                  -- _evaluate: compute_metrics(temperature=cfg.eval_temperature)
│
│   【四 stage runner dtype 安全策略（修复 CUDAGuardImpl.h:28 ASSERT）】
│   · 所有累加器：`zero = torch.zeros((), device=y.device, dtype=torch.float32)`
│   · 每步加法：`nll_sum = nll_sum + gaussian_nll(..., sigma_reg_weight=λ).to(torch.float32)`
│   · λ 全部显式 `float(cfg.lambda_x) * x`，禁止 cfg.lambda_* 是 tensor 参与
│   · total 末步 `.to(dtype=y.dtype)` cast 回模型 dtype 一次
│   · aux tensor 非 tensor 情况统一 torch.as_tensor(x, device=y.device, dtype=torch.float32)

├── experiments/
│   ├── main.py                   # 主实验入口
│   ├── ablation.py               # 消融实验
│   ├── robustness.py             # 鲁棒性测试
│   ├── synthetic.py              # 合成污染实验 (拒绝验证)
│   └── configs/                  # 各实验配置
│
├── baselines/
│   ├── __init__.py
│   ├── mafs.py                   # MAFS 复现 (TIP 2025)
│   ├── time_moe.py               # Time-MoE 接口 (ICLR 2025)
│   ├── m2fmoe.py                 # M²FMoE 接口
│   └── patchtst.py               # PatchTST 接口
│
├── utils/
│   ├── metrics.py                # MSE/MAE/CRPS/ECE/覆盖率
│   ├── visualization.py          # 可视化工具
│   └── logger.py                 # 日志与检查点
│
├── tests/
│   ├── test_agent.py             # 智能体单元测试
│   ├── test_consensus.py         # 通信层单元测试
│   └── test_tacf.py              # 端到端测试
│
└── scripts/
    ├── run_experiment.sh         # 实验运行脚本
    └── plot_results.py           # 结果绘图


附：目录补充说明
=========================

  dataset/                         # 原始 CSV 数据 (ETT-small, weather, electricity, traffic, exchange_rate)
  preprocess/                      # TACF-ready 预处理输出 (8 × {train/val/test.npz, scaler.pkl, meta.json})
  Time-Series-Library/             # 上游 TSLib 仓库 (源数据 + 参考 data_provider)
  logs/                            # 训练日志与 best checkpoints
  figures/                         # 实验绘图输出

Git LFS 追踪规则（已写入 .gitattributes，不需要再改 .gitignore）
========================
为突破 GitHub 100 MB 单文件硬限制，以下路径的二进制资产将被 Git LFS 管理

  dataset/**/*.csv                 # 所有原始 CSV（最大 131 MB traffic.csv，
  preprocess/**/*.npz              #   92 MB electricity.csv，小于 50 MB 的
  preprocess/**/*.pkl              #   ETT/weather/exchange_rate 同样一起登记）
