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
│   └── calibrator.py             # GaussianNLL + 校准指标
│
├── training/
│   ├── __init__.py
│   ├── stage1_pretrain.py        # 四阶段训练实现
│   ├── stage2_comm.py
│   ├── stage3_aggregator.py
│   ├── stage4_finetune.py
│   └── trainer.py                # 通用训练循环
│
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
