src/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml              # 默认配置
│   ├── ett.yaml                  # ETT数据集配置
│   └── electricity.yaml          # Electricity数据集配置
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