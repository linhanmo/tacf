# WSL2 配置与安装
```powershell
wsl --install --no-distribution # 请重启电脑
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart # 请重启电脑
wsl --update
wsl --status
wsl --install Ubuntu-22.04
# 接下来是设置用户名和密码，设置完后重启powershell
wsl --shutdown
wsl --manage Ubuntu-22.04 --move "E:\wsl\Ubuntu"
wsl --shutdown
sc stop LxssManager
icacls "D:\wsl\Ubuntu\ext4.vhdx" /grant "$echo:USERNAME":F /T /C
sc start LxssManager
wsl -d Ubuntu -e echo "WSL 启动成功"
takeown /f "D:\wsl\Ubuntu\ext4.vhdx"
cacls "D:\wsl\Ubuntu\ext4.vhdx" /setowner "$echo:USERNAME" /T /C # 请重启电脑
wsl
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd ':')
echo 'export PATH=$(echo "$PATH" | tr ":" "\n" | grep -v "^/mnt/" | paste -sd ":")' >> ~/.bashrc
cd ~
sudo apt update && sudo apt install -y wget build-essential cmake git ninja-build dkms libopenmpi-dev
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
chmod +x cuda_12.6.3_560.35.05_linux.run
sudo sh cuda_12.6.3_560.35.05_linux.run # 协议填accept，其他默认
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
wget https://repo.anaconda.com/miniconda/Miniconda3-py312_24.11.1-0-Linux-x86_64.sh
bash Miniconda3-py312_24.11.1-0-Linux-x86_64.sh
source ~/.bashrc
rm Miniconda3-py312_24.11.1-0-Linux-x86_64.sh
rm cuda_12.6.3_560.35.05_linux.run
nvidia-smi # 验证cuda安装
conda --version # 验证conda安装
python --version # 查看conda的python版本
conda config --set auto_activate_base false # 验证完成后关闭conda的自动激活，重启powershell
wsl # 重启wsl，进入下一步：配置Mamba-SSM
```
# 配置Mamba-SSM
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc #安装uv并注册
mkdir ~/tacf # 创建项目目录
cd ~/tacf #进入项目目录
uv venv --python=3.10
source .venv/bin/activate
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install "numpy<2"
uv pip install setuptools
uv pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install "transformers==4.44.2"
uv pip install "tokenizers<0.20" "numpy<2"  # 进入下一步：验证安装
```

# 验证安装
```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # 忽略FutureWarning
import torch, causal_conv1d, mamba_ssm
print(torch.__version__) # 打印torch版本
from mamba_ssm import Mamba
m = Mamba(d_model=256, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 128, 256).cuda()
m(x).sum().backward() # 跑一次mamba推理实例验证
print("OK")
```

# TACF 训练配置与脚本（Scale-Up + Gradient Accumulation）

> **核心约定（2026-09-12 v2）**：
> - GPU tier 只保留 **24G** 和 **8G** 两档（去掉 16G/12G）。
> - **同一数据集的 24G 和 8G 版本，`d_model / n_layers / agg_d_model / agg_n_layers` 完全相同**，
>   仅通过 **physical batch size × gradient accumulation steps** 来适配显存。
> - 等效 batch size = `BS_phys × accum_steps`，对同一数据集 24G/8G 保持一致，保证 BatchNorm 统计
>   和梯度噪声在不同显卡上等价。
> - traffic (D=862) 数据集暂时从跑计划里移除，但保留 YAML 配置文件；需要跑时传 `--force-traffic`。

---

## 0. 一键脚本：`scaled_runs.sh`

位置：[scaled_runs.sh](file:///home/lin/tacf/scaled_runs.sh)

### 用法
```bash
bash scaled_runs.sh --help                                    # 查看全部参数
bash scaled_runs.sh --dataset ETTh1         --gpu-tier 24G    # ETT × 24G 全流程
bash scaled_runs.sh --dataset weather       --gpu-tier 8G     # Weather × 8G
bash scaled_runs.sh --dataset electricity   --gpu-tier 24G    # Elec D=321 × 24G
bash scaled_runs.sh --dataset exchange_rate --gpu-tier 8G     # Exchange D=8 × 8G

# traffic 默认禁跑，要强制请传 --force-traffic：
bash scaled_runs.sh --dataset traffic --gpu-tier 24G --force-traffic

# 常用额外参数
bash scaled_runs.sh --dataset exchange_rate --gpu-tier 8G \
   --no-vram-check \                             # 跳过 estimate_vram_fast VRAM 预检
   --extra "--max-epochs-stage1 1 --device cpu"  # 透传 main.py 的参数
```

### 支持的 dataset / tier 组合
| 数据集 | 24G | 8G |
|---|---|---|
| ETTh1 / ETTh2 / ETTm1 / ETTm2 | ✅ | ✅ |
| weather (D=21) | ✅ | ✅ |
| electricity (D=321) | ✅ | ✅ |
| exchange_rate (D=8) | ✅ | ✅ |
| traffic (D=862) | ✅（需 `--force-traffic`） | ✅（需 `--force-traffic`，有 OOM 风险） |

### 启动前会打印
```
  d_model / n_layers / agg_d_model     :  锁定（24G 和 8G 完全相同）
  物理 batch size (→ 决定 VRAM)        :  X
  梯度累积 accum_steps                 :  Y
  等效 batch size (→ 决定梯度噪声/BN)  :  X × Y
```

---

## 1. 配置文件一览：`src/configs/scales/*.yaml`

目录：[src/configs/scales/](file:///home/lin/tacf/src/configs/scales/)

### 总览表（d_model / layers 全锁定，只改 phys BS × accum）

| 数据集 | YAML 文件 | 模型 (24G & 8G **完全相同**) | 24G phys × accum = eff | 8G phys × accum = eff |
|---|---|---|---|---|
| **ETT ×4** | [ett_24g.yaml](file:///home/lin/tacf/src/configs/scales/ett_24g.yaml)<br>[ett_8g.yaml](file:///home/lin/tacf/src/configs/scales/ett_8g.yaml) | d=512, L=4<br>agg d=256, L=2 | 16 × 4 = **64** | 4 × 16 = **64** |
| **Weather** | [weather_24g.yaml](file:///home/lin/tacf/src/configs/scales/weather_24g.yaml)<br>[weather_8g.yaml](file:///home/lin/tacf/src/configs/scales/weather_8g.yaml) | d=512, L=4<br>agg d=256, L=2 | 12 × 4 = **48** | 3 × 16 = **48** |
| **Electricity** | [electricity_24g.yaml](file:///home/lin/tacf/src/configs/scales/electricity_24g.yaml)<br>[electricity_8g.yaml](file:///home/lin/tacf/src/configs/scales/electricity_8g.yaml) | d=384, L=3<br>agg d=192, L=2 | 10 × 4 = **40** | 2 × 20 = **40** |
| **Exchange_rate** | [exchange_24g.yaml](file:///home/lin/tacf/src/configs/scales/exchange_24g.yaml)<br>[exchange_8g.yaml](file:///home/lin/tacf/src/configs/scales/exchange_8g.yaml) | d=512, L=4<br>agg d=256, L=2 | 32 × 2 = **64** | 8 × 8 = **64** |
| **Traffic**<br>（保留，禁跑） | [traffic_24g.yaml](file:///home/lin/tacf/src/configs/scales/traffic_24g.yaml)<br>[traffic_8g.yaml](file:///home/lin/tacf/src/configs/scales/traffic_8g.yaml) | d=384, L=3<br>agg d=192, L=2 | 2 × 32 = 64 | 1 × 64 = 64<br>（borderline） |

> 所有 YAML 的学习率 / epochs / early-stop 约定：
> - `stage1 40ep / stop15`, `stage2 15ep / stop6`, `stage3 15ep / stop6`, `stage4 30ep / stop8`
>   （traffic stage1 60 stop15, stage2-3 20 stop8）
> - `stage4_lr_mult=0.03`（比之前 0.10 小 3×，防止 aggregator 回退）
> - `stage4_early_stop=5`
> - Loss：`lambda_nll=2.0`, `lambda_cov_penalty=0.25`, `target_q=0.95`（sigma 校准三件套）

---

## 2. 常见全流水线 batch 跑法

### 7 数据集 × 2 档（traffic 自动跳过）
```bash
for ds in ETTh1 ETTh2 ETTm1 ETTm2 weather electricity exchange_rate; do
  for tier in 24G 8G; do
    echo "=== [$ds @ $tier] ==="
    bash scaled_runs.sh --dataset "$ds" --gpu-tier "$tier" || { echo "FAILED: $ds @ $tier" ; break 2 ; }
  done
done
```

### 单次 smoke（最小数据集 exchange，每 stage 只跑 1 epoch，CPU）
```bash
bash scaled_runs.sh --dataset exchange_rate --gpu-tier 8G --no-vram-check --extra "
  --max-epochs-stage1 1
  --max-epochs-stage2 1
  --max-epochs-stage3 1
  --max-epochs-stage4 1
  --device cpu
  --num-workers 0
"
```
完成后检查：`logs/tacf_*/checkpoints/final_best.pt` 是否存在，`logs/tacf_*/results.json` 是否含 `final_best` 段。

---

## 3. Gradient Accumulation 怎么生效的（原理速查）

1. CLI `--grad-accum N` 在 [main.py](file:///home/lin/tacf/src/experiments/main.py#L69-L71) 进入
   `cfg_common['accum_steps'] = N`；
2. 4 个 stage 的 `TrainerConfig(**cfg_common)` 都带上 `accum_steps=N`；
3. [trainer.py::_train_one_epoch](file:///home/lin/tacf/src/training/trainer.py#L233-L245)：
   ```python
   loss = loss / self.cfg.accum_steps
   self.scaler.scale(loss).backward()
   if (step + 1) % self.cfg.accum_steps == 0:
       # grad_clip + scaler.step() + scaler.update() + zero_grad() 每 N 步才执行一次
   ```
4. 物理显存 ∝ `batch_size`（physical），**和等效 BS 无关**，因此 24G 卡上能跑出等效 64 的大 batch。