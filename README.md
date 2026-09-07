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