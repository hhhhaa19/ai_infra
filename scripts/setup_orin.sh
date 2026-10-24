#!/bin/bash
# 在 Jetson AGX Orin（JetPack 5.1.2）上建 PyTorch 开发环境。
# 用法：bash scripts/setup_orin.sh        （系统包那一步需要 sudo 密码）
# 不要在 conda 里运行：NVIDIA 的 torch 只适配系统 Python 3.8。
set -euo pipefail

VENV=${VENV:-$HOME/venvs/ai_infra}
WHEEL_DIR=${WHEEL_DIR:-$HOME/wheels}
WHEEL=torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
WHEEL_URL=https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/$WHEEL
WHEEL_SHA256=5112e2ef5051f1003ae2ffb545ae596377df66979d62a954d774f18009064dbe
REPO=$(cd "$(dirname "$0")/.." && pwd)

# 1. 系统依赖：venv 需要 python3-venv；torch 运行时需要 OpenBLAS。
sudo apt-get update
sudo apt-get install -y python3-venv libopenblas-dev

# 2. 下载并校验 NVIDIA 的 torch 安装包。
mkdir -p "$WHEEL_DIR"
[ -f "$WHEEL_DIR/$WHEEL" ] || wget -q -O "$WHEEL_DIR/$WHEEL" "$WHEEL_URL"
echo "$WHEEL_SHA256  $WHEEL_DIR/$WHEEL" | sha256sum -c -

# 3. venv：--system-site-packages 让它看得到 JetPack 自带的 TensorRT Python 接口、numpy。
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
. "$VENV/bin/activate"
python -m pip install -q --upgrade "pip<24.1"
pip install -q "$WHEEL_DIR/$WHEEL"
pip install -q -r "$REPO/requirements-orin.txt"

# 4. 验证：GPU 可用，能编译 CUDA 扩展（需要 nvcc 在 PATH 里）。
export PATH=/usr/local/cuda/bin:$PATH
python - <<PY
import torch
assert torch.cuda.is_available(), "torch 看不到 GPU：确认没在 conda 里，并且用的是系统 Python 3.8"
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
echo "完成。以后使用前：source $VENV/bin/activate && export PATH=/usr/local/cuda/bin:\$PATH"
