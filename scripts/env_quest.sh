# Quest A100 上的 PyTorch 环境：source scripts/env_quest.sh
# torch 2.14.1+cu130 / Triton 3.8.0 在项目共享的 conda 环境 aiIfa 里；
# 编译 PyTorch CUDA 扩展还需要和 torch 同一大版本的 nvcc（CUDA 13）以及 ninja。
module purge
module load gcc/12.4.0-gcc-8.5.0 cuda/13.0.0-gcc-12.4.0 ninja/1.11.1-gcc-12.3.0
export CONDA_ENV=/projects/p53267/ai_infra/conda-envs/aiIfa
export PATH="$CONDA_ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0   # 只为 A100 编译，缩短扩展编译时间
