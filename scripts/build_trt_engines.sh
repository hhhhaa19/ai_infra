#!/bin/bash
# 用 trtexec 把 export_onnx.py 导出的 ONNX 建成 engine，再测时间（Orin，TensorRT 8.5，FP32）。
# 用法（在仓库根目录）：scripts/build_trt_engines.sh <eager|ours> [onnx 目录]
#   eager：RMSNorm 由 TensorRT 自己处理；ours：RmsNorm 节点用我们的 plugin
#          （先 cmake --build build-orin --target rmsnorm_trt_plugin）
#   onnx 目录默认 build-onnx/layer_<impl>，即 python src/llama/export_onnx.py <impl> 的输出目录
# 建 engine 时 TensorRT 会实测挑 kernel，要先锁频（sudo jetson_clocks）。
# 每种形状在 onnx 目录里产出：
#   <impl>_<形状>_fp32.engine     engine 本身（python src/llama/trt_check.py 用它查正确性）
#   *_layers.json                 每层用了什么 kernel（tactic）
#   *_profile.json                每层时间
#   trt_build_*.log               建 engine + 普通计时；trt_cg_*.log CUDA Graph 计时；trt_prof_*.log 逐层计时
set -euo pipefail
impl=${1:?"用法：scripts/build_trt_engines.sh <eager|ours> [onnx 目录]"}
dir=${2:-build-onnx/layer_$impl}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
plugin_args=()
if [ "$impl" != eager ]; then
    plugin_args=(--plugins="$PWD/build-orin/librmsnorm_trt_plugin.so")
fi
# --noTF32：FP32 矩阵乘不偷偷用 TF32，和 PyTorch 公平比较
# --noDataTransfers：不计 CPU<->GPU 拷贝；--useSpinWait：CPU 忙等 GPU，计时更稳
common=(--noDataTransfers --useSpinWait)

cd "$dir"
for s in decode_b1 decode_b8 prefill_s512; do
    e=${impl}_${s}_fp32
    "$TRTEXEC" --onnx=layer_${impl}_${s}.onnx --saveEngine=$e.engine "${plugin_args[@]}" "${common[@]}" \
        --noTF32 --profilingVerbosity=detailed --exportLayerInfo=${e}_layers.json > trt_build_$e.log 2>&1
    "$TRTEXEC" --loadEngine=$e.engine "${plugin_args[@]}" "${common[@]}" --useCudaGraph > trt_cg_$e.log 2>&1
    "$TRTEXEC" --loadEngine=$e.engine "${plugin_args[@]}" "${common[@]}" \
        --dumpProfile --separateProfileRun --exportProfile=${e}_profile.json > trt_prof_$e.log 2>&1
    median() { grep "GPU Compute Time:" "$1" | grep -o "median = [0-9.]* ms" | cut -d" " -f3; }
    echo "$s: $(median trt_build_$e.log) ms | CUDA Graph $(median trt_cg_$e.log) ms"
done
