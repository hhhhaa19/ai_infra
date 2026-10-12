import numpy as np
import onnx
import onnxruntime as ort

# 1. 检查图结构
model_path = "data/resnet50.onnx"
onnx.checker.check_model(model_path)
print("ONNX 结构检查通过")

# 2. 读取原来保存的同一份输入和 PyTorch 参考输出
x = np.fromfile("data/input.bin", dtype=np.float32)
reference = np.fromfile("data/output_ref.bin", dtype=np.float32)

assert x.size == 1 * 3 * 224 * 224, "输入文件大小不对"
assert reference.size == 1000, "参考输出文件大小不对"

x = x.reshape(1, 3, 224, 224)
reference = reference.reshape(1, 1000)

# 3. 使用 CPU 执行 ONNX 模型
session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

actual = session.run(
    [output_name],
    {input_name: x},
)[0]

# 4. 检查输出，再计算误差
assert actual.shape == reference.shape, "输出形状不一致"
assert np.isfinite(reference).all(), "参考输出包含 NaN/Inf"
assert np.isfinite(actual).all(), "ONNX 输出包含 NaN/Inf"

error = np.abs(actual - reference)

print("输出形状：", actual.shape)
print("最大绝对误差：", error.max())
print("平均绝对误差：", error.mean())
print("PyTorch 前 5 个值：", reference[0, :5])
print("ONNX    前 5 个值：", actual[0, :5])

# 先采用这组 FP32 验证容差；失败时检查原因，不直接放宽
np.testing.assert_allclose(
    actual,
    reference,
    rtol=1e-4,
    atol=1e-5,
)

print("数值对齐通过")