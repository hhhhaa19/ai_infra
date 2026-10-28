from pathlib import Path
import numpy as np

ref_path = Path("data/output_ref.bin")
actual_path = Path("data/output_trt.bin")

# [1, 1000] 的 FP32 输出应为 4000 字节。
expected_bytes = 1000 * np.dtype(np.float32).itemsize

for path in (ref_path, actual_path):
    assert path.stat().st_size == expected_bytes, (
        f"{path}: 文件大小不是 {expected_bytes} 字节"
    )

reference = np.fromfile(ref_path, dtype=np.float32)
actual = np.fromfile(actual_path, dtype=np.float32)

assert np.isfinite(reference).all(), "参考输出包含 NaN/Inf"
assert np.isfinite(actual).all(), "TensorRT 输出包含 NaN/Inf"

# 用 float64 计算误差，原始数据仍来自 FP32 文件。
error = np.abs(
    actual.astype(np.float64) - reference.astype(np.float64)
)

rtol = 1e-4
atol = 1e-5
tolerance = atol + rtol * np.abs(reference.astype(np.float64))
passed = error <= tolerance

worst = int(error.argmax())

print("Max absolute error:", error.max())
print("Mean absolute error:", error.mean())
print("Elements outside tolerance:", int((~passed).sum()))
print(f"Worst index: {worst}")
print(f"  PyTorch:  {reference[worst]:.9g}")
print(f"  TensorRT: {actual[worst]:.9g}")
print("Top-1 index:", reference.argmax(), actual.argmax())

np.testing.assert_allclose(
    actual, reference, rtol=rtol, atol=atol
)

print("PASS: 全部 1000 个输出在设定容差内")