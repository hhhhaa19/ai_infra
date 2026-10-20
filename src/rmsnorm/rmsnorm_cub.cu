// 阅读示例：单个 block 处理一行；FP32 输入/输出，所有行共享 weight。
// kernel 在 rmsnorm_cub.cuh；先读 kernel，再读 main 中的数据与校验流程。
#include <cuda_runtime.h>
#include "rmsnorm_cub.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// RAII：对象销毁时释放 GPU 显存；禁止复制，防止重复释放。
struct DeviceBuffer {
    float* data = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(&data, count * sizeof(float)));
    }
    ~DeviceBuffer() {
        if (data) cudaFree(data);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

int main() try {
    constexpr int rows = 3;
    constexpr int n = 1003;          // 故意不是线程数的整数倍
    constexpr int threads = 256;
    constexpr float eps = 1e-5f;
    constexpr double atol = 2e-6;
    constexpr double rtol = 2e-5;
    const std::size_t count = std::size_t(rows) * n;
    const std::size_t bytes = count * sizeof(float);

    std::vector<float> x(count), weight(n), actual(count);
    std::vector<double> reference(count);

    // A. CPU 生成不同的行，以及所有行共用的权重。
    for (int col = 0; col < n; ++col) {
        weight[col] = float(col % 13 - 6) * 0.25f;
    }
    for (int row = 0; row < rows; ++row) {
        const std::size_t base = std::size_t(row) * n;
        double sum = 0.0;
        for (int col = 0; col < n; ++col) {
            float value = float((col * 7 + row * 11) % 37 - 18) * 0.125f;
            x[base + col] = value;
            sum += double(value) * value;
        }
        // CPU 双精度参考值：最终比较允许浮点误差。
        const double scale = 1.0 / std::sqrt(sum / n + double(eps));
        for (int col = 0; col < n; ++col) {
            reference[base + col] = double(x[base + col]) * scale * weight[col];
        }
    }

    // B. 分配显存并拷入数据。
    DeviceBuffer dx(count), dw(n), dy(count);
    check(cudaMemcpy(dx.data, x.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw.data, weight.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemset(dy.data, 0xff, bytes));  // NaN：帮助发现未写入的输出

    // C. 启动一个 kernel。模板参数与实际 block 线程数必须一致。
    // TempStorage 是静态共享内存，不需要传入第三个启动参数。
    rmsNormCub<threads><<<rows, threads>>>(dx.data, dw.data, dy.data, n, eps);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    // D. 拷回并逐元素校验，不只看前五个数。
    check(cudaMemcpy(actual.data(), dy.data, bytes, cudaMemcpyDeviceToHost));
    double maxAbs = 0.0;
    std::size_t failures = 0;
    for (std::size_t i = 0; i < count; ++i) {
        double error = std::abs(double(actual[i]) - reference[i]);
        double limit = atol + rtol * std::abs(reference[i]);
        if (!std::isfinite(actual[i]) || error > limit) {
            if (failures == 0) {
                std::cerr << "First mismatch: index=" << i
                          << " CPU=" << reference[i] << " GPU=" << actual[i] << '\n';
            }
            ++failures;
        }
        maxAbs = std::max(maxAbs, error);
    }
    std::cout << "Shape: [" << rows << ", " << n << "]\n"
              << "Threads per block: " << threads << '\n'
              << "Max absolute error: " << maxAbs << '\n'
              << "Failures: " << failures << '/' << count << '\n';
    if (failures != 0) return 1;
    std::cout << "PASS: all outputs match CPU reference within tolerance\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
}
