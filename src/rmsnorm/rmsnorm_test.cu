#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include "rmsNorm.cu"

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Buffer {
    float* data = nullptr;
    explicit Buffer(size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

int main() {
    try {
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << '\n';
        constexpr int width = 256;
        constexpr float eps = 1e-5f;
        constexpr double atol = 2e-6, rtol = 2e-5;
        std::cout << "FP32; eps=" << eps << "; atol=" << atol << "; rtol=" << rtol << '\n';
        const char* names[] = {"ones_unit_weight", "zero_varied_weight",
                               "mixed_sign_varied_weight", "tiny_input", "large_input"};
        int cases = 0;
        for (int rows : {1, 3, 17}) {
            for (int mode = 0; mode < 5; ++mode) {
                const size_t count = size_t(rows) * width;
                std::vector<float> x(count), weight(width), actual(count), reference(count);
                for (int col = 0; col < width; ++col)
                    weight[col] = mode == 0 ? 1.0f : float(col % 13 - 6) * 0.25f;
                for (int row = 0; row < rows; ++row) {
                    double sum = 0;
                    for (int col = 0; col < width; ++col) {
                        float value = float((col * 7 + row * 11) % 37 - 18) * 0.125f
                                      + float(row) * 0.03125f;
                        if (mode == 0) value = 1.0f;
                        if (mode == 1) value = 0.0f;
                        if (mode == 3) value *= 1e-7f;
                        if (mode == 4) value *= 1e3f;
                        x[size_t(row) * width + col] = value;
                        sum += double(value) * value;
                    }
                    const double scale = 1.0 / std::sqrt(sum / width + double(eps));
                    for (int col = 0; col < width; ++col)
                        reference[size_t(row) * width + col] = float(
                            double(x[size_t(row) * width + col]) * scale * weight[col]);
                }
                Buffer dx(count), dw(width), dy(count);
                check(cudaMemcpy(dx.data, x.data(), count * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dw.data, weight.data(), width * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemset(dy.data, 0xff, count * sizeof(float)));
                rmsNorm256<<<rows, width>>>(dx.data, dw.data, dy.data, eps);
                check(cudaGetLastError());
                check(cudaDeviceSynchronize());
                check(cudaMemcpy(actual.data(), dy.data, count * sizeof(float), cudaMemcpyDeviceToHost));
                double maxAbs = 0, maxRatio = 0;
                size_t failures = 0;
                for (size_t i = 0; i < count; ++i) {
                    const double error = std::abs(double(actual[i]) - reference[i]);
                    const double limit = atol + rtol * std::abs(double(reference[i]));
                    if (!std::isfinite(actual[i]) || error > limit) ++failures;
                    maxAbs = std::max(maxAbs, error);
                    maxRatio = std::max(maxRatio, error / limit);
                }
                std::cout << "rows=" << rows << " case=" << names[mode]
                          << " checked=" << count << " max_abs=" << std::setprecision(9)
                          << maxAbs << " max_tolerance_ratio=" << maxRatio
                          << " " << (failures ? "FAIL" : "PASS") << '\n';
                if (failures) return 1;
                ++cases;
            }
        }
        std::cout << "ALL " << cases << " CASES PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
