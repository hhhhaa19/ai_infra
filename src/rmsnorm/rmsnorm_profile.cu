// Nsight Compute 采集：reports/rmsnorm/20261017_multiline 的下一步 1。
// 比较 rmsNormN（rmsNormWarp.cu）与 rmsNormCub 在 [4096,256]、[4096,4096] 上的 128/256/512 线程配置。
// 每个配置预热后只在 cudaProfilerStart/Stop 之间 launch 1 次，共 12 次；配合 ncu --profile-from-start off。
// 数据生成与 rmsnorm_n_bench.cu 一致；kernel 实现未改动。
#include "rmsNormWarp.cu"
#include "rmsnorm_cub.cuh"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    float* data = nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(&data, count * sizeof(float))); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

enum Method { OWN, CUB };
const char* name(Method m) { return m == OWN ? "own" : "cub"; }

void launch(Method m, int threads, int rows, const float* x, const float* w, float* y, int n, float eps) {
    if (m == OWN) {
        rmsNormN<<<rows, threads, (threads / 32) * sizeof(float)>>>(x, w, y, n, eps);
        return;
    }
    switch (threads) {
        case 128: rmsNormCub<128><<<rows, 128>>>(x, w, y, n, eps); break;
        case 256: rmsNormCub<256><<<rows, 256>>>(x, w, y, n, eps); break;
        case 512: rmsNormCub<512><<<rows, 512>>>(x, w, y, n, eps); break;
        default: throw std::runtime_error("Unsupported CUB thread count");
    }
}

// 理论 occupancy：不需要 GPU 性能计数器权限。
int blocksPerSM(Method m, int threads) {
    int blocks = 0;
    if (m == OWN) {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, rmsNormN, threads, (threads / 32) * sizeof(float)));
    } else if (threads == 128) {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, rmsNormCub<128>, 128, 0));
    } else if (threads == 256) {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, rmsNormCub<256>, 256, 0));
    } else {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, rmsNormCub<512>, 512, 0));
    }
    return blocks;
}

int main() try {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));
    constexpr float eps = 1e-5f;
    constexpr int rows = 4096;
    std::cout << "GPU: " << prop.name << " SMs=" << prop.multiProcessorCount
              << "\nrows=" << rows << "; eps=1e-5; 12 captured launches\n";

    std::cout << "OCCUPANCY method,threads,blocks_per_sm,warps_per_sm\n";
    for (Method m : {OWN, CUB}) for (int threads : {128, 256, 512}) {
        int b = blocksPerSM(m, threads);
        std::cout << "OCCUPANCY " << name(m) << ',' << threads << ',' << b << ',' << b * threads / 32 << '\n';
    }

    int captured = 0;
    for (int n : {256, 4096}) {
        const size_t count = size_t(rows) * n;
        std::vector<float> x(count), w(n), actual(count);
        std::vector<double> reference(count);
        for (int c = 0; c < n; ++c) w[c] = float(c % 19 - 9) / 7;
        for (int row = 0; row < rows; ++row) {
            double sum = 0;
            for (int c = 0; c < n; ++c) {
                const size_t i = size_t(row) * n + c;
                x[i] = float((c * 7 + row * 13) % 97 - 48) / 13;
                sum += double(x[i]) * x[i];
            }
            const double scale = 1 / std::sqrt(sum / n + double(eps));
            for (int c = 0; c < n; ++c) {
                const size_t i = size_t(row) * n + c;
                reference[i] = double(x[i]) * w[c] * scale;
            }
        }
        Buffer dx(count), dw(n), dy(count);
        check(cudaMemcpy(dx.data, x.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dw.data, w.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        for (Method m : {OWN, CUB}) for (int threads : {128, 256, 512}) {
            // 该形状/实现/线程数单独预热至少 1 秒。
            auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            do {
                for (int i = 0; i < 20; ++i) launch(m, threads, rows, dx.data, dw.data, dy.data, n, eps);
                check(cudaGetLastError());
                check(cudaDeviceSynchronize());
            } while (std::chrono::steady_clock::now() < until);
            // 在采集范围外写入 NaN，用于发现未写入的输出。
            check(cudaMemset(dy.data, 0xff, count * sizeof(float)));
            check(cudaDeviceSynchronize());
            std::cout << "CAPTURE rows=" << rows << " n=" << n << " method=" << name(m)
                      << " threads=" << threads << std::endl;
            check(cudaProfilerStart());
            launch(m, threads, rows, dx.data, dw.data, dy.data, n, eps);
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
            check(cudaProfilerStop());
            check(cudaMemcpy(actual.data(), dy.data, count * sizeof(float), cudaMemcpyDeviceToHost));
            double maxAbs = 0;
            for (size_t i = 0; i < count; ++i) {
                double error = std::abs(double(actual[i]) - reference[i]);
                if (!std::isfinite(actual[i]) || error > 2e-6 + 2e-5 * std::abs(reference[i]))
                    throw std::runtime_error("Validation failed at " + std::to_string(i));
                maxAbs = std::max(maxAbs, error);
            }
            std::cout << "PASS max_abs=" << maxAbs << std::endl;
            ++captured;
        }
    }
    std::cout << "ALL " << captured << " PROFILE CASES PASS\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
