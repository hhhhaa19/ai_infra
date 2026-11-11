// 手写 softmax 的测试台：对每个形状，跑 kernels.cuh 里已实现的版本，先对答案再计时。
// 用法：softmax_bench            先跑几组"不整齐"的小形状查正确性，再跑几组典型形状计时
//       softmax_bench rows n     只跑这一个形状
// 数据：X 在 [-10, 10] 均匀分布；每 7 行里有 1 行整体加 100（不减最大值的话 exp 会溢出成 inf，结果变 NaN）。
// 参考答案：CPU 上用 double 计算。检查每个元素的绝对误差，最大误差 < 1e-6 算通过；
//           Y 先填 NaN，漏写的位置会被查出来。
// 计时：CUDA event 包住单次 kernel，预热 5 次，取 20 次中位数。
// 指标：有效带宽 GB/s = 读 X 一遍 + 写 Y 一遍的字节数 / 时间 = 2·rows·n·4 / 时间（多读的遍数不算进去）；
//       copy 一行是同样数据量的直接复制，当作天花板。
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "kernels.cuh"

#define CHECK_CUDA(call) do { \
    cudaError_t status_ = (call); \
    if (status_ != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
        std::exit(1); \
    } \
} while (0)

using LaunchFn = bool (*)(const float*, float*, int, int, cudaStream_t);

struct Impl { const char* name; LaunchFn fn; bool is_copy; };
static const Impl kImpls[] = {
    {"copy", launch_copy, true},
    {"naive", launch_naive, false},
    {"block", launch_block, false},
    {"online", launch_online, false},
    {"warp", launch_warp, false},
};

// 最大绝对误差；NaN / inf 返回 inf
static double max_abs_err(const std::vector<float>& y, const std::vector<double>& ref) {
    double worst = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i])) return INFINITY;
        worst = std::max(worst, std::fabs(double(y[i]) - ref[i]));
    }
    return worst;
}

static void run_shape(int rows, int n, bool timing) {
    const size_t total = size_t(rows) * n;
    std::vector<float> hX(total), hY(total);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < n; ++j) hX[size_t(r) * n + j] = dist(rng) + (r % 7 == 3 ? 100.0f : 0.0f);

    std::vector<double> ref(total);
    for (int r = 0; r < rows; ++r) {
        const float* x = &hX[size_t(r) * n];
        double m = -INFINITY, d = 0;
        for (int j = 0; j < n; ++j) m = std::max(m, double(x[j]));
        for (int j = 0; j < n; ++j) d += std::exp(double(x[j]) - m);
        for (int j = 0; j < n; ++j) ref[size_t(r) * n + j] = std::exp(double(x[j]) - m) / d;
    }

    float *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dY, total * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dX, hX.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    const double bytes = 2.0 * total * sizeof(float);
    std::printf("\nrows=%d n=%d  (%.1f MiB read + %.1f MiB write)\n", rows, n, bytes / 2 / 1048576, bytes / 2 / 1048576);
    double copy_ms = 0;
    std::vector<std::pair<const char*, double>> times;
    for (const Impl& impl : kImpls) {
        CHECK_CUDA(cudaMemset(dY, 0xff, total * sizeof(float)));   // 先填 NaN
        if (!impl.fn(dX, dY, rows, n, s)) {
            std::printf("  %-7s not implemented / skipped for this shape\n", impl.name);
            continue;
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(s));
        CHECK_CUDA(cudaMemcpy(hY.data(), dY, total * sizeof(float), cudaMemcpyDeviceToHost));
        bool pass;
        double err = 0;
        if (impl.is_copy) {
            pass = std::equal(hY.begin(), hY.end(), hX.begin());
        } else {
            err = max_abs_err(hY, ref);
            pass = err < 1e-6;
        }
        if (!timing || !pass) {
            if (impl.is_copy) std::printf("  %-7s %s\n", impl.name, pass ? "PASS" : "FAIL");
            else std::printf("  %-7s max_abs_err %.2e  %s\n", impl.name, err, pass ? "PASS" : "FAIL");
            continue;
        }
        for (int i = 0; i < 5; ++i) impl.fn(dX, dY, rows, n, s);
        std::vector<float> ms(20);
        for (float& t : ms) {
            CHECK_CUDA(cudaEventRecord(e0, s));
            impl.fn(dX, dY, rows, n, s);
            CHECK_CUDA(cudaEventRecord(e1, s));
            CHECK_CUDA(cudaEventSynchronize(e1));
            CHECK_CUDA(cudaEventElapsedTime(&t, e0, e1));
        }
        std::sort(ms.begin(), ms.end());
        const double med = (ms[9] + ms[10]) / 2;
        if (impl.is_copy) copy_ms = med; else times.push_back({impl.name, med});
        if (impl.is_copy) std::printf("  %-7s PASS                    %9.3f ms  %7.1f GB/s\n", impl.name, med, bytes / med / 1e6);
        else std::printf("  %-7s max_abs_err %.2e  PASS  %9.3f ms  %7.1f GB/s\n", impl.name, err, med, bytes / med / 1e6);
    }
    if (copy_ms > 0 && !times.empty()) {
        std::printf("  vs copy:");
        for (auto& t : times) std::printf("  %s %.0f%%", t.first, 100 * copy_ms / t.second);
        std::printf("\n");
    }
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaStreamDestroy(s));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
}

int main(int argc, char** argv) {
    cudaDeviceProp p{};
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    std::printf("GPU: %s, %d SMs, L2 %.1f MiB\n", p.name, p.multiProcessorCount, p.l2CacheSize / 1048576.0);
    std::printf("(on Orin lock clocks with jetson_clocks before timing; DRAM peak ≈ 204.8 GB/s)\n");
    if (argc == 3) {
        run_shape(std::atoi(argv[1]), std::atoi(argv[2]), true);
    } else {
        // 小形状：n=1、不是 32 的倍数、超过 blockDim、刚好超过 warp 版上限
        const int small[][2] = {{3, 1}, {5, 33}, {7, 1000}, {9, 1025}, {2, 4099}};
        for (auto& sh : small) run_shape(sh[0], sh[1], false);
        // 典型形状：attention 里 rows = batch·heads·seq、n = seq；最后一行是词表大小 32000 的采样 softmax
        const int big[][2] = {{65536, 128}, {16384, 1024}, {4096, 4096}, {512, 32000}};
        for (auto& sh : big) run_shape(sh[0], sh[1], true);
    }
}
