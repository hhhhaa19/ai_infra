// 手写转置的测试台：对每个形状，跑 kernels.cuh 里已实现的版本，先对答案再计时。
// 用法：transpose_bench          先跑几组"不整齐"的小形状查正确性，再跑几组大形状计时
//       transpose_bench M N      只跑这一个形状（A 是 M×N）
// 参考答案：CPU 上直接转置。转置只搬数据不做计算，所以要求逐元素完全相等。
// 检查：B 先填 NaN，漏写的位置会被查出来；报告不相等的元素个数（0 才算通过）。
// 计时：CUDA event 包住单次 kernel，预热 5 次，取 20 次中位数。
// 指标：有效带宽 GB/s = 读 A + 写 B 的字节数 / 时间 = 2·M·N·4 / 时间；copy 一行是同样数据量的直接复制，当作天花板。
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    {"shared", launch_shared, false},
    {"shared_pad", launch_shared_pad, false},
};

// 不相等的元素个数；按位比较，NaN（没写到）也算不相等
static size_t count_mismatch(const std::vector<float>& got, const std::vector<float>& ref) {
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (std::memcmp(&got[i], &ref[i], sizeof(float)) != 0) ++bad;
    return bad;
}

static void run_shape(int M, int N, bool timing) {
    const size_t n = size_t(M) * N;
    std::vector<float> hA(n), hB(n), ref(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(rng);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) ref[size_t(j) * M + i] = hA[size_t(i) * N + j];

    float *dA, *dB;
    CHECK_CUDA(cudaMalloc(&dA, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    const double bytes = 2.0 * n * sizeof(float);
    std::printf("\nA: M=%d N=%d  (%.1f MiB read + %.1f MiB write)\n", M, N, bytes / 2 / 1048576, bytes / 2 / 1048576);
    double copy_ms = 0;
    std::vector<std::pair<const char*, double>> times;
    for (const Impl& impl : kImpls) {
        CHECK_CUDA(cudaMemset(dB, 0xff, n * sizeof(float)));   // 先填 NaN
        if (!impl.fn(dA, dB, M, N, s)) {
            std::printf("  %-10s not implemented\n", impl.name);
            continue;
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(s));
        CHECK_CUDA(cudaMemcpy(hB.data(), dB, n * sizeof(float), cudaMemcpyDeviceToHost));
        const size_t bad = count_mismatch(hB, impl.is_copy ? hA : ref);
        if (!timing || bad) {
            std::printf("  %-10s mismatch %zu / %zu  %s\n", impl.name, bad, n, bad ? "FAIL" : "PASS");
            continue;
        }
        for (int i = 0; i < 5; ++i) impl.fn(dA, dB, M, N, s);
        std::vector<float> ms(20);
        for (float& t : ms) {
            CHECK_CUDA(cudaEventRecord(e0, s));
            impl.fn(dA, dB, M, N, s);
            CHECK_CUDA(cudaEventRecord(e1, s));
            CHECK_CUDA(cudaEventSynchronize(e1));
            CHECK_CUDA(cudaEventElapsedTime(&t, e0, e1));
        }
        std::sort(ms.begin(), ms.end());
        const double med = (ms[9] + ms[10]) / 2;
        if (impl.is_copy) copy_ms = med; else times.push_back({impl.name, med});
        std::printf("  %-10s PASS  %9.3f ms  %7.1f GB/s\n", impl.name, med, bytes / med / 1e6);
    }
    if (copy_ms > 0 && !times.empty()) {
        std::printf("  vs copy:");
        for (auto& t : times) std::printf("  %s %.0f%%", t.first, 100 * copy_ms / t.second);
        std::printf("\n");
    }
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaStreamDestroy(s));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
}

int main(int argc, char** argv) {
    cudaDeviceProp p{};
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    std::printf("GPU: %s, %d SMs, L2 %.1f MiB\n", p.name, p.multiProcessorCount, p.l2CacheSize / 1048576.0);
    std::printf("(on Orin lock clocks with jetson_clocks before timing; DRAM peak ≈ 204.8 GB/s)\n");
    if (argc == 3) {
        run_shape(std::atoi(argv[1]), std::atoi(argv[2]), true);
    } else {
        const int small[][2] = {{64, 48}, {127, 129}, {1, 300}, {200, 3}, {33, 65}};
        for (auto& sh : small) run_shape(sh[0], sh[1], false);
        const int big[][2] = {{4096, 4096}, {2048, 8192}, {8192, 8192}};
        for (auto& sh : big) run_shape(sh[0], sh[1], true);
    }
}
