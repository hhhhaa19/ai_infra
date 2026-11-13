// 手写 scan（前缀和）的测试台：对每个长度 n，跑 kernels.cuh 里已实现的版本，先对答案再计时。
// 用法：scan_bench        先跑几组"不整齐"的小长度查正确性，再跑几组大长度计时
//       scan_bench n      只跑这一个长度
// 做的是 inclusive scan：Y[i] = X[0] + X[1] + … + X[i]。
// 数据：X 是 [-8, 8] 的随机整数。用 int 是为了精确比较（浮点加法换顺序结果会差一点，int 不会）；
//       范围小、有正有负，2^26 个数加起来也不会溢出 int。
// 检查：和 CPU 串行结果逐个比较，必须完全相等；不对时打印第一个错的位置，方便查。
//       Y 先填 0xAAAAAAAA（一个很大的负数），漏写的位置会被查出来。
// 计时：CUDA event 包住单次 launch（多 kernel 的版本把几个 kernel 都算进去），预热 5 次，取 20 次中位数。
// 指标：有效带宽 GB/s = 读 X 一遍 + 写 Y 一遍的字节数 / 时间 = 2·n·4 / 时间；copy 是同样数据量的直接复制，当作天花板。
//       cub 是 CUDA 自带的 CUB 库（cub::DeviceScan::InclusiveSum），当作"工业界做到多快"的参考，和 GEMM 里的 cuBLAS 一样。
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
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

// CUB 参考：第一次调用时问 CUB 要多少临时空间，之后复用
static void* g_cub_tmp = nullptr;
static size_t g_cub_bytes = 0;
static bool launch_cub(const int* X, int* Y, int n, int* /*work*/, cudaStream_t s) {
    size_t need = 0;
    cub::DeviceScan::InclusiveSum(nullptr, need, X, Y, n, s);
    if (need > g_cub_bytes) {
        if (g_cub_tmp) CHECK_CUDA(cudaFree(g_cub_tmp));
        CHECK_CUDA(cudaMalloc(&g_cub_tmp, need));
        g_cub_bytes = need;
    }
    cub::DeviceScan::InclusiveSum(g_cub_tmp, g_cub_bytes, X, Y, n, s);
    return true;
}

using LaunchFn = bool (*)(const int*, int*, int, int*, cudaStream_t);

struct Impl { const char* name; LaunchFn fn; bool is_copy; };
static const Impl kImpls[] = {
    {"copy", launch_copy, true},
    {"block", launch_block, false},
    {"multi", launch_multi, false},
    {"warp", launch_warp, false},
    {"lookback", launch_lookback, false},
    {"cub", launch_cub, false},
};

static void run_n(int n, bool timing) {
    std::vector<int> hX(n), hY(n), ref(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-8, 8);
    for (int& v : hX) v = dist(rng);
    int acc = 0;
    for (int i = 0; i < n; ++i) { acc += hX[i]; ref[i] = acc; }

    int *dX, *dY, *dWork;
    CHECK_CUDA(cudaMalloc(&dX, size_t(n) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dY, size_t(n) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dWork, kWorkInts(n) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dX, hX.data(), size_t(n) * sizeof(int), cudaMemcpyHostToDevice));
    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    const double bytes = 2.0 * n * sizeof(int);
    std::printf("\nn=%d  (%.1f MiB read + %.1f MiB write)\n", n, bytes / 2 / 1048576, bytes / 2 / 1048576);
    double copy_ms = 0;
    std::vector<std::pair<const char*, double>> times;
    for (const Impl& impl : kImpls) {
        CHECK_CUDA(cudaMemset(dY, 0xAA, size_t(n) * sizeof(int)));
        if (!impl.fn(dX, dY, n, dWork, s)) {
            std::printf("  %-8s not implemented / skipped for this n\n", impl.name);
            continue;
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(s));
        CHECK_CUDA(cudaMemcpy(hY.data(), dY, size_t(n) * sizeof(int), cudaMemcpyDeviceToHost));
        const std::vector<int>& want = impl.is_copy ? hX : ref;
        int bad = -1;
        for (int i = 0; i < n; ++i) if (hY[i] != want[i]) { bad = i; break; }
        if (bad >= 0) {
            std::printf("  %-8s FAIL  first wrong i=%d: got %d, want %d\n", impl.name, bad, hY[bad], want[bad]);
            continue;
        }
        if (!timing) { std::printf("  %-8s PASS\n", impl.name); continue; }
        for (int i = 0; i < 5; ++i) impl.fn(dX, dY, n, dWork, s);
        std::vector<float> ms(20);
        for (float& t : ms) {
            CHECK_CUDA(cudaEventRecord(e0, s));
            impl.fn(dX, dY, n, dWork, s);
            CHECK_CUDA(cudaEventRecord(e1, s));
            CHECK_CUDA(cudaEventSynchronize(e1));
            CHECK_CUDA(cudaEventElapsedTime(&t, e0, e1));
        }
        std::sort(ms.begin(), ms.end());
        const double med = (ms[9] + ms[10]) / 2;
        if (impl.is_copy) copy_ms = med; else times.push_back({impl.name, med});
        std::printf("  %-8s PASS  %9.3f ms  %7.1f GB/s\n", impl.name, med, bytes / med / 1e6);
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
    CHECK_CUDA(cudaFree(dWork));
}

int main(int argc, char** argv) {
    cudaDeviceProp p{};
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    std::printf("GPU: %s, %d SMs, L2 %.1f MiB\n", p.name, p.multiProcessorCount, p.l2CacheSize / 1048576.0);
    std::printf("(on Orin lock clocks with jetson_clocks before timing; DRAM peak ≈ 204.8 GB/s)\n");
    if (argc == 2) {
        run_n(std::atoi(argv[1]), true);
    } else {
        // 小长度：1、不满一个 warp、刚好 / 刚超过一个 warp、刚好 / 刚超过 1024（一个 block）、
        //         多个 block 但不整齐、block 和数多到要再分一层
        const int small[] = {1, 31, 32, 33, 1000, 1024, 1025, 4097, 100003, 5000011};
        for (int n : small) run_n(n, false);
        // 大长度：1M（放得进 L2 附近）、16M、64M
        const int big[] = {1 << 20, 1 << 24, 1 << 26};
        for (int n : big) run_n(n, true);
    }
    if (g_cub_tmp) cudaFree(g_cub_tmp);
}
