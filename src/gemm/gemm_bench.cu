// 手写 GEMM 的测试台：对每个形状，跑 kernels.cuh 里已实现的版本和 cuBLAS，先对答案再计时。
// 用法：gemm_bench            先跑几组"不整齐"的小形状查正确性，再跑 1024 / 2048 / 4096 方阵计时
//       gemm_bench M N K      只跑这一个形状
// 参考答案：小形状（M·N·K ≤ 2^24）用 CPU double 三重循环；大形状用 cuBLAS（FP32，不开 TF32）。
// 指标：rel_l2 = ‖C − ref‖ / ‖ref‖，< 1e-5 算通过（只是累加顺序不同，FP32 下一般 1e-7 量级）。
// 计时：CUDA event 包住单次 kernel，预热 5 次，取 20 次中位数；GFLOPS = 2·M·N·K / 时间。
#include <cublas_v2.h>
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
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status_ = (call); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d: cuBLAS error %d\n", __FILE__, __LINE__, static_cast<int>(status_)); \
        std::exit(1); \
    } \
} while (0)

using LaunchFn = bool (*)(const float*, const float*, float*, int, int, int, cudaStream_t);

static cublasHandle_t g_cublas;

// cuBLAS 是列优先。行优先的 C = A·B 等于列优先的 Cᵀ = Bᵀ·Aᵀ，所以把 B、A 对调传进去，不需要真的转置。
static bool launch_cublas(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSetStream(g_cublas, s));
    CHECK_CUBLAS(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
    return true;
}

struct Impl { const char* name; LaunchFn fn; };
static const Impl kImpls[] = {
    {"naive", launch_naive}, {"tiled", launch_tiled}, {"reg", launch_reg}, {"cublas", launch_cublas},
};

static double rel_l2(const std::vector<float>& c, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        if (!std::isfinite(c[i])) return INFINITY;   // 没写到的位置还是 NaN
        num += (c[i] - ref[i]) * (c[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / den);
}

static void run_shape(int M, int N, int K, bool timing) {
    const size_t nA = size_t(M) * K, nB = size_t(K) * N, nC = size_t(M) * N;
    std::vector<float> hA(nA), hB(nB), hC(nC);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, nA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, nB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, nC * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));
    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));

    // 参考答案
    std::vector<double> ref(nC);
    const bool cpu_ref = double(M) * N * K <= double(1 << 24);
    if (cpu_ref) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double acc = 0;
                for (int k = 0; k < K; ++k) acc += double(hA[size_t(i) * K + k]) * hB[size_t(k) * N + j];
                ref[size_t(i) * N + j] = acc;
            }
    } else {
        launch_cublas(dA, dB, dC, M, N, K, s);
        CHECK_CUDA(cudaStreamSynchronize(s));
        CHECK_CUDA(cudaMemcpy(hC.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
        std::copy(hC.begin(), hC.end(), ref.begin());
    }

    const double flops = 2.0 * M * N * K;
    std::printf("\nM=%d N=%d K=%d  (reference: %s)\n", M, N, K, cpu_ref ? "CPU double" : "cuBLAS");
    double cublas_ms = 0;
    std::vector<std::pair<const char*, double>> times;
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    for (const Impl& impl : kImpls) {
        CHECK_CUDA(cudaMemset(dC, 0xff, nC * sizeof(float)));   // 先填 NaN，漏写的位置会被查出来
        if (!impl.fn(dA, dB, dC, M, N, K, s)) {
            std::printf("  %-7s not implemented\n", impl.name);
            continue;
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(s));
        CHECK_CUDA(cudaMemcpy(hC.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
        const double err = rel_l2(hC, ref);
        const bool pass = err < 1e-5;
        if (!timing || !pass) {
            std::printf("  %-7s rel_l2 %.2e  %s\n", impl.name, err, pass ? "PASS" : "FAIL");
            continue;
        }
        for (int i = 0; i < 5; ++i) impl.fn(dA, dB, dC, M, N, K, s);
        std::vector<float> ms(20);
        for (float& t : ms) {
            CHECK_CUDA(cudaEventRecord(e0, s));
            impl.fn(dA, dB, dC, M, N, K, s);
            CHECK_CUDA(cudaEventRecord(e1, s));
            CHECK_CUDA(cudaEventSynchronize(e1));
            CHECK_CUDA(cudaEventElapsedTime(&t, e0, e1));
        }
        std::sort(ms.begin(), ms.end());
        const double med = (ms[9] + ms[10]) / 2;
        times.push_back({impl.name, med});
        if (impl.fn == launch_cublas) cublas_ms = med;
        std::printf("  %-7s rel_l2 %.2e  PASS  %9.3f ms  %8.1f GFLOPS\n", impl.name, err, med, flops / med / 1e6);
    }
    if (cublas_ms > 0 && times.size() > 1) {
        std::printf("  vs cuBLAS:");
        for (auto& t : times) if (t.second != cublas_ms) std::printf("  %s %.0f%%", t.first, 100 * cublas_ms / t.second);
        std::printf("\n");
    }
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaStreamDestroy(s));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
}

int main(int argc, char** argv) {
    cudaDeviceProp p{};
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    // FP32 核心数 / SM：sm_80 是 64，sm_86 / 87 / 89 / 90 是 128
    const int cores = (p.major == 8 && p.minor == 0) ? 64 : 128;
    int clock_khz = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    std::printf("GPU: %s, %d SMs × %d FP32 cores, max clock %.0f MHz → FP32 peak ≈ %.0f GFLOPS\n",
                p.name, p.multiProcessorCount, cores, clock_khz / 1e3,
                2.0 * p.multiProcessorCount * cores * clock_khz / 1e6);
    std::printf("(peak uses the max clock; on Orin lock clocks with jetson_clocks before timing)\n");
    CHECK_CUBLAS(cublasCreate(&g_cublas));

    if (argc == 4) {
        run_shape(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), true);
    } else {
        const int small[][3] = {{64, 48, 37}, {127, 129, 65}, {1, 300, 77}, {200, 3, 513}};
        for (auto& sh : small) run_shape(sh[0], sh[1], sh[2], false);
        for (int n : {1024, 2048, 4096}) run_shape(n, n, n, true);
    }
    CHECK_CUBLAS(cublasDestroy(g_cublas));
}
