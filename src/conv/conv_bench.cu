// 手写卷积的测试台：对每组形状，跑 kernels.cuh 里已实现的版本，先对答案再计时。
// 用法：conv_bench          跑全部形状（前几组小形状只查正确性，后面的真实形状查正确性并计时）
//       conv_bench 名字      只跑这一组，比如 conv_bench resnet3x3
// 数据：X、Wt 都是 [-1, 1] 的随机 float，固定种子。
// 检查：CPU 用 double 累加算参考答案，逐个比较，|got − ref| ≤ 1e-3 + 1e-3·|ref| 算对；
//       不对时打印第一个错的位置 (n,k,p,q)。Y 先填 NaN，漏写的位置一定会被查出来。
// 计时：CUDA event 包住单次 launch，预热 3 次，取 20 次中位数。
// 指标：TFLOPS = 2·N·K·P·Q·C·R·S / 时间（每个乘加算 2 次浮点运算）。
//       Orin FP32 峰值约 5.3 TFLOPS（2048 CUDA 核 × 2 × 1.3 GHz），A100 约 19.5。
// 参考：编译时加 -DUSE_CUDNN（并链接 -lcudnn）会多跑一个 cuDNN，当作"工业界做到多快"，和 GEMM 里的 cuBLAS 一样。
//       cuDNN 设成 CUDNN_FMA_MATH：只用普通 FP32 乘加，不用 TF32 Tensor Core，和手写版本公平比较。
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "kernels.cuh"
#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#define CHECK_CUDA(call) do { \
    cudaError_t status_ = (call); \
    if (status_ != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
        std::exit(1); \
    } \
} while (0)

#ifdef USE_CUDNN
#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status_ = (call); \
    if (status_ != CUDNN_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status_)); \
        std::exit(1); \
    } \
} while (0)

// cuDNN 参考：每组形状第一次调用时建好描述符、选算法、分配 workspace，之后复用
struct CudnnState {
    cudnnHandle_t h = nullptr;
    cudnnTensorDescriptor_t xd, yd;
    cudnnFilterDescriptor_t wd;
    cudnnConvolutionDescriptor_t cd;
    cudnnConvolutionFwdAlgo_t algo;
    void* ws = nullptr;
    size_t ws_bytes = 0;
    ConvShape sh{};
    bool ready = false;
};
static CudnnState g_dnn;

static void cudnn_setup(const ConvShape& sh, cudaStream_t s) {
    if (!g_dnn.h) {
        CHECK_CUDNN(cudnnCreate(&g_dnn.h));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&g_dnn.xd));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&g_dnn.yd));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&g_dnn.wd));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&g_dnn.cd));
    }
    CHECK_CUDNN(cudnnSetStream(g_dnn.h, s));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(g_dnn.xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sh.N, sh.C, sh.H, sh.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(g_dnn.wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, sh.K, sh.C, sh.R, sh.S));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(g_dnn.cd, sh.pad, sh.pad, sh.stride, sh.stride, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnSetConvolutionMathType(g_dnn.cd, CUDNN_FMA_MATH));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(g_dnn.yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sh.N, sh.K, sh.P(), sh.Q()));
    cudnnConvolutionFwdAlgoPerf_t perf[8];
    int got = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(g_dnn.h, g_dnn.xd, g_dnn.wd, g_dnn.cd, g_dnn.yd, 8, &got, perf));
    g_dnn.algo = perf[0].algo;
    for (int i = 0; i < got; ++i)
        if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].mathType == CUDNN_FMA_MATH) { g_dnn.algo = perf[i].algo; break; }
    size_t need = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(g_dnn.h, g_dnn.xd, g_dnn.wd, g_dnn.cd, g_dnn.yd, g_dnn.algo, &need));
    if (need > g_dnn.ws_bytes) {
        if (g_dnn.ws) CHECK_CUDA(cudaFree(g_dnn.ws));
        CHECK_CUDA(cudaMalloc(&g_dnn.ws, need));
        g_dnn.ws_bytes = need;
    }
    g_dnn.sh = sh;
    g_dnn.ready = true;
}

static bool launch_cudnn(const float* X, const float* Wt, float* Y, const ConvShape& sh, cudaStream_t s) {
    if (!g_dnn.ready || std::memcmp(&g_dnn.sh, &sh, sizeof(sh)) != 0) cudnn_setup(sh, s);
    const float one = 1.f, zero = 0.f;
    CHECK_CUDNN(cudnnConvolutionForward(g_dnn.h, &one, g_dnn.xd, X, g_dnn.wd, Wt, g_dnn.cd, g_dnn.algo,
                                        g_dnn.ws, g_dnn.ws_bytes, &zero, g_dnn.yd, Y));
    return true;
}
#endif

using LaunchFn = bool (*)(const float*, const float*, float*, const ConvShape&, cudaStream_t);

struct Impl { const char* name; LaunchFn fn; };
static const Impl kImpls[] = {
    {"direct", launch_direct},
    {"igemm", launch_igemm},
#ifdef USE_CUDNN
    {"cudnn", launch_cudnn},
#endif
};

struct NamedShape { const char* name; ConvShape sh; bool timing; };
static const NamedShape kShapes[] = {
    //                       N    C    H    W    K    R   S  stride pad
    // 小形状，只查对错：和动画一样的 4×4 输入、两层、K=4；再来几组不整齐的
    {"tiny",            {1,   2,   4,   4,   4,   3,  3,  1,  0}, false},
    {"odd_pad",         {2,   3,   7,   9,   5,   3,  3,  1,  1}, false},
    {"odd_stride2",     {2,   3,  11,  10,   6,   3,  3,  2,  1}, false},
    {"one_by_one",      {1,  16,   9,   7,  10,   1,  1,  1,  0}, false},
    // 真实形状
    // ResNet 中间的 3×3 卷积：56×56，64 → 64，pad 1，输出还是 56×56
    {"resnet3x3",       {1,  64,  56,  56,  64,   3,  3,  1,  1}, true},
    // ResNet 下采样的 3×3 卷积：stride 2，128 → 256，28×28 → 14×14
    {"resnet3x3_s2",    {1, 128,  28,  28, 256,   3,  3,  2,  1}, true},
    // ResNet 的 1×1 卷积：其实就是矩阵乘
    {"resnet1x1",       {1, 256,  56,  56,  64,   1,  1,  1,  0}, true},
    // SmolVLA 视觉编码器（SigLIP）的 patch embed：3 个相机，512×512 RGB，16×16 不重叠切块，每块变成 768 维
    {"siglip_patch",    {3,   3, 512, 512, 768,  16, 16, 16,  0}, true},
};

static void cpu_conv(const std::vector<float>& X, const std::vector<float>& Wt, std::vector<float>& Y, const ConvShape& sh) {
    const int P = sh.P(), Q = sh.Q();
    for (int n = 0; n < sh.N; ++n)
    for (int k = 0; k < sh.K; ++k)
    for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q) {
        double sum = 0;
        for (int c = 0; c < sh.C; ++c)
        for (int r = 0; r < sh.R; ++r) {
            int h = p * sh.stride - sh.pad + r;
            if (h < 0 || h >= sh.H) continue;
            for (int s = 0; s < sh.S; ++s) {
                int w = q * sh.stride - sh.pad + s;
                if (w < 0 || w >= sh.W) continue;
                sum += double(X[((size_t(n) * sh.C + c) * sh.H + h) * sh.W + w]) *
                       double(Wt[((size_t(k) * sh.C + c) * sh.R + r) * sh.S + s]);
            }
        }
        Y[((size_t(n) * sh.K + k) * P + p) * Q + q] = float(sum);
    }
}

static void run_shape(const NamedShape& ns) {
    const ConvShape& sh = ns.sh;
    const int P = sh.P(), Q = sh.Q();
    const size_t nx = size_t(sh.N) * sh.C * sh.H * sh.W;
    const size_t nw = size_t(sh.K) * sh.C * sh.R * sh.S;
    const size_t ny = size_t(sh.N) * sh.K * P * Q;
    std::vector<float> hX(nx), hW(nw), hY(ny), ref(ny);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (float& v : hX) v = dist(rng);
    for (float& v : hW) v = dist(rng);
    cpu_conv(hX, hW, ref, sh);

    float *dX, *dW, *dY;
    CHECK_CUDA(cudaMalloc(&dX, nx * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dW, nw * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dY, ny * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dX, hX.data(), nx * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dW, hW.data(), nw * sizeof(float), cudaMemcpyHostToDevice));
    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    const double flops = 2.0 * sh.N * sh.K * P * Q * double(sh.C) * sh.R * sh.S;
    std::printf("\n%s  X[%d,%d,%d,%d] * W[%d,%d,%d,%d] stride %d pad %d -> Y[%d,%d,%d,%d]  (%.2f GFLOP)\n",
                ns.name, sh.N, sh.C, sh.H, sh.W, sh.K, sh.C, sh.R, sh.S, sh.stride, sh.pad,
                sh.N, sh.K, P, Q, flops / 1e9);
    for (const Impl& impl : kImpls) {
        std::vector<float> nan_fill(ny, NAN);
        CHECK_CUDA(cudaMemcpy(dY, nan_fill.data(), ny * sizeof(float), cudaMemcpyHostToDevice));
        if (!impl.fn(dX, dW, dY, sh, s)) {
            std::printf("  %-7s not implemented / skipped for this shape\n", impl.name);
            continue;
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(s));
        CHECK_CUDA(cudaMemcpy(hY.data(), dY, ny * sizeof(float), cudaMemcpyDeviceToHost));
        size_t bad = ny;
        double max_err = 0;
        for (size_t i = 0; i < ny; ++i) {
            double err = std::fabs(double(hY[i]) - ref[i]);
            if (!(err <= 1e-3 + 1e-3 * std::fabs(ref[i]))) { bad = i; break; }   // NaN 也会进这里
            max_err = std::max(max_err, err);
        }
        if (bad < ny) {
            size_t i = bad;
            int q = int(i % Q); i /= Q;
            int p = int(i % P); i /= P;
            int k = int(i % sh.K); int n = int(i / sh.K);
            std::printf("  %-7s FAIL  first wrong (n,k,p,q)=(%d,%d,%d,%d): got %g, want %g\n",
                        impl.name, n, k, p, q, hY[bad], ref[bad]);
            continue;
        }
        if (!ns.timing) { std::printf("  %-7s PASS  max err %.1e\n", impl.name, max_err); continue; }
        for (int i = 0; i < 3; ++i) impl.fn(dX, dW, dY, sh, s);
        std::vector<float> ms(20);
        for (float& t : ms) {
            CHECK_CUDA(cudaEventRecord(e0, s));
            impl.fn(dX, dW, dY, sh, s);
            CHECK_CUDA(cudaEventRecord(e1, s));
            CHECK_CUDA(cudaEventSynchronize(e1));
            CHECK_CUDA(cudaEventElapsedTime(&t, e0, e1));
        }
        std::sort(ms.begin(), ms.end());
        const double med = (ms[9] + ms[10]) / 2;
        std::printf("  %-7s PASS  max err %.1e  %9.3f ms  %6.3f TFLOPS\n", impl.name, max_err, med, flops / med / 1e9);
    }
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaStreamDestroy(s));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dW));
    CHECK_CUDA(cudaFree(dY));
}

int main(int argc, char** argv) {
    cudaDeviceProp p{};
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    std::printf("GPU: %s, %d SMs\n", p.name, p.multiProcessorCount);
    std::printf("(on Orin lock clocks with jetson_clocks before timing; FP32 peak ≈ 5.3 TFLOPS)\n");
#ifndef USE_CUDNN
    std::printf("(built without -DUSE_CUDNN: no cuDNN reference)\n");
#endif
    bool any = false;
    for (const NamedShape& ns : kShapes) {
        if (argc == 2 && std::strcmp(argv[1], ns.name) != 0) continue;
        run_shape(ns);
        any = true;
    }
    if (!any) {
        std::printf("unknown shape '%s'; choose one of:", argv[1]);
        for (const NamedShape& ns : kShapes) std::printf(" %s", ns.name);
        std::printf("\n");
        return 1;
    }
}
