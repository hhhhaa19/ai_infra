// 手写 attention 的测试台：对每组形状，跑 kernels.cuh 里已实现的版本，先对答案再计时。
// 用法：attention_bench          跑全部形状（前几组小形状只查正确性，后面的真实形状查正确性并计时）
//       attention_bench 名字      只跑这一组，比如 attention_bench llm1k
// 数据：Q、K、V 都是 [-1, 1] 的随机 float，固定种子。h % 4 == 3 的 Q 头整体乘 100：
//       分数会到几百，softmax 不减最大值的话 exp 溢出成 inf，结果变 NaN。
// 检查：O 先填 NaN，整块扫一遍查漏写；再抽行和 CPU double 参考比
//       （每个头抽约 64 行，含第 0 行和最后一行；N 小时全查）。|got − ref| ≤ 1e-4 + 1e-3·|ref| 算对，
//       不对时打印第一个错的位置 (b, h, i, c)。
// 计时：CUDA event 包住一次 launch（朴素版一次 launch 里有 3·B·Hq 个 kernel），预热 3 次，取 20 次中位数。
// 指标：TFLOPS = 4·B·Hq·N²·d / 时间（causal 按一半算）。参考：你的 gemm_reg 在 Orin 上约 2.9 TFLOPS，峰值约 5.3。
//       另外打印朴素版 S 的显存流量：每个头 S 写 1 次 + softmax 读写各 1 次 + P·V 读 1 次 = 4·N²·4 字节。
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

using AttnLaunch = bool (*)(const float*, const float*, const float*, float*, const AttnShape&, cudaStream_t);
struct Impl { const char* name; AttnLaunch fn; };
static const Impl kImpls[] = {
    {"naive", launch_attn_naive},
    {"flash", launch_attn_flash},
};

//                  name            B  Hq Hkv     N    d  causal
static const AttnShape kShapes[] = {
    {"tiny",            1,  1,  1,    4,   4, false},   // 和手算一样大
    {"odd",             2,  3,  3,   33,  64, false},   // N 不是 16 / 32 的倍数，B > 1
    {"gqa_odd",         1,  6,  2,   77,  32, false},   // GQA，每 3 个 Q 头共用一组 K/V
    {"causal_odd",      1,  4,  2,  100,  64, true},
    {"d128_odd",        1,  4,  4,  130, 128, false},
    {"smolvla_prefix",  1, 15,  5,  241,  64, false},   // SmolVLA VLM 部分：241 个 prefix token
    {"llm1k",           1, 16, 16, 1024,  64, false},
    {"llm2k_causal",    1, 16, 16, 2048,  64, true},
    {"llm4k_d128",      1,  8,  8, 4096, 128, false},
};
static const int kFirstTimed = 5;   // 从 smolvla_prefix 开始计时

static size_t q_off(const AttnShape& sh, int b, int h)  { return (size_t(b) * sh.Hq  + h)  * sh.N * sh.d; }
static size_t kv_off(const AttnShape& sh, int b, int kv) { return (size_t(b) * sh.Hkv + kv) * sh.N * sh.d; }

// 参考：O[b][h][i][:]，double 计算
static void ref_row(const AttnShape& sh, const std::vector<float>& Q, const std::vector<float>& K,
                    const std::vector<float>& V, int b, int h, int i, std::vector<double>& out) {
    const int N = sh.N, d = sh.d;
    const float* q = &Q[q_off(sh, b, h) + size_t(i) * d];
    const float* k = &K[kv_off(sh, b, h / sh.group())];
    const float* v = &V[kv_off(sh, b, h / sh.group())];
    const int jend = sh.causal ? i + 1 : N;
    std::vector<double> s(jend);
    double m = -INFINITY;
    for (int j = 0; j < jend; ++j) {
        double acc = 0;
        for (int c = 0; c < d; ++c) acc += double(q[c]) * k[size_t(j) * d + c];
        s[j] = acc / std::sqrt(double(d));
        m = std::max(m, s[j]);
    }
    double sum = 0;
    for (int j = 0; j < jend; ++j) { s[j] = std::exp(s[j] - m); sum += s[j]; }
    out.assign(d, 0.0);
    for (int j = 0; j < jend; ++j)
        for (int c = 0; c < d; ++c) out[c] += s[j] / sum * v[size_t(j) * d + c];
}

static void run_shape(const AttnShape& sh, bool timing) {
    const size_t nq = size_t(sh.B) * sh.Hq * sh.N * sh.d;
    const size_t nkv = size_t(sh.B) * sh.Hkv * sh.N * sh.d;
    std::vector<float> hQ(nq), hK(nkv), hV(nkv), hO(nq);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hQ) x = dist(rng);
    for (auto& x : hK) x = dist(rng);
    for (auto& x : hV) x = dist(rng);
    for (int b = 0; b < sh.B; ++b)
        for (int h = 3; h < sh.Hq; h += 4)
            for (size_t t = 0; t < size_t(sh.N) * sh.d; ++t) hQ[q_off(sh, b, h) + t] *= 100.0f;

    // 抽查的行：N ≤ 64 全查，否则约 64 行，含第 0 行和最后一行
    std::vector<int> rows;
    const int step = std::max(1, sh.N / 64);
    for (int i = 0; i < sh.N; i += step) rows.push_back(i);
    if (rows.back() != sh.N - 1) rows.push_back(sh.N - 1);
    std::vector<double> ref(size_t(sh.B) * sh.Hq * rows.size() * sh.d), tmp;
    size_t p = 0;
    for (int b = 0; b < sh.B; ++b)
        for (int h = 0; h < sh.Hq; ++h)
            for (int i : rows) {
                ref_row(sh, hQ, hK, hV, b, h, i, tmp);
                std::copy(tmp.begin(), tmp.end(), ref.begin() + p);
                p += sh.d;
            }

    float *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, nq * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK, nkv * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV, nkv * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dO, nq * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), nq * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), nkv * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), nkv * sizeof(float), cudaMemcpyHostToDevice));
    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    const double flop = 4.0 * sh.B * sh.Hq * double(sh.N) * sh.N * sh.d * (sh.causal ? 0.5 : 1.0);
    const double io_mb = 2.0 * (nq + nkv) * sizeof(float) / 1e6;                       // Q、O、K、V
    const double s_mb = 4.0 * sh.B * sh.Hq * double(sh.N) * sh.N * sizeof(float) / 1e6;  // 朴素版 S/P 流量
    std::printf("\n%s  B=%d Hq=%d Hkv=%d N=%d d=%d%s  (%.2f GFLOP; Q/K/V/O %.1f MB, 朴素版 S/P 流量 %.1f MB)\n",
                sh.name, sh.B, sh.Hq, sh.Hkv, sh.N, sh.d, sh.causal ? " causal" : "",
                flop / 1e9, io_mb, s_mb);

    for (const Impl& impl : kImpls) {
        CHECK_CUDA(cudaMemset(dO, 0xff, nq * sizeof(float)));   // 先填 NaN
        if (!impl.fn(dQ, dK, dV, dO, sh, s)) {
            std::printf("  %-6s not implemented / skipped\n", impl.name);
            continue;
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(s));
        CHECK_CUDA(cudaMemcpy(hO.data(), dO, nq * sizeof(float), cudaMemcpyDeviceToHost));

        bool pass = true;
        char where[128] = "";
        for (size_t t = 0; t < nq && pass; ++t)
            if (!std::isfinite(hO[t])) {
                pass = false;
                std::snprintf(where, sizeof(where), "non-finite at flat index %zu", t);
            }
        double worst = 0;
        p = 0;
        for (int b = 0; b < sh.B && pass; ++b)
            for (int h = 0; h < sh.Hq && pass; ++h)
                for (int i : rows) {
                    const float* got = &hO[q_off(sh, b, h) + size_t(i) * sh.d];
                    for (int c = 0; c < sh.d; ++c) {
                        double r = ref[p + c], e = std::fabs(double(got[c]) - r);
                        worst = std::max(worst, e);
                        if (pass && e > 1e-4 + 1e-3 * std::fabs(r)) {
                            pass = false;
                            std::snprintf(where, sizeof(where), "(b=%d,h=%d,i=%d,c=%d) got %.6f ref %.6f",
                                          b, h, i, c, got[c], r);
                        }
                    }
                    p += sh.d;
                    if (!pass) break;
                }
        if (!pass) {
            std::printf("  %-6s FAIL  %s\n", impl.name, where);
            continue;
        }
        if (!timing) {
            std::printf("  %-6s max_abs_err %.2e  PASS\n", impl.name, worst);
            continue;
        }
        for (int i = 0; i < 3; ++i) impl.fn(dQ, dK, dV, dO, sh, s);
        std::vector<float> ms(20);
        for (float& t : ms) {
            CHECK_CUDA(cudaEventRecord(e0, s));
            impl.fn(dQ, dK, dV, dO, sh, s);
            CHECK_CUDA(cudaEventRecord(e1, s));
            CHECK_CUDA(cudaEventSynchronize(e1));
            CHECK_CUDA(cudaEventElapsedTime(&t, e0, e1));
        }
        std::sort(ms.begin(), ms.end());
        const double med = (ms[9] + ms[10]) / 2;
        std::printf("  %-6s max_abs_err %.2e  PASS  %9.3f ms  %6.3f TFLOPS\n",
                    impl.name, worst, med, flop / med / 1e9);
    }
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaStreamDestroy(s));
    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
}

int main(int argc, char** argv) {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::printf("(on Orin lock clocks with jetson_clocks before timing)\n");
    const int count = int(sizeof(kShapes) / sizeof(kShapes[0]));
    bool found = false;
    for (int i = 0; i < count; ++i) {
        if (argc == 2 && std::strcmp(argv[1], kShapes[i].name) != 0) continue;
        found = true;
        run_shape(kShapes[i], i >= kFirstTimed);
    }
    if (!found) {
        std::printf("unknown shape %s; choices:", argv[1]);
        for (int i = 0; i < count; ++i) std::printf(" %s", kShapes[i].name);
        std::printf("\n");
        return 1;
    }
}
