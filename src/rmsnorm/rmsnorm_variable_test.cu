#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << __LINE__ << ": " << cudaGetErrorString(err) << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// One block, n > 0; complete warps and a power-of-two warp count.
__global__ void rmsNorm(const float* x, const float* w, float* y,
                        int n, float eps) {
    extern __shared__ float p[];
    __shared__ float scale;
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int lane = tid % 32;
    const int warpCount = blockDim.x / 32;

    // Each thread accumulates several input elements in its own register.
    float sum = 0.0f;
    for (int col = tid; col < n; col += blockDim.x) {
        float value = x[col];
        sum += value * value;
    }

    // Reduce within each warp, then publish one sum per warp.
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, sum, offset);
        if (lane < offset) sum += other;
    }
    if (lane == 0) p[warpId] = sum;
    __syncthreads();

    // Reduce the warp sums across this block.
    for (int stride = warpCount / 2; stride > 0; stride /= 2) {
        if (lane == 0 && warpId < stride)
            p[warpId] += p[warpId + stride];
        __syncthreads();
    }
    if (tid == 0) scale = rsqrtf(p[0] / static_cast<float>(n) + eps);
    __syncthreads();

    for (int col = tid; col < n; col += blockDim.x)
        y[col] = x[col] * scale * w[col];
}

bool testCase(int n, int threads, bool zeros) {
    constexpr float eps = 1e-5f;
    constexpr double atol = 2e-6, rtol = 2e-5;
    std::vector<float> x(n), w(n), y(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        x[i] = zeros ? 0.0f : static_cast<float>((i % 97) - 48) / 13.0f;
        w[i] = static_cast<float>((i % 19) - 9) / 7.0f;
        sum += static_cast<double>(x[i]) * x[i];
    }
    const double scale = 1.0 / std::sqrt(sum / n + eps);
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    float *dx = nullptr, *dw = nullptr, *dy = nullptr;
    CHECK_CUDA(cudaMalloc(&dx, bytes));
    CHECK_CUDA(cudaMalloc(&dw, bytes));
    CHECK_CUDA(cudaMalloc(&dy, bytes));
    CHECK_CUDA(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw, w.data(), bytes, cudaMemcpyHostToDevice));
    // NaN poison helps detect output elements that were never written.
    CHECK_CUDA(cudaMemset(dy, 0xff, bytes));
    rmsNorm<<<1, threads, (threads / 32) * sizeof(float)>>>(dx, dw, dy, n, eps);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y.data(), dy, bytes, cudaMemcpyDeviceToHost));

    bool pass = true;
    double maxAbs = 0.0;
    for (int i = 0; i < n; ++i) {
        const double reference = static_cast<double>(x[i]) * w[i] * scale;
        const double error = std::abs(static_cast<double>(y[i]) - reference);
        if (!std::isfinite(y[i]) || error > atol + rtol * std::abs(reference)) {
            if (pass) std::cerr << "First mismatch: index=" << i
                << " CPU=" << reference << " GPU=" << y[i] << '\n';
            pass = false;
        }
        maxAbs = std::max(maxAbs, error);
    }
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dw));
    CHECK_CUDA(cudaFree(dy));
    std::cout << (pass ? "PASS" : "FAIL") << " n=" << n
              << " threads=" << threads << " input=" << (zeros ? "zeros" : "mixed")
              << " max_abs=" << maxAbs << '\n';
    return pass;
}

int main() {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\nTolerance: atol=2e-6 rtol=2e-5\n";
    bool allPass = true;
    int cases = 0;
    for (int n : {1, 17, 256, 1003, 4096}) {
        for (int threads : {32, 64, 128, 256, 512}) {
            if (threads > prop.maxThreadsPerBlock) return 1;
            for (bool zeros : {false, true}) {
                allPass = testCase(n, threads, zeros) && allPass;
                ++cases;
            }
        }
    }
    std::cout << (allPass ? "ALL PASS: " : "FAILED: ") << cases << " cases\n";
    return allPass ? 0 : 1;
}
