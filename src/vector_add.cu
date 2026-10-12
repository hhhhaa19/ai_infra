#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::cerr << __FILE__ << ':' << __LINE__ << ": " \
                  << cudaGetErrorString(status) << '\n'; \
        return 1; \
    } \
} while (0)

struct DeviceBuffer {
    void* ptr = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
};

struct CudaEvent {
    cudaEvent_t event = nullptr;
    CudaEvent() = default;
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() { if (event) cudaEventDestroy(event); }
};

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    int threadsPerBlock = 256;
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [threads_per_block]\n";
        return 1;
    }
    if (argc == 2) {
        try {
            std::size_t used = 0;
            threadsPerBlock = std::stoi(argv[1], &used);
            if (used != std::string(argv[1]).size() || threadsPerBlock <= 0)
                throw std::invalid_argument("invalid thread count");
        } catch (const std::exception&) {
            std::cerr << "threads_per_block must be a positive integer\n";
            return 1;
        }
    }

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    cudaFuncAttributes attributes{};
    CHECK_CUDA(cudaFuncGetAttributes(&attributes, vectorAdd));
    const int limit = std::min({properties.maxThreadsPerBlock,
                               properties.maxThreadsDim[0],
                               attributes.maxThreadsPerBlock});
    if (threadsPerBlock > limit) {
        std::cerr << "threads_per_block exceeds device/kernel limit: " << limit << '\n';
        return 1;
    }

    // Deliberately not divisible by 128, 256, or 512 to test the boundary guard.
    constexpr int n = 1000003;
    constexpr int warmupCount = 50;
    constexpr int repeatCount = 200;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        // Small integers have exactly representable FP32 sums.
        a[i] = static_cast<float>(i % 1024);
        b[i] = static_cast<float>(i % 17 - 8);
    }
    DeviceBuffer da, db, dc;
    CHECK_CUDA(cudaMalloc(&da.ptr, bytes));
    CHECK_CUDA(cudaMalloc(&db.ptr, bytes));
    CHECK_CUDA(cudaMalloc(&dc.ptr, bytes));
    CHECK_CUDA(cudaMemcpy(da.ptr, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db.ptr, b.data(), bytes, cudaMemcpyHostToDevice));
    // Poison the output so missing stores fail validation (0xffffffff is NaN).
    CHECK_CUDA(cudaMemset(dc.ptr, 0xff, bytes));

    CudaEvent start, stop;
    CHECK_CUDA(cudaEventCreate(&start.event));
    CHECK_CUDA(cudaEventCreate(&stop.event));
    for (int i = 0; i < warmupCount; ++i) {
        vectorAdd<<<blocks, threadsPerBlock>>>(
            static_cast<const float*>(da.ptr), static_cast<const float*>(db.ptr),
            static_cast<float*>(dc.ptr), n);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(dc.ptr, 0xff, bytes));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> timesMs;
    timesMs.reserve(repeatCount);
    for (int i = 0; i < repeatCount; ++i) {
        CHECK_CUDA(cudaEventRecord(start.event));
        vectorAdd<<<blocks, threadsPerBlock>>>(
            static_cast<const float*>(da.ptr), static_cast<const float*>(db.ptr),
            static_cast<float*>(dc.ptr), n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop.event));
        CHECK_CUDA(cudaEventSynchronize(stop.event));
        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start.event, stop.event));
        timesMs.push_back(elapsedMs);
    }

    // D2H and full CPU reference validation are outside the timed region.
    CHECK_CUDA(cudaMemcpy(c.data(), dc.ptr, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        if (!std::isfinite(c[i]) || c[i] != expected) {
            std::cerr << "FAIL at " << i << ": " << c[i]
                      << " expected " << expected << '\n';
            return 1;
        }
    }

    const double mean = std::accumulate(timesMs.begin(), timesMs.end(), 0.0) / repeatCount;
    std::sort(timesMs.begin(), timesMs.end());
    const double median = (static_cast<double>(timesMs[99]) + timesMs[100]) / 2;
    const std::size_t p95Index = (95 * timesMs.size() + 99) / 100 - 1;
    std::cout << "GPU: " << properties.name << '\n'
              << "PASS: all " << n << " elements match CPU reference\n"
              << "Blocks: " << blocks << ", threads per block: " << threadsPerBlock << '\n'
              << "Warmup: " << warmupCount << ", measured runs: " << repeatCount << '\n'
              << "Timing: CUDA events around kernel; excludes allocation and copies\n"
              << "Mean: " << mean * 1000 << " us\n"
              << "Median: " << median * 1000 << " us\n"
              << "P95: " << timesMs[p95Index] * 1000 << " us\n";
}
