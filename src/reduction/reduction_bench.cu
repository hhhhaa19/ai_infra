#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Both algorithms use 256 threads, one input per thread, and the same
// multi-kernel reduction tree. Only the within-block reduction differs.
constexpr int kThreads = 256;
constexpr int kSamples = 50;
constexpr int kWarmup = 30;
constexpr int kBatch = 50;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    float* ptr = nullptr;
    explicit Buffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float)));
    }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreate(&value)); }
    ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
struct Event {
    cudaEvent_t value{};
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};

__global__ void sharedReduce(const float* input, float* output, int n) {
    __shared__ float partial[kThreads];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    partial[tid] = i < n ? input[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = partial[0];
}

// Mirrors your warp-first algorithm: warp leaders merge the shared sums.
__global__ void warpReduce(const float* input, float* output, int n) {
    __shared__ float partial[kThreads / 32];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warpId = tid / 32;
    const int i = blockIdx.x * blockDim.x + tid;
    float value = i < n ? input[i] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other = __shfl_down_sync(0xffffffff, value, offset);
        if (lane < offset) value += other;
    }
    if (lane == 0) partial[warpId] = value;
    __syncthreads();
    for (int stride = (kThreads / 32) / 2; stride > 0; stride /= 2) {
        if (lane == 0 && warpId < stride)
            partial[warpId] += partial[warpId + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = partial[0];
}

// Same stream orders all stages. Each call reduces the ORIGINAL input again.
float* launch(bool warp, const float* input, float* a, float* b,
              int n, cudaStream_t stream) {
    float* output = a;
    while (true) {
        const int blocks = (n + kThreads - 1) / kThreads;
        if (warp) warpReduce<<<blocks, kThreads, 0, stream>>>(input, output, n);
        else sharedReduce<<<blocks, kThreads, 0, stream>>>(input, output, n);
        if (blocks == 1) return output;
        n = blocks;
        input = output;
        output = output == a ? b : a;
    }
}

int main(int argc, char** argv) {
    try {
        const std::string prefix = argc > 1 ? argv[1] : "reduction_bench";
        std::ofstream summary(prefix + ".csv"), raw(prefix + "_samples.csv");
        if (!summary || !raw) throw std::runtime_error("Cannot open report files");
        summary << "n,threads,stages,round,mode,method,mean_us,median_us,p95_us,result,reference,status\n";
        raw << "n,round,mode,method,sample,us_per_reduction\n";
        summary << std::setprecision(9);
        raw << std::setprecision(9);
        int device = 0;
        check(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device));
        std::cout << "GPU: " << prop.name << "\nThreads: " << kThreads
                  << "\nWarmup per case: " << kWarmup << "\nSamples: " << kSamples
                  << "\nBatch repetitions per sample: " << kBatch << '\n';
        Stream stream;
        Event start, stop;
        for (int n : {250, 256, 1000003, 16000003}) {
            std::vector<float> host(n);
            double reference = 0;
            for (int i = 0; i < n; ++i) {
                host[i] = n <= 256 ? float(i + 1) : float(i % 17 - 8);
                reference += host[i];
            }
            const int blocks = (n + kThreads - 1) / kThreads;
            Buffer input(n), a(blocks), b(blocks);
            check(cudaMemcpy(input.ptr, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            int stages = 0;
            for (int size = n;;) {
                size = (size + kThreads - 1) / kThreads;
                ++stages;
                if (size == 1) break;
            }
            // Untimed 1-second conditioning, alternating algorithms.
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            do {
                for (int j = 0; j < 20; ++j)
                    launch(j % 2, input.ptr, a.ptr, b.ptr, n, stream.value);
                check(cudaGetLastError());
                check(cudaStreamSynchronize(stream.value));
            } while (std::chrono::steady_clock::now() < until);

            for (int round = 1; round <= 4; ++round) {
                for (int modeIndex = 0; modeIndex < 2; ++modeIndex) {
                    const int repetitions = modeIndex == 0 ? 1 : kBatch;
                    const char* mode = modeIndex == 0 ? "single" : "batch";
                    for (int order = 0; order < 2; ++order) {
                        const bool warp = (order + round - 1) % 2;
                        const char* method = warp ? "warp" : "shared";
                        // Poison scratch memory so a missing final write fails validation.
                        check(cudaMemsetAsync(a.ptr, 0xff, blocks * sizeof(float), stream.value));
                        check(cudaMemsetAsync(b.ptr, 0xff, blocks * sizeof(float), stream.value));
                        float* final = launch(warp, input.ptr, a.ptr, b.ptr, n, stream.value);
                        check(cudaGetLastError());
                        check(cudaStreamSynchronize(stream.value));
                        float result = 0;
                        check(cudaMemcpy(&result, final, sizeof(float), cudaMemcpyDeviceToHost));
                        if (!std::isfinite(result) || double(result) != reference)
                            throw std::runtime_error("Correctness check failed");
                        for (int j = 0; j < kWarmup; ++j)
                            launch(warp, input.ptr, a.ptr, b.ptr, n, stream.value);
                        check(cudaGetLastError());
                        check(cudaStreamSynchronize(stream.value));
                        std::vector<double> times;
                        times.reserve(kSamples);
                        for (int sample = 0; sample < kSamples; ++sample) {
                            check(cudaEventRecord(start.value, stream.value));
                            for (int j = 0; j < repetitions; ++j)
                                final = launch(warp, input.ptr, a.ptr, b.ptr, n, stream.value);
                            check(cudaEventRecord(stop.value, stream.value));
                            check(cudaGetLastError());
                            check(cudaEventSynchronize(stop.value));
                            float ms = 0;
                            check(cudaEventElapsedTime(&ms, start.value, stop.value));
                            const double us = ms * 1000.0 / repetitions;
                            times.push_back(us);
                            raw << n << ',' << round << ',' << mode << ',' << method << ','
                                << sample << ',' << us << '\n';
                        }
                        check(cudaMemcpy(&result, final, sizeof(float), cudaMemcpyDeviceToHost));
                        if (!std::isfinite(result) || double(result) != reference)
                            throw std::runtime_error("Post-timing correctness check failed");
                        double mean = 0;
                        for (double t : times) mean += t / times.size();
                        std::sort(times.begin(), times.end());
                        const double median = (times[24] + times[25]) / 2;
                        const double p95 = times[(95 * times.size() + 99) / 100 - 1];
                        summary << n << ',' << kThreads << ',' << stages << ',' << round << ','
                                << mode << ',' << method << ',' << mean << ',' << median << ','
                                << p95 << ',' << result << ',' << reference << ",PASS\n";
                        std::cout << "n=" << n << " round=" << round << ' ' << mode << ' '
                                  << method << " median=" << median << " us PASS\n";
                    }
                }
            }
        }
        std::cout << "ALL 64 CASES PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
