#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>


#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)       \
                      << " at line " << __LINE__ << '\n';                \
            return EXIT_FAILURE;                                        \
        }                                                               \
    } while (0)

struct DeviceBuffer {
    void* ptr = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

// 单个block 计算 用warp计算多线程，数组大小恰好是线程大小
__global__ void warpAdd(const float* a,  float* out, int n){
    const int lane = threadIdx.x %32 ; //warp内
    const int laneID = threadIdx.x /32 ; // 每个线程所在warp编号
    const int warpCount = blockDim.x / 32;

    extern __shared__ float partial[];
    const int fid = threadIdx.x + blockIdx.x * blockDim.x;
    float value = a[fid];
    //每个warp
    for (int offset = 16;offset >0; offset /=2){
        float other = __shfl_down_sync(
            0xffffffff,
            value,
            offset
        );
        if (lane < offset) {
            value += other;
        }
    }
    if (lane  == 0){
        partial[laneID] = value;
    }
    __syncthreads();
    //合并wrap
    for (int stride = warpCount/2; stride > 0;stride /= 2 ){
        if(lane == 0 && laneID < stride){
            partial[laneID ] += partial[laneID+stride]; 
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        out[0] = partial[0];
    }
}

int main() {
    constexpr int n = 256;
    constexpr int threads = 256;
    constexpr int warpCount = threads / 32;

    // 1. CPU 输入 1～256，并计算参考结果。
    std::vector<float> input(n);
    float reference = 0.0f;
    for (int i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i + 1);
        reference += input[i];
    }

    float result = 0.0f;
    const std::size_t inputBytes = input.size() * sizeof(float);

    // 2. GPU 输入、输出空间。
    DeviceBuffer deviceInput;
    DeviceBuffer deviceOutput;
    CHECK_CUDA(cudaMalloc(&deviceInput.ptr, inputBytes));
    CHECK_CUDA(cudaMalloc(&deviceOutput.ptr, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(
        deviceInput.ptr,
        input.data(),
        inputBytes,
        cudaMemcpyHostToDevice
    ));

    // 3. 单 block、256 线程；共享内存保存 8 个 warp 的部分和。
    warpAdd<<<1, threads, warpCount * sizeof(float)>>>(
        static_cast<const float*>(deviceInput.ptr),
        static_cast<float*>(deviceOutput.ptr),
        n
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. 取回结果并与 CPU 参考值比较。
    CHECK_CUDA(cudaMemcpy(
        &result,
        deviceOutput.ptr,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    std::cout << "CPU reference: " << reference << '\n';
    std::cout << "GPU result:    " << result << '\n';

    // 本例的整数及其中间和均可用 FP32 精确表示。
    if (result != reference) {
        std::cerr << "FAIL\n";
        return EXIT_FAILURE;
    }
    std::cout << "PASS: sum of 1..256 equals 32896\n";
    return EXIT_SUCCESS;
}
