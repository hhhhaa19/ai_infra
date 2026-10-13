#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)       \
                      << " at line " << __LINE__ << '\n';                \
            return EXIT_FAILURE;                                        \
        }                                                               \
    } while (0)

// 自动管理 GPU 显存。
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
__global__ void reduceSumBlock(const float* input, float* output,int n){
    extern __shared__ float partial[];
    const int nid = blockIdx.x * blockDim.x + threadIdx.x;
    partial[threadIdx.x] = (nid < n) ? input[nid] : 0.0f;
    __syncthreads();
       for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = partial[0];
    }
}


__global__ void reduceSumN(const float* input, float* output,int n) {
    extern __shared__ float partial[];
    const int tid = threadIdx.x;
    partial[tid] = (tid < n) ? input[tid] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = partial[0];
    }
}

// 教学版本：必须用 <<<1, 8>>> 启动，输入必须包含 8 个数。
__global__ void reduceSum8(const float* input, float* output) {
    // 同一个 block 的线程共享这一份数组。
    __shared__ float partial[8];

    // 当前线程在 block 内的编号：0～7。
    const int tid = threadIdx.x;

    // 每个线程把一个输入搬到共享内存。
    partial[tid] = input[tid];

    // 等所有线程写完，才能读取彼此的数据。
    __syncthreads();

    // 8 个数 → 4 个部分和 → 2 个部分和 → 1 个总和。
    for (int stride = 4; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }

        // 所有线程都要执行，不能放进上面的 if。
        __syncthreads();
    }

    // 只有线程 0 负责把最终结果写回全局内存。
    if (tid == 0) {
        output[0] = partial[0];
    }
}

int main() {
    // 1. 准备 CPU 输入和参考结果。
    constexpr int n = 1000;
    constexpr int threadsPerBlock = 256;
    constexpr int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    // 本例第一轮得到 4 个部分和，第二轮使用 4 个线程。
    static_assert(blocks == 4, "This example expects four partial sums");

    float hostInput[n];
    for (float& value : hostInput) {
        value = 1.0f;
    }
    float hostOutput = 0.0f;

    float reference = 0.0f;
    for (float value : hostInput) {
        reference += value;
    }

    // 2. 分配 GPU 输入和输出空间。
    DeviceBuffer deviceInput;
    DeviceBuffer deviceOutput1;
    DeviceBuffer deviceOutput2;

    CHECK_CUDA(cudaMalloc(&deviceInput.ptr, sizeof(hostInput)));
    CHECK_CUDA(cudaMalloc(&deviceOutput1.ptr, sizeof(float) * blocks));
    CHECK_CUDA(cudaMalloc(&deviceOutput2.ptr, sizeof(hostOutput)));

    // 3. 输入从 CPU 复制到 GPU。
    CHECK_CUDA(cudaMemcpy(
        deviceInput.ptr,
        hostInput,
        sizeof(hostInput),
        cudaMemcpyHostToDevice
    ));

    // 4. 第一轮：1000 个数 -> 4 个部分和。
    // 每个 block 使用 256 个线程和 256 个 float 的共享内存。
    reduceSumBlock<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        static_cast<const float*>(deviceInput.ptr),
        static_cast<float*>(deviceOutput1.ptr),
        n
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 第二轮：4 个部分和 -> 1 个总和。
    reduceSumN<<<1, blocks, blocks * sizeof(float)>>>(
        static_cast<const float*>(deviceOutput1.ptr),
        static_cast<float*>(deviceOutput2.ptr),
        blocks
    );
    CHECK_CUDA(cudaGetLastError());

    // 5. 把一个 float 的结果复制回 CPU。
    CHECK_CUDA(cudaMemcpy(
        &hostOutput,
        deviceOutput2.ptr,
        sizeof(hostOutput),
        cudaMemcpyDeviceToHost
    ));

    // 6. 验证。本例的小整数求和可以精确表示。
    std::cout << "CPU reference: " << reference << '\n';
    std::cout << "GPU result:    " << hostOutput << '\n';

    if (hostOutput != reference) {
        std::cerr << "FAIL\n";
        return EXIT_FAILURE;
    }

    std::cout << "PASS: sum of 1000 ones equals 1000\n";
    return EXIT_SUCCESS;
}