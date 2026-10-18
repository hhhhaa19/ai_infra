// Fixed width: exactly 256 threads per block, one block per row.
template<bool Warp>
__global__ void rmsNormCompare(const float* x, const float* w, float* y, float eps) {
    const int tid = threadIdx.x;
    const int index = blockIdx.x * 256 + tid;
    const int lane = tid % 32;
    const int warpId = tid / 32;
    const float value = x[index];
    float sum = value * value;
    __shared__ float partial[Warp ? 8 : 256];
    __shared__ float scale;
    if constexpr (Warp) {
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, sum, offset);
            if (lane < offset) sum += other;
        }
        if (lane == 0) partial[warpId] = sum;
        __syncthreads();
        for (int stride = 4; stride > 0; stride /= 2) {
            if (lane == 0 && warpId < stride)
                partial[warpId] += partial[warpId + stride];
            __syncthreads();
        }
    } else {
        partial[tid] = sum;
        __syncthreads();
        for (int stride = 128; stride > 0; stride /= 2) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
    }
    if (tid == 0) scale = rsqrtf(partial[0] / 256.0f + eps);
    __syncthreads();
    y[index] = w[tid] * value * scale;
}
