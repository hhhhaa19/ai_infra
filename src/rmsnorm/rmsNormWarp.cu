#include <cuda_runtime.h>
#include <cstddef>

// 单block 不固定线程 不固定n 计算，x 为1*n 的连续空间 我们先处理不溢出
__global__ void rmsNorm(
    const float* x,
    const float* w,
    float* y,
    int n,
    float eps
){
    extern __shared__ float p[];
    __shared__ float scale;
    int xid = threadIdx.x;
    int warpid = xid/32;
    int lane = xid%32;
    int warpCount = blockDim.x/32;
    float sum = 0.0f;
   
    for (int col = xid ; col < n; col += blockDim.x){
        float curValue = x[col] ;
        curValue *= curValue;
        sum += curValue;
    }
    for ( int offset = 16; offset > 0; offset/=2){
        float other = __shfl_down_sync(
            0xffffffff,
            sum,
            offset
        );
        if (lane < offset){
            sum += other;
        }
    }
    if (lane == 0 ){
        p[warpid] = sum;
    }
    __syncthreads();

    for (int stride =warpCount/2 ;stride > 0; stride /= 2){
        if (lane == 0 && warpid < stride){
            p[warpid] += p[warpid+stride];
        }
        __syncthreads();
    }
    if (xid == 0){
        scale = rsqrtf(p[0] / n + eps);
    }
    __syncthreads();
    for (int col = xid ; col < n; col += blockDim.x){
        y[col] = x[col] * scale* w[col];
    }

}

__global__ void rmsNormN(
    const float* x,
    const float* w,
    float* y,
    int n,
    float eps
){
    extern __shared__ float p[];
    __shared__ float scale;
    int xid = threadIdx.x;
    int warpid = xid/32;
    int lane = xid%32;
    int warpCount = blockDim.x/32;
    float sum = 0.0f;
    const std::size_t rowOffset =
    static_cast<std::size_t>(blockIdx.x) * n;
    for (int col = xid ; col < n; col += blockDim.x){
        float curValue = x[rowOffset + col];
        curValue *= curValue;
        sum += curValue;
    }
    for ( int offset = 16; offset > 0; offset/=2){
        float other = __shfl_down_sync(
            0xffffffff,
            sum,
            offset
        );
        if (lane < offset){
            sum += other;
        }
    }
    if (lane == 0 ){
        p[warpid] = sum;
    }
    __syncthreads();

    for (int stride =warpCount/2 ;stride > 0; stride /= 2){
        if (lane == 0 && warpid < stride){
            p[warpid] += p[warpid+stride];
        }
        __syncthreads();
    }
    if (xid == 0){
        scale = rsqrtf(p[0] / n + eps);
    }
    __syncthreads();
    for (int col = xid ; col < n; col += blockDim.x){
        y[col+rowOffset] = x[col+rowOffset] * scale* w[col];
    }

}

// 对照实验：与 rmsNormN 完全相同，只把 blockDim.x 换成编译期常量 BLOCK。
// 用来验证 CUB 版的优势是否来自“线程数编译期已知 → 循环展开”。
// 启动时 BLOCK 必须等于实际每 block 线程数。
template<int BLOCK>
__global__ void rmsNormNT(
    const float* x,
    const float* w,
    float* y,
    int n,
    float eps
){
    extern __shared__ float p[];
    __shared__ float scale;
    int xid = threadIdx.x;
    int warpid = xid/32;
    int lane = xid%32;
    constexpr int warpCount = BLOCK/32;
    float sum = 0.0f;
    const std::size_t rowOffset =
    static_cast<std::size_t>(blockIdx.x) * n;
    for (int col = xid ; col < n; col += BLOCK){
        float curValue = x[rowOffset + col];
        curValue *= curValue;
        sum += curValue;
    }
    for ( int offset = 16; offset > 0; offset/=2){
        float other = __shfl_down_sync(
            0xffffffff,
            sum,
            offset
        );
        if (lane < offset){
            sum += other;
        }
    }
    if (lane == 0 ){
        p[warpid] = sum;
    }
    __syncthreads();

    for (int stride =warpCount/2 ;stride > 0; stride /= 2){
        if (lane == 0 && warpid < stride){
            p[warpid] += p[warpid+stride];
        }
        __syncthreads();
    }
    if (xid == 0){
        scale = rsqrtf(p[0] / n + eps);
    }
    __syncthreads();
    for (int col = xid ; col < n; col += BLOCK){
        y[col+rowOffset] = x[col+rowOffset] * scale* w[col];
    }

}
