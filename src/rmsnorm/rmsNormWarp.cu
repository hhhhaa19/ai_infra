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
    float eps,
    int stride = 0   // 行间距（元素个数）；0 表示紧密存放，等于 n
){
    extern __shared__ float p[];
    __shared__ float scale;
    int xid = threadIdx.x;
    int warpid = xid/32;
    int lane = xid%32;
    int warpCount = blockDim.x/32;
    float sum = 0.0f;
    const std::size_t rowOffset =
    static_cast<std::size_t>(blockIdx.x) * (stride > 0 ? stride : n);
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
    float eps,
    int stride = 0   // 行间距（元素个数）；0 表示紧密存放，等于 n
){
    extern __shared__ float p[];
    __shared__ float scale;
    int xid = threadIdx.x;
    int warpid = xid/32;
    int lane = xid%32;
    constexpr int warpCount = BLOCK/32;
    float sum = 0.0f;
    const std::size_t rowOffset =
    static_cast<std::size_t>(blockIdx.x) * (stride > 0 ? stride : n);
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

// 对照实验：一行 x 只读一遍，存在寄存器里给第二遍用。
// 每个线程固定处理 PER_THREAD 个元素（编译期常量，循环完全展开，数组才能留在寄存器里），
// 越界元素用 col < n 屏蔽，所以 n 不整除时所有线程仍走同一条指令序列。
// 要求 n <= BLOCK * PER_THREAD；启动时 BLOCK 必须等于实际每 block 线程数。
template<int BLOCK, int PER_THREAD>
__global__ void rmsNormReg(
    const float* x,
    const float* w,
    float* y,
    int n,
    float eps,
    int stride = 0   // 行间距（元素个数）；0 表示紧密存放，等于 n
){
    extern __shared__ float p[];
    __shared__ float scale;
    int xid = threadIdx.x;
    int warpid = xid/32;
    int lane = xid%32;
    constexpr int warpCount = BLOCK/32;
    const std::size_t rowOffset =
    static_cast<std::size_t>(blockIdx.x) * (stride > 0 ? stride : n);
    float v[PER_THREAD];
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < PER_THREAD; ++i){
        int col = xid + i * BLOCK;
        v[i] = col < n ? x[rowOffset + col] : 0.0f;
        sum += v[i] * v[i];
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
    #pragma unroll
    for (int i = 0; i < PER_THREAD; ++i){
        int col = xid + i * BLOCK;
        if (col < n){
            y[col+rowOffset] = v[i] * scale* w[col];
        }
    }

}
// 一个 warp 处理一行：没有 shared memory，也没有 __syncthreads()。
// 每个线程固定处理 PER_LANE 个元素（编译期常量，完全展开，数组留在寄存器里）。
// 要求 n <= 32 * PER_LANE；一个 block 放 WARPS 个 warp，同时处理 WARPS 行。
template <int PER_LANE, int WARPS>
__global__ void rmsNormWarpRow(const float* x, const float* w, float* y,
                            int rows, int n, float eps, int stride = 0) {
    const int  lane = threadIdx.x %32;
    const int warp = threadIdx.x/32;
    const int row = blockIdx.x *WARPS+warp; //我们在处理第几行
    if (row >= rows) return;
    float v[PER_LANE]; //寄存器每个线程独立 warp可以直接计算
    // float p[WARPS];
    float sum = 0.0f;
    const std::size_t rowOffset = static_cast<std::size_t>(row)* (stride >0 ? stride : n);
    #pragma unroll
    for (int i =0 ;i< PER_LANE ;i++){
        int index = i*32+lane;
        float cur = index < n ? x[index+rowOffset] : 0;
        v[i] = cur;
        sum += cur *cur;
    }
    for (int offset=16;offset >0 ;offset /=2 ){
        // xor：两两交换后相加，结束后 32 个线程都是这一行的总和
        sum += __shfl_xor_sync(
            0xffffffff,
            sum,
            offset
        );
    }

    const float scale = rsqrtf(sum / n + eps); // 每个线程都有总和，直接算
    #pragma unroll
    for (int i = 0; i < PER_LANE; ++i){
        int index = i*32+lane;
        if (index < n){
            y[index+rowOffset] = v[i] * scale * w[index];
        }
    }

}

// 融合版：h = x + r；y = rmsnorm(h) * w。h 只写一次，归一化直接用寄存器里的值。
// 要求和 rmsNormReg 一样：n <= BLOCK * PER_THREAD；BLOCK 等于实际每 block 线程数。
template<int BLOCK, int PER_THREAD>
__global__ void fusedAddRmsNormReg(
    const float* x,
    const float* r,      // residual
    const float* w,
    float* h,            // 输出：x + r
    float* y,            // 输出：rmsnorm(h) * w
    int n,
    float eps
){
    extern __shared__ float p[];
    __shared__ float scale;
    int xid = threadIdx.x;
    int warpid = xid/32;
    int lane = xid%32;
    constexpr int warpCount = BLOCK/32;
    const std::size_t rowOffset = static_cast<std::size_t>(blockIdx.x) * n;
    float v[PER_THREAD];
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < PER_THREAD; ++i){
        int col = xid + i * BLOCK;
        // TODO 1: col < n 时 v[i] = x + r（都要加 rowOffset），否则为 0
        v[i] = col < n ? x[rowOffset + col] +r[rowOffset + col]: 0.0f;
        // TODO 2: col < n 时把 v[i] 写到 h
        if (col < n) h[rowOffset + col] = v[i];
        sum += v[i] * v[i];
    }

    // TODO 3: 从这里往下和 rmsNormReg 完全一样：warp 内规约 → shared 规约 → 算 scale → 写 y
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
    #pragma unroll
    for (int i = 0; i < PER_THREAD; ++i){
        int col = xid + i * BLOCK;
        if (col < n){
            y[col+rowOffset] = v[i] * scale* w[col];
        }
    }
}

// ===== float4 版：一条指令读 / 写 4 个 float（16 字节） =====
// 一行看成 n/4 个 float4，线程 t 处理第 t、t+BLOCK、…… 个 float4：一个 warp 一次读连续的 32 个 float4（512 字节）。
// 每个线程处理 VPT 个 float4，存在寄存器里给第二遍用。要求 n % 4 == 0、所有指针 16 字节对齐（调用方检查）。
template<int BLOCK, int VPT>
__global__ void rmsNormVec4(
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
    const int n4 = n / 4;                                   // 这一行有几个 float4
    const std::size_t row4 = static_cast<std::size_t>(blockIdx.x) * n4;
    const float4* x4 = reinterpret_cast<const float4*>(x) + row4;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* y4 = reinterpret_cast<float4*>(y) + row4;
    float4 v[VPT];
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; ++i){
        int c = xid + i * BLOCK;                            // 第几个 float4
        v[i] = c < n4 ? x4[c] : make_float4(0.f, 0.f, 0.f, 0.f);
        sum += v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z + v[i].w * v[i].w;
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
    #pragma unroll
    for (int i = 0; i < VPT; ++i){
        int c = xid + i * BLOCK;
        if (c < n4){
            float4 ww = w4[c];
            y4[c] = make_float4(v[i].x * scale * ww.x, v[i].y * scale * ww.y,
                                v[i].z * scale * ww.z, v[i].w * scale * ww.w);
        }
    }
}

// 融合 + float4：h = x + r；y = rmsnorm(h) * w。要求同 rmsNormVec4。
template<int BLOCK, int VPT>
__global__ void fusedAddRmsNormVec4(
    const float* x,
    const float* r,      // residual
    const float* w,
    float* h,            // 输出：x + r
    float* y,            // 输出：rmsnorm(h) * w
    int n,
    float eps
){
    extern __shared__ float p[];
    __shared__ float scale;
    int xid = threadIdx.x;
    int warpid = xid/32;
    int lane = xid%32;
    constexpr int warpCount = BLOCK/32;
    const int n4 = n / 4;
    const std::size_t row4 = static_cast<std::size_t>(blockIdx.x) * n4;
    const float4* x4 = reinterpret_cast<const float4*>(x) + row4;
    const float4* r4 = reinterpret_cast<const float4*>(r) + row4;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* h4 = reinterpret_cast<float4*>(h) + row4;
    float4* y4 = reinterpret_cast<float4*>(y) + row4;
    float4 v[VPT];
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; ++i){
        int c = xid + i * BLOCK;
        if (c < n4){
            float4 a = x4[c], b = r4[c];                    // 各 1 条 128 位读取
            v[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            h4[c] = v[i];                                   // 1 条 128 位写入
        } else {
            v[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        sum += v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z + v[i].w * v[i].w;
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
    #pragma unroll
    for (int i = 0; i < VPT; ++i){
        int c = xid + i * BLOCK;
        if (c < n4){
            float4 ww = w4[c];
            y4[c] = make_float4(v[i].x * scale * ww.x, v[i].y * scale * ww.y,
                                v[i].z * scale * ww.z, v[i].w * scale * ww.w);
        }
    }
}
