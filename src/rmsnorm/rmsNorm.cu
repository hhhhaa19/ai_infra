__global__ void rmsNorm256(
    const float* x,
    const float* w,
    float* y,
    float eps
){
    int xid = threadIdx.x;
    int index = blockIdx.x * 256 + xid;
    float curValue = x[index];
    __shared__ float p[256]; 
    __shared__ float scale;
    curValue *= curValue;
    p[xid] = curValue;
    __syncthreads();
    for (int stride = 256/2;stride > 0;stride /= 2){
        if (xid < stride){
            p[xid] +=p[xid+stride] ;
        }
        __syncthreads();
    }
    if (xid == 0){
        scale = rsqrtf(p[0] / 256.0f + eps);
    }
    __syncthreads();
    y[index] = w[xid] * x[index] * scale;
    
}
// 单个block 用wrap 计算
__global__ void reduceAdd(const float* x,float* y,int n){
    extern __shared__ float p []; 
    int tid = threadIdx.x;
    int warpCount = blockDim.x/32;
    int warpid = tid/32;
    int lane = tid%32;
    int bid = blockIdx.x;
    int blockCount = gridDim.x;
    int fid = blockDim.x * blockIdx.x+threadIdx.x;
    float curVaule = fid < n ? x[fid] : 0.0f;
    for (int offset = 16 ; offset > 0;offset/=2){
        float other = __shfl_down_sync(
            0xffffffff,
            curVaule,
            offset
        ); //从相隔offset的那个线程中取 curVaule 这个变量
        // 更新前offset个
        if (lane < offset){
            curVaule += other;
        }
    }
    if (lane == 0){
        p[warpid] = curVaule;
    }
    __syncthreads();
    for (int strip = warpCount/2 ; strip >0 ;strip /=2){
        if (lane ==0 && warpid < strip){
            p[warpid] += p[warpid+strip];
        }
        __syncthreads();
    }
    if (tid == 0){
        y[bid] = p[warpid];
    }
}