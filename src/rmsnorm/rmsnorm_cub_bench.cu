// 手写 rmsNormN 与 CUB BlockReduce 版对照；own_t 是 rmsNormN 的模板版（线程数编译期常量）。
// 形状、数据、校验、计时方法与 rmsnorm_n_bench.cu 一致，只多了 method 维度。
#include "rmsNormWarp.cu"
#include "rmsnorm_cub.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#define CHECK(call) do { auto e = (call); if(e != cudaSuccess) \
    throw std::runtime_error(cudaGetErrorString(e)); } while(0)
struct Buffer {
    float* p=nullptr;
    explicit Buffer(size_t bytes) { CHECK(cudaMalloc(&p,bytes)); }
    ~Buffer(){ cudaFree(p); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};
enum Method { OWN, OWN_T, CUB };
const char* name(Method m){ return m==OWN?"own":m==OWN_T?"own_t":"cub"; }

struct Data {
    int rows,n;
    size_t bytes;
    std::vector<float> x,w,y;
    std::vector<double> ref;
    Buffer dx,dw,dy;
    Data(int r,int width,bool zero=false): rows(r),n(width),
        bytes(size_t(r)*width*sizeof(float)),x(size_t(r)*width),w(width),y(x.size()),ref(x.size()),
        dx(bytes),dw(size_t(width)*sizeof(float)),dy(bytes) {
        for(int c=0;c<n;++c) w[c]=float(c%19-9)/7;
        for(int row=0;row<rows;++row){
            double sum=0;
            for(int c=0;c<n;++c){
                size_t i=size_t(row)*n+c;
                x[i]=zero?0.0f:float((c*7+row*13)%97-48)/13;
                sum+=double(x[i])*x[i];
            }
            double scale=1/std::sqrt(sum/n+double(1e-5f));
            for(int c=0;c<n;++c){ size_t i=size_t(row)*n+c; ref[i]=double(x[i])*w[c]*scale; }
        }
        CHECK(cudaMemcpy(dx.p,x.data(),bytes,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dw.p,w.data(),size_t(n)*sizeof(float),cudaMemcpyHostToDevice));
    }
    void launch(Method m,int threads) {
        if(m==OWN){
            rmsNormN<<<rows,threads,(threads/32)*sizeof(float)>>>(dx.p,dw.p,dy.p,n,1e-5f);
            return;
        }
        const size_t shm=(threads/32)*sizeof(float);
        if(m==OWN_T){
            switch(threads){
                case 128: rmsNormNT<128><<<rows,128,shm>>>(dx.p,dw.p,dy.p,n,1e-5f); break;
                case 256: rmsNormNT<256><<<rows,256,shm>>>(dx.p,dw.p,dy.p,n,1e-5f); break;
                case 512: rmsNormNT<512><<<rows,512,shm>>>(dx.p,dw.p,dy.p,n,1e-5f); break;
                default: throw std::runtime_error("Unsupported own_t thread count");
            }
            return;
        }
        // CUB 的线程数是模板参数：运行时值映射到编译期实例。
        switch(threads){
            case 128: rmsNormCub<128><<<rows,128>>>(dx.p,dw.p,dy.p,n,1e-5f); break;
            case 256: rmsNormCub<256><<<rows,256>>>(dx.p,dw.p,dy.p,n,1e-5f); break;
            case 512: rmsNormCub<512><<<rows,512>>>(dx.p,dw.p,dy.p,n,1e-5f); break;
            default: throw std::runtime_error("Unsupported CUB thread count");
        }
    }
    double verify(Method m,int threads) {
        CHECK(cudaMemset(dy.p,0xff,bytes));
        launch(m,threads);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(y.data(),dy.p,bytes,cudaMemcpyDeviceToHost));
        double maxError=0;
        for(size_t i=0;i<y.size();++i){
            double err=std::abs(double(y[i])-ref[i]);
            if(!std::isfinite(y[i]) || err>2e-6+2e-5*std::abs(ref[i]))
                throw std::runtime_error(std::string(name(m))+" mismatch at element "+std::to_string(i));
            maxError=std::max(maxError,err);
        }
        return maxError;
    }
};
// "128,256" -> {128,256}
std::vector<int> parseList(const std::string& text) {
    std::vector<int> values;
    std::stringstream in(text);
    for(std::string item; std::getline(in,item,',');) values.push_back(std::stoi(item));
    if(values.empty()) throw std::runtime_error("Empty list: "+text);
    return values;
}
// 用法：rmsnorm_cub_bench [输出前缀] [rows 列表] [n 列表]，列表用逗号分隔。
// 不给 rows/n 时使用默认形状 rows=1,128,4096 × n=256,1003,4096。
int main(int argc,char** argv) try {
    std::string prefix=argc>1?argv[1]:"reports/rmsnorm_cub_vs_own";
    const std::vector<int> rowsList=argc>2?parseList(argv[2]):std::vector<int>{1,128,4096};
    const std::vector<int> nList=argc>3?parseList(argv[3]):std::vector<int>{256,1003,4096};
    std::ofstream summary(prefix+".csv"),raw(prefix+"_samples.csv");
    if(!summary || !raw) throw std::runtime_error("Cannot create CSV");
    summary<<"rows,n,method,threads,round,mean_us,median_us,p95_us,max_abs\n";
    raw<<"rows,n,method,threads,round,sample,us_per_launch\n";
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<"GPU: "<<prop.name<<"\nTiming: 50 launches per event pair, 50 samples; excludes copies/allocation\n";
    const int threadList[]={128,256,512};
    int checks=0;
    double worst=0;
    for(int rows:{1,3,17}) for(int n:{17,256,1003,4096}) for(bool zero:{false,true}) {
        Data d(rows,n,zero);
        for(Method m:{OWN,OWN_T,CUB}) for(int threads:threadList) { worst=std::max(worst,d.verify(m,threads)); ++checks; }
    }
    // 自定义 n 也单独校验（3 行，含全零输入）。
    if(argc>3) for(int n:nList) for(bool zero:{false,true}) {
        Data d(3,n,zero);
        for(Method m:{OWN,OWN_T,CUB}) for(int threads:threadList) { worst=std::max(worst,d.verify(m,threads)); ++checks; }
    }
    std::cout<<"ALL "<<checks<<" CORRECTNESS CASES PASS; max_abs="<<worst<<std::endl;

    // 每轮 9 个配置，三种实现交错，起点逐轮后移，避免固定顺序偏差。
    std::vector<std::pair<Method,int>> configs;
    for(int t:threadList){ configs.push_back({OWN,t}); configs.push_back({OWN_T,t}); configs.push_back({CUB,t}); }
    std::map<std::tuple<int,int,int,int>,std::vector<float>> roundMedians; // rows,n,method,threads
    cudaEvent_t start,stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    int cases=0;
    for(int rows:rowsList) for(int n:nList) {
        Data d(rows,n);
        // 每种形状至少预热 1 秒，三种实现、三种线程数交替。
        auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        do {
            for(auto [m,t]:configs) for(int i=0;i<20;++i) d.launch(m,t);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        } while(std::chrono::steady_clock::now()<until);
        for(int round=0;round<3;++round) for(size_t k=0;k<configs.size();++k) {
            auto [m,threads]=configs[(round+k)%configs.size()];
            double error=d.verify(m,threads);
            for(int i=0;i<50;++i) d.launch(m,threads);
            CHECK(cudaDeviceSynchronize());
            std::vector<float> samples;
            for(int sample=0;sample<50;++sample){
                CHECK(cudaEventRecord(start));
                for(int b=0;b<50;++b) d.launch(m,threads);
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                CHECK(cudaGetLastError());
                float ms=0; CHECK(cudaEventElapsedTime(&ms,start,stop));
                float us=ms*1000/50;
                samples.push_back(us);
                raw<<rows<<','<<n<<','<<name(m)<<','<<threads<<','<<round+1<<','<<sample<<','<<us<<'\n';
            }
            // 计时后再校验一次。
            error=std::max(error,d.verify(m,threads));
            double mean=0; for(float v:samples) mean+=v/samples.size();
            std::sort(samples.begin(),samples.end());
            float median=(samples[24]+samples[25])/2;
            roundMedians[{rows,n,int(m),threads}].push_back(median);
            summary<<rows<<','<<n<<','<<name(m)<<','<<threads<<','<<round+1<<','<<mean<<','<<median<<','<<samples[47]<<','<<error<<'\n';
            ++cases;
        }
        std::cout<<"Completed rows="<<rows<<" n="<<n<<std::endl;
    }
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    std::cout<<"ALL "<<cases<<" BENCHMARK CASES PASS\n\n";

    // 三轮中位数的中位数；比值以 own 为分母，<1 表示比 own 快。
    auto mid=[&](int r,int n,Method m,int t){
        auto v=roundMedians.at({r,n,int(m),t}); std::sort(v.begin(),v.end()); return v[1];
    };
    std::cout<<"rows,n,threads,own_us,own_t_us,cub_us,own_t_over_own,cub_over_own\n"<<std::fixed<<std::setprecision(3);
    for(int rows:rowsList) for(int n:nList) for(int t:threadList){
        float o=mid(rows,n,OWN,t),ot=mid(rows,n,OWN_T,t),c=mid(rows,n,CUB,t);
        std::cout<<rows<<','<<n<<','<<t<<','<<o<<','<<ot<<','<<c<<','<<ot/o<<','<<c/o<<'\n';
    }
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
