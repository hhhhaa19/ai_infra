#include "../kernels.cu"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
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
    void launch(int threads) {
        rmsNormN<<<rows,threads,(threads/32)*sizeof(float)>>>(dx.p,dw.p,dy.p,n,1e-5f);
    }
    double verify(int threads) {
        CHECK(cudaMemset(dy.p,0xff,bytes));
        launch(threads);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(y.data(),dy.p,bytes,cudaMemcpyDeviceToHost));
        double maxError=0;
        for(size_t i=0;i<y.size();++i){
            double err=std::abs(double(y[i])-ref[i]);
            if(!std::isfinite(y[i]) || err>2e-6+2e-5*std::abs(ref[i]))
                throw std::runtime_error("Mismatch at element "+std::to_string(i));
            maxError=std::max(maxError,err);
        }
        return maxError;
    }
};
int main(int argc,char** argv) try {
    std::string prefix=argc>1?argv[1]:"reports/rmsnorm_n";
    std::ofstream summary(prefix+".csv"),raw(prefix+"_samples.csv");
    if(!summary || !raw) throw std::runtime_error("Cannot create CSV");
    summary<<"rows,n,threads,round,mean_us,median_us,p95_us,max_abs\n";
    raw<<"rows,n,threads,round,sample,us_per_launch\n";
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<"GPU: "<<prop.name<<"\nTiming: 50 launches per event pair, 50 samples; excludes copies/allocation\n";
    int checks=0;
    double worst=0;
    for(int rows:{1,3,17}) for(int n:{17,256,1003,4096}) for(bool zero:{false,true}) {
        Data d(rows,n,zero);
        for(int threads:{128,256,512}) { worst=std::max(worst,d.verify(threads)); ++checks; }
    }
    std::cout<<"ALL "<<checks<<" CORRECTNESS CASES PASS; max_abs="<<worst<<std::endl;
    cudaEvent_t start,stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    int cases=0;
    for(int rows:{1,128,4096}) for(int n:{256,1003,4096}) {
        Data d(rows,n);
        // Warm up each shape for at least one second, alternating thread counts.
        auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        do {
            for(int t:{128,256,512}) for(int i=0;i<20;++i) d.launch(t);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        } while(std::chrono::steady_clock::now()<until);
        for(int round=0;round<3;++round) for(int k=0;k<3;++k) {
            const int threads=128<<((round+k)%3);
            double error=d.verify(threads);
            for(int i=0;i<50;++i) d.launch(threads);
            CHECK(cudaDeviceSynchronize());
            std::vector<float> samples;
            for(int sample=0;sample<50;++sample){
                CHECK(cudaEventRecord(start));
                for(int b=0;b<50;++b) d.launch(threads);
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                CHECK(cudaGetLastError());
                float ms=0; CHECK(cudaEventElapsedTime(&ms,start,stop));
                float us=ms*1000/50;
                samples.push_back(us);
                raw<<rows<<','<<n<<','<<threads<<','<<round+1<<','<<sample<<','<<us<<'\n';
            }
            // Verify once again after the timed work.
            error=std::max(error,d.verify(threads));
            double mean=0; for(float v:samples) mean+=v/samples.size();
            std::sort(samples.begin(),samples.end());
            float median=(samples[24]+samples[25])/2;
            summary<<rows<<','<<n<<','<<threads<<','<<round+1<<','<<mean<<','<<median<<','<<samples[47]<<','<<error<<'\n';
            ++cases;
        }
        std::cout<<"Completed rows="<<rows<<" n="<<n<<std::endl;
    }
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    std::cout<<"ALL "<<cases<<" BENCHMARK CASES PASS\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
