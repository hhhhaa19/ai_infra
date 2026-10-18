#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include "rmsnorm_variants.cuh"
#include <fstream>
#include <chrono>

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Buffer {
    float* data = nullptr;
    explicit Buffer(size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

int main(int argc, char** argv) {
    try {
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << '\n';
        constexpr int width = 256;
        constexpr float eps = 1e-5f;
        constexpr double atol = 2e-6, rtol = 2e-5;
        std::cout << "FP32; eps=" << eps << "; atol=" << atol << "; rtol=" << rtol << '\n';
        const char* names[] = {"ones_unit_weight", "zero_varied_weight",
                               "mixed_sign_varied_weight", "tiny_input", "large_input"};
        int cases = 0;
        for (int variant = 0; variant < 2; ++variant) {
        for (int rows : {1, 3, 17}) {
            for (int mode = 0; mode < 5; ++mode) {
                const size_t count = size_t(rows) * width;
                std::vector<float> x(count), weight(width), actual(count), reference(count);
                for (int col = 0; col < width; ++col)
                    weight[col] = mode == 0 ? 1.0f : float(col % 13 - 6) * 0.25f;
                for (int row = 0; row < rows; ++row) {
                    double sum = 0;
                    for (int col = 0; col < width; ++col) {
                        float value = float((col * 7 + row * 11) % 37 - 18) * 0.125f
                                      + float(row) * 0.03125f;
                        if (mode == 0) value = 1.0f;
                        if (mode == 1) value = 0.0f;
                        if (mode == 3) value *= 1e-7f;
                        if (mode == 4) value *= 1e3f;
                        x[size_t(row) * width + col] = value;
                        sum += double(value) * value;
                    }
                    const double scale = 1.0 / std::sqrt(sum / width + double(eps));
                    for (int col = 0; col < width; ++col)
                        reference[size_t(row) * width + col] = float(
                            double(x[size_t(row) * width + col]) * scale * weight[col]);
                }
                Buffer dx(count), dw(width), dy(count);
                check(cudaMemcpy(dx.data, x.data(), count * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(dw.data, weight.data(), width * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemset(dy.data, 0xff, count * sizeof(float)));
                if (variant) rmsNormCompare<true><<<rows, width>>>(dx.data, dw.data, dy.data, eps);
                else rmsNormCompare<false><<<rows, width>>>(dx.data, dw.data, dy.data, eps);
                check(cudaGetLastError());
                check(cudaDeviceSynchronize());
                check(cudaMemcpy(actual.data(), dy.data, count * sizeof(float), cudaMemcpyDeviceToHost));
                double maxAbs = 0, maxRatio = 0;
                size_t failures = 0;
                for (size_t i = 0; i < count; ++i) {
                    const double error = std::abs(double(actual[i]) - reference[i]);
                    const double limit = atol + rtol * std::abs(double(reference[i]));
                    if (!std::isfinite(actual[i]) || error > limit) ++failures;
                    maxAbs = std::max(maxAbs, error);
                    maxRatio = std::max(maxRatio, error / limit);
                }
                std::cout << "variant=" << (variant ? "warp" : "shared") << " rows=" << rows << " case=" << names[mode]
                          << " checked=" << count << " max_abs=" << std::setprecision(9)
                          << maxAbs << " max_tolerance_ratio=" << maxRatio
                          << " " << (failures ? "FAIL" : "PASS") << '\n';
                if (failures) return 1;
                ++cases;
            }
        }
        }
        std::cout << "ALL " << cases << " CASES PASS\n";

        const std::string prefix = argc > 1 ? argv[1] : "rmsnorm_compare";
        std::ofstream csv(prefix + ".csv"), raw(prefix + "_samples.csv");
        if (!csv || !raw) throw std::runtime_error("Cannot open reports");
        csv << "rows,round,mode,method,mean_us,median_us,p95_us,max_abs,status\n";
        raw << "rows,round,mode,method,sample,us\n";
        csv << std::setprecision(9); raw << std::setprecision(9);
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        for (int rows : {1, 128, 4096, 65536}) {
            size_t count = size_t(rows) * 256;
            std::vector<float> x(count), w(256), reference(count), actual(count);
            for (int j=0; j<256; ++j) w[j] = float(j%13-6)*0.25f;
            for (int row=0; row<rows; ++row) {
                double sum=0;
                for (int col=0; col<256; ++col) {
                    float v=float((col*7+row*11)%37-18)*0.125f;
                    x[size_t(row)*256+col]=v; sum+=double(v)*v;
                }
                double scale=1/std::sqrt(sum/256+double(eps));
                for(int col=0; col<256; ++col)
                    reference[size_t(row)*256+col]=float(double(x[size_t(row)*256+col])*w[col]*scale);
            }
            Buffer dx(count), dw(256), dy(count);
            check(cudaMemcpy(dx.data,x.data(),count*sizeof(float),cudaMemcpyHostToDevice));
            check(cudaMemcpy(dw.data,w.data(),256*sizeof(float),cudaMemcpyHostToDevice));
            auto launch=[&](bool warp){
                if(warp) rmsNormCompare<true><<<rows,256>>>(dx.data,dw.data,dy.data,eps);
                else rmsNormCompare<false><<<rows,256>>>(dx.data,dw.data,dy.data,eps);
            };
            auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
            do {
                for(int j=0;j<20;++j) launch(j%2);
                check(cudaGetLastError()); check(cudaDeviceSynchronize());
            } while(std::chrono::steady_clock::now()<until);
            for(int round=1;round<=4;++round) for(int mode=0;mode<2;++mode) for(int order=0;order<2;++order){
                bool warp=(order+round-1)%2;
                const char* method=warp?"warp":"shared";
                const char* modeName=mode?"batch":"single";
                int batch=mode?50:1;
                check(cudaMemset(dy.data,0xff,count*sizeof(float)));
                launch(warp);check(cudaGetLastError());check(cudaDeviceSynchronize());
                auto validate=[&](){
                    check(cudaMemcpy(actual.data(),dy.data,count*sizeof(float),cudaMemcpyDeviceToHost));
                    double maxAbs=0;
                    for(size_t i=0;i<count;++i){
                        double err=std::abs(double(actual[i])-reference[i]);
                        if(!std::isfinite(actual[i])||err>atol+rtol*std::abs(double(reference[i])))
                            throw std::runtime_error("Benchmark correctness failure");
                        maxAbs=std::max(maxAbs,err);
                    }
                    return maxAbs;
                };
                validate();
                for(int j=0;j<30;++j)launch(warp);
                check(cudaGetLastError());check(cudaDeviceSynchronize());
                std::vector<double> times; times.reserve(50);
                for(int sample=0;sample<50;++sample){
                    check(cudaEventRecord(start));
                    for(int j=0;j<batch;++j)launch(warp);
                    check(cudaEventRecord(stop));check(cudaGetLastError());check(cudaEventSynchronize(stop));
                    float ms;check(cudaEventElapsedTime(&ms,start,stop));
                    double us=ms*1000.0/batch;times.push_back(us);
                    raw<<rows<<','<<round<<','<<modeName<<','<<method<<','<<sample<<','<<us<<'\n';
                }
                double maxAbs=validate(),mean=0;for(double t:times)mean+=t/times.size();
                std::sort(times.begin(),times.end());double median=(times[24]+times[25])/2;
                csv<<rows<<','<<round<<','<<modeName<<','<<method<<','<<mean<<','<<median<<','<<times[47]<<','<<maxAbs<<",PASS\n";
                std::cout<<"BENCH rows="<<rows<<" round="<<round<<' '<<modeName<<' '<<method<<" median_us="<<median<<" PASS\n";
            }
        }
        check(cudaEventDestroy(start));check(cudaEventDestroy(stop));
        std::cout<<"ALL 64 BENCHMARK CASES PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
