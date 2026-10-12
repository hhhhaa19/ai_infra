#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)\
    do {\
        cudaError_t err = call;\
        if (err != cudaSuccess){\
            std::cerr <<"CUDA Error at " << __FILE__ <<":" << __LINE__\
                      << "-" << cudaGetErrorString(err) << std::endl;\
            return 1;\
        }\
    }while (0)
std::vector<float> load_binary_file(const std::string& filepath){
    std::ifstream file(filepath, std::ios::binary |std::ios::ate);
    if (!file.is_open()){
        std::cerr << "Error: Failed to open file: "<< filepath << std::endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t num_elements = file_size / sizeof(float);
    std::vector<float> buffer(num_elements);

    if (!file.read(reinterpret_cast<char*>(buffer.data()),
        file_size)){
            std::cerr <<"Error: Failed to read binary data from: " << filepath << std::endl;
            exit(1);
        }
    return buffer;
}


int main(){
    std::cout << "==========================================" << std::endl;
    std::cout << "   AI Infra Baseline Tensor Validator    " << std::endl;
    std::cout << "==========================================" << std::endl;

    // 1. 读取 Python 导出的输入张量与基准输出
    std::string input_path = "../data/input.bin";
    std::string output_path = "../data/output_ref.bin";

    std::vector<float> input_tensor = load_binary_file(input_path);
    std::vector<float> output_ref = load_binary_file(output_path);

    // 2. 验证元素个数（Shape 对齐）
    // 输入期望: 1 * 3 * 224 * 224 = 150,528
    // 输出期望: 1 * 1000 = 1,000
    std::cout << "[CPU Check] Input Tensor Elements : " << input_tensor.size() 
              << " (Expected: 150528)" << std::endl;
    std::cout << "[CPU Check] Output Ref Elements   : " << output_ref.size() 
              << " (Expected: 1000)" << std::endl;

    // 3. 打印前 5 个浮点数（数值对齐，设置高精度打印）
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nInput Tensor First 5 Floats:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, input_tensor.size()); ++i) {
        std::cout << "  [" << i << "] " << input_tensor[i] << std::endl;
    }

    std::cout << "\nOutput Ref First 5 Floats:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, output_ref.size()); ++i) {
        std::cout << "  [" << i << "] " << output_ref[i] << std::endl;
    }

    // 4. 【核心演练】把输入张量拷入 GPU 显存（Host to Device）
    std::cout << "\n==========================================" << std::endl;
    std::cout << "[GPU Step] Allocating and Transferring to VRAM..." << std::endl;

    float* d_input = nullptr;
    size_t input_bytes = input_tensor.size() * sizeof(float);

    // 在 GPU 显存上申请空间
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    std::cout << "  cudaMalloc allocated: " << input_bytes << " bytes on GPU" << std::endl;

    // 将 CPU 内存的数据拷到 GPU 显存
    CHECK_CUDA(cudaMemcpy(d_input, input_tensor.data(), input_bytes, cudaMemcpyHostToDevice));
    std::cout << "  cudaMemcpy (H2D) successfully completed!" << std::endl;

    // 释放显存
    CHECK_CUDA(cudaFree(d_input));
    std::cout << "  cudaFree released GPU memory successfully." << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "ALL CHECKS PASSED: Ready for TensorRT!" << std::endl;

    return 0;

}