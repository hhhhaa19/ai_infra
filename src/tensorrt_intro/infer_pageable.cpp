#include <NvInfer.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <numeric>
#include <chrono>

#define CHECK_CUDA(call)\
    do {\
        cudaError_t err = call;\
        if (err != cudaSuccess){\
            std::cerr <<"CUDA Error at " << __FILE__ <<":" << __LINE__\
                      << "-" << cudaGetErrorString(err) << std::endl;\
            return 1;\
        }\
    }while (0)

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

// 离开作用域时自动释放 GPU 显存。
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

// 离开作用域时先等待 GPU 工作，再销毁流。
struct CudaStream {
    cudaStream_t stream = nullptr;

    CudaStream() = default;
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    ~CudaStream() {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
};

int main() {
    Logger logger;

    // 1. 打开 engine 文件，初始位置放在文件末尾。
    std::ifstream file(
        "data/resnet50.engine",
        std::ios::binary | std::ios::ate
    );

    if (!file) {
        std::cerr << "Failed to open engine file\n";
        return 1;
    }

    // 2. 获取文件字节数。
    const std::streamoff fileSize = file.tellg();

    if (fileSize <= 0) {
        std::cerr << "Engine file is empty or size query failed\n";
        return 1;
    }

 
   // 3. 分配 CPU 内存，然后从文件开头读取全部字节。
    std::vector<char> engineData(
        static_cast<std::size_t>(fileSize)
    );

    file.seekg(0, std::ios::beg);

    if (!file.read(
            engineData.data(),
            static_cast<std::streamsize>(engineData.size()))) {
        std::cerr << "Failed to read engine file\n";
        return 1;
    }

    std::cout << "Engine file read: "
              << engineData.size() << " bytes\n";

    // 4. 创建 Runtime，负责反序列化引擎。
    std::unique_ptr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(logger)
    };

    if (!runtime) {
        std::cerr << "Failed to create runtime\n";
        return 1;
    }

    // 5. 把文件中的字节恢复成 TensorRT 引擎对象。
    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(
            engineData.data(),
            engineData.size()
        )
    };

    if (!engine) {
        std::cerr << "Failed to deserialize engine\n";
        return 1;
    }

    std::cout << "Engine loaded successfully!\n";

    // 6. 创建执行上下文，为后续推理准备执行状态。
    std::unique_ptr<nvinfer1::IExecutionContext> context{
        engine->createExecutionContext()
    };

    if (!context) {
        std::cerr << "Failed to create execution context\n";
        return 1;
    }

    std::cout << "Execution context created successfully!\n";
    //

    const char* inputName = nullptr;
    const char* outputName = nullptr;
    int inputCount = 0;
    int outputCount = 0;

    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);

        if (engine->getTensorIOMode(name)
            == nvinfer1::TensorIOMode::kINPUT) {
            inputName = name;
            ++inputCount;
        } else {
            outputName = name;
            ++outputCount;
        }
    }

    if (inputCount != 1 || outputCount != 1) {
        std::cerr << "Expected one input and one output\n";
        return 1;
    }

    const auto inputShape = engine->getTensorShape(inputName);
    const auto outputShape = engine->getTensorShape(outputName);

    const bool shapesOK =
        inputShape.nbDims == 4 &&
        inputShape.d[0] == 1 &&
        inputShape.d[1] == 3 &&
        inputShape.d[2] == 224 &&
        inputShape.d[3] == 224 &&
        outputShape.nbDims == 2 &&
        outputShape.d[0] == 1 &&
        outputShape.d[1] == 1000;

    const bool typesOK =
        engine->getTensorDataType(inputName)
            == nvinfer1::DataType::kFLOAT &&
        engine->getTensorDataType(outputName)
            == nvinfer1::DataType::kFLOAT;

    const bool formatsOK =
        engine->getTensorFormat(inputName)
            == nvinfer1::TensorFormat::kLINEAR &&
        engine->getTensorFormat(outputName)
            == nvinfer1::TensorFormat::kLINEAR;

    const bool locationsOK =
        engine->getTensorLocation(inputName)
            == nvinfer1::TensorLocation::kDEVICE &&
        engine->getTensorLocation(outputName)
            == nvinfer1::TensorLocation::kDEVICE;

    if (!shapesOK || !typesOK || !formatsOK || !locationsOK) {
        std::cerr << "Engine I/O does not match this example\n";
        return 1;
    }
    //
    const std::size_t inputElements = 1ULL * 3 * 224 * 224;
    const std::size_t outputElements = 1ULL * 1000;

    const std::size_t inputBytes = inputElements * sizeof(float);
    const std::size_t outputBytes = outputElements * sizeof(float);

    // Ordinary pageable CPU buffers. Allocation is outside timing.
    std::vector<float> inputHostBuffer(inputElements);
    std::vector<float> outputHostBuffer(outputElements);
    float* hostInput = inputHostBuffer.data();
    float* hostOutput = outputHostBuffer.data();

    std::ifstream inputFile(
        "data/input.bin",
        std::ios::binary | std::ios::ate
    );

    if (!inputFile ||
        inputFile.tellg() != static_cast<std::streamoff>(inputBytes)) {
        std::cerr << "Input file missing or byte count mismatch\n";
        return 1;
    }

    inputFile.seekg(0, std::ios::beg);

    if (!inputFile.read(
            reinterpret_cast<char*>(hostInput),
            static_cast<std::streamsize>(inputBytes))) {
        std::cerr << "Failed to read input\n";
        return 1;
    }

    DeviceBuffer deviceInput;
    DeviceBuffer deviceOutput;

    CHECK_CUDA(cudaMalloc(&deviceInput.ptr, inputBytes));
    CHECK_CUDA(cudaMalloc(&deviceOutput.ptr, outputBytes));

    // 在缓冲区之后声明，退出时会先清理 stream。
    CudaStream stream;

    CHECK_CUDA(cudaStreamCreate(&stream.stream));

     
    if (!context->setTensorAddress(inputName, deviceInput.ptr) ||
        !context->setTensorAddress(outputName, deviceOutput.ptr)) {
        std::cerr << "Failed to bind tensor addresses\n";
        return 1;
    }

    constexpr int warmupCount = 50;
    constexpr int repeatCount = 200;

    // Warm up the complete request path, outside the timed region.
    for (int i = 0; i < warmupCount; ++i) {
        CHECK_CUDA(cudaMemcpy(
            deviceInput.ptr, hostInput,
            inputBytes, cudaMemcpyHostToDevice
        ));
        if (!context->enqueueV3(stream.stream)) {
            std::cerr << "Warmup failed\n";
            return 1;
        }
        CHECK_CUDA(cudaStreamSynchronize(stream.stream));
        CHECK_CUDA(cudaMemcpy(
            hostOutput, deviceOutput.ptr,
            outputBytes, cudaMemcpyDeviceToHost
        ));
    }

    std::vector<double> timesMs;
    timesMs.reserve(repeatCount);

    // CPU wall time: H2D + enqueue + wait + D2H.
    // Allocation, initialization, warmup, and file I/O are excluded.
    for (int i = 0; i < repeatCount; ++i) {
        const auto startTime = std::chrono::steady_clock::now();

        CHECK_CUDA(cudaMemcpy(
            deviceInput.ptr, hostInput,
            inputBytes, cudaMemcpyHostToDevice
        ));
        if (!context->enqueueV3(stream.stream)) {
            std::cerr << "Inference failed\n";
            return 1;
        }
        CHECK_CUDA(cudaStreamSynchronize(stream.stream));
        CHECK_CUDA(cudaMemcpy(
            hostOutput, deviceOutput.ptr,
            outputBytes, cudaMemcpyDeviceToHost
        ));

        const auto endTime = std::chrono::steady_clock::now();
        timesMs.push_back(
            std::chrono::duration<double, std::milli>(
                endTime - startTime
            ).count()
        );
    }

    // Summarize measured requests.
const double meanMs =
    std::accumulate(timesMs.begin(), timesMs.end(), 0.0)
    / timesMs.size();

std::sort(timesMs.begin(), timesMs.end());

const std::size_t n = timesMs.size();
const double medianMs =
    (static_cast<double>(timesMs[n / 2 - 1])
     + timesMs[n / 2]) / 2.0;

// nearest-rank P95；当前样本数固定为 200。
const std::size_t p95Index = (95 * n + 99) / 100 - 1;

std::cout << "Mode: pageable host memory + synchronous copies\n";
std::cout << "Timing: H2D + enqueue + stream synchronization + D2H\n";
std::cout << "Warmup runs: " << warmupCount << '\n';
std::cout << "Batch: 1\n";
std::cout << "Measured runs: " << n << '\n';
std::cout << "Mean:   " << meanMs << " ms\n";
std::cout << "Median: " << medianMs << " ms\n";
std::cout << "P95:    " << timesMs[p95Index] << " ms\n";
    std::ofstream outputFile(
        "data/output_trt_pageable.bin",
        std::ios::binary | std::ios::trunc
    );

    if (!outputFile) {
        std::cerr << "Failed to open output file\n";
        return 1;
    }

    outputFile.write(
        reinterpret_cast<const char*>(hostOutput),
        static_cast<std::streamsize>(outputBytes)
    );
    outputFile.close();

    if (!outputFile) {
        std::cerr << "Failed to write output\n";
        return 1;
    }

    std::cout << "Inference completed. First 5 logits:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << hostOutput[i] << ' ';
    }
    std::cout << "\nSaved: data/output_trt_pageable.bin\n";

    // stream、显存、context、engine、runtime 自动依次清理。
    return 0;

  
}