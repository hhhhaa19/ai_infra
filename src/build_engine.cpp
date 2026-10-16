#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

int main(int argc, char** argv) {
    if (argc > 3) {
        std::cerr << "Usage: build_engine [model.onnx] [model.engine]\n";
        return EXIT_FAILURE;
    }
    const char* onnxPath = argc > 1 ? argv[1] : "data/resnet50.onnx";
    const char* enginePath = argc > 2 ? argv[2] : "data/resnet50.engine";

    // Logger must outlive all TensorRT objects using it.
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder{
        nvinfer1::createInferBuilder(logger)
    };
    if (!builder) {
        std::cerr << "Failed to create builder\n";
        return EXIT_FAILURE;
    }

    // TensorRT 10 uses explicit batch dimensions by default.
    std::unique_ptr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(0U)
    };
    if (!network) {
        std::cerr << "Failed to create network\n";
        return EXIT_FAILURE;
    }

    std::unique_ptr<nvonnxparser::IParser> parser{
        nvonnxparser::createParser(*network, logger)
    };
    if (!parser || !parser->parseFromFile(
            onnxPath,
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX: " << onnxPath << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "ONNX parsed successfully!\n"
              << "Inputs: " << network->getNbInputs() << '\n'
              << "Outputs: " << network->getNbOutputs() << '\n';

    // This teaching tool supports static shapes. Dynamic inputs require profiles.
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto* input = network->getInput(i);
        const auto shape = input->getDimensions();
        if (input->isShapeTensor()) {
            std::cerr << "Shape-tensor inputs need optimization profiles\n";
            return EXIT_FAILURE;
        }
        for (int d = 0; d < shape.nbDims; ++d) {
            if (shape.d[d] < 0) {
                std::cerr << "Dynamic input needs an optimization profile: "
                          << input->getName() << '\n';
                return EXIT_FAILURE;
            }
        }
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config{
        builder->createBuilderConfig()
    };
    if (!config) {
        std::cerr << "Failed to create config\n";
        return EXIT_FAILURE;
    }
    constexpr std::size_t workspaceBytes = 1ULL << 30; // 1 GiB limit, not preallocation.
    config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE, workspaceBytes
    );
    // No FP16/INT8 flags. TensorRT's default TF32 permission is unchanged.
    std::cout << "Building engine...\n";
    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine{
        builder->buildSerializedNetwork(*network, *config)
    };
    if (!serializedEngine) {
        std::cerr << "Failed to build engine\n";
        return EXIT_FAILURE;
    }
    if (serializedEngine->size() >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        std::cerr << "Engine exceeds file-stream write size limit\n";
        return EXIT_FAILURE;
    }

    // Opening with trunc replaces an existing file at enginePath.
    std::ofstream file(enginePath, std::ios::binary | std::ios::trunc);
    if (!file) {
        std::cerr << "Cannot open output: " << enginePath << '\n';
        return EXIT_FAILURE;
    }
    file.write(
        static_cast<const char*>(serializedEngine->data()),
        static_cast<std::streamsize>(serializedEngine->size())
    );
    file.close();
    if (!file) {
        std::cerr << "Failed to write engine\n";
        return EXIT_FAILURE;
    }
    std::cout << "Engine saved to: " << enginePath << '\n'
              << "Engine bytes: " << serializedEngine->size() << '\n';
    return EXIT_SUCCESS;
}
