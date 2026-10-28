// TensorRT plugin：ONNX 里的 RmsNorm 节点（export_onnx.py 的 symbolic 产生）由这里执行。
// TensorRT 解析 ONNX 遇到不认识的节点时，按节点名 "RmsNorm"、版本 "1" 去 plugin 注册表里找 creator，
// 节点属性 eps 作为 PluginField 交给 createPlugin。
// 编出来是一个 .so：trtexec --plugins=librmsnorm_trt_plugin.so；Python 里先 ctypes.CDLL 加载再反序列化 engine。
// 和 PyTorch 扩展对应：getOutputDimensions ≈ Meta 实现（只推形状），enqueue ≈ CUDA 实现（真正启动 kernel）。
// 目前只支持 FP32、行优先连续（kLINEAR）。
#include <NvInfer.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "../launch.cuh"

using namespace nvinfer1;

namespace {
constexpr char const* kName = "RmsNorm";
constexpr char const* kVersion = "1";

class RmsNormPlugin : public IPluginV2DynamicExt {
public:
    explicit RmsNormPlugin(float eps) : eps_(eps) {}
    // 反序列化：engine 文件里只存了 eps 这一个 float（见 serialize）
    RmsNormPlugin(void const* data, size_t) { std::memcpy(&eps_, data, sizeof(float)); }

    // ===== 形状和类型：build engine 时调用 =====
    int32_t getNbOutputs() const noexcept override { return 1; }
    // 输出形状 = x 的形状（inputs[0]）
    DimsExprs getOutputDimensions(int32_t, DimsExprs const* inputs, int32_t, IExprBuilder&) noexcept override {
        return inputs[0];
    }
    DataType getOutputDataType(int32_t, DataType const* inputTypes, int32_t) const noexcept override {
        return inputTypes[0];
    }
    // pos 依次是 输入 x、输入 weight、输出 y；TensorRT 会挨个问"这种类型/布局你支持吗"
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t, int32_t) noexcept override {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    void configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept override {
        return 0;
    }

    // ===== 执行：每次推理调用，指针和 stream 由 TensorRT 给 =====
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                    void* const* outputs, void*, cudaStream_t stream) noexcept override {
        Dims const& d = inputDesc[0].dims;
        const int n = d.d[d.nbDims - 1];
        int rows = 1;                                  // 前面的维度全部展平成行
        for (int i = 0; i < d.nbDims - 1; ++i) rows *= d.d[i];
        const char* err = launchRMSNorm(rows, n, static_cast<float const*>(inputs[0]),
                                        static_cast<float const*>(inputs[1]), static_cast<float*>(outputs[0]),
                                        eps_, stream);
        if (err == nullptr && cudaPeekAtLastError() == cudaSuccess) return 0;
        std::fprintf(stderr, "RmsNorm plugin: %s\n", err ? err : cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    // ===== 存进 engine 文件 / 从文件读回：只有 eps =====
    size_t getSerializationSize() const noexcept override { return sizeof(float); }
    void serialize(void* buffer) const noexcept override { std::memcpy(buffer, &eps_, sizeof(float)); }

    // ===== 其余是 TensorRT 要求的固定写法 =====
    char const* getPluginType() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    IPluginV2DynamicExt* clone() const noexcept override {
        auto* p = new RmsNormPlugin(eps_);
        p->setPluginNamespace(ns_.c_str());
        return p;
    }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

private:
    // 基类里还有同名的旧接口（隐式 batch 维），这里只实现动态形状版；using 消除"同名函数被隐藏"的警告
    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2::enqueue;
    using IPluginV2Ext::configurePlugin;

    float eps_ = 1e-6f;
    std::string ns_;
};

// creator：TensorRT 按名字找到它，再用它创建 plugin（从 ONNX 属性创建，或从 engine 文件反序列化）
class RmsNormPluginCreator : public IPluginCreator {
public:
    RmsNormPluginCreator() {
        fields_.emplace_back("eps", nullptr, PluginFieldType::kFLOAT32, 1);
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }

    // 从 ONNX 节点建：fc 里是节点属性，找出 eps
    IPluginV2* createPlugin(char const*, PluginFieldCollection const* fc) noexcept override {
        float eps = 1e-6f;
        for (int i = 0; i < fc->nbFields; ++i)
            if (std::strcmp(fc->fields[i].name, "eps") == 0 && fc->fields[i].type == PluginFieldType::kFLOAT32)
                eps = *static_cast<float const*>(fc->fields[i].data);
        auto* p = new RmsNormPlugin(eps);
        p->setPluginNamespace(ns_.c_str());
        return p;
    }
    // 从 engine 文件建：serialData 就是 serialize 写进去的内容
    IPluginV2* deserializePlugin(char const*, void const* serialData, size_t serialLength) noexcept override {
        auto* p = new RmsNormPlugin(serialData, serialLength);
        p->setPluginNamespace(ns_.c_str());
        return p;
    }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

private:
    std::vector<PluginField> fields_;
    PluginFieldCollection fc_{};
    std::string ns_;
};
}  // namespace

// 加载 .so 时自动把 creator 登记到 TensorRT 的全局 plugin 注册表
REGISTER_TENSORRT_PLUGIN(RmsNormPluginCreator);
