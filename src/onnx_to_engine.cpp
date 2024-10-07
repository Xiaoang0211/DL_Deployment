#include <NvOnnxParser.h>
#include "onnx_to_engine.h"

#define isFP16 true

using namespace nvinfer1;

/**
 * @brief converting .onnx model to .engine+
 * @param model_path .onnx model path and output path for .engine
 * @param logger Nvinfer ILogger 
 */
DLModel::DLModel()
{}

void DLModel::init(std::string model_path, nvinfer1::ILogger& logger)
{
    // Deserialize an engine
    if (model_path.find(".onnx") == std::string::npos)
    {
        std::ifstream engineStream(model_path, std::ios::binary);
        engineStream.seekg(0, std::ios::end);
        const size_t modelSize = engineStream.tellg();
        engineStream.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engineData(new char[modelSize]);
        engineStream.read(engineData.get(), modelSize);
        engineStream.close();

        // create tensorrt model
        runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
        context = engine->createExecutionContext();

    }
    // Build an engine from an onnx model
    else{
        build(model_path, logger);
        saveEngine(model_path);
    }
}

/**
 * @brief Destroy the DLModel::DLModel object
 * 
 */
DLModel::~DLModel(){}

/**
 * @brief build .engine for inference
 * 
 * @param onnxPath 
 * @param logger 
 */
void DLModel::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    auto builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig* config = builder->createBuilderConfig();
    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config)};

    runtime = createInferRuntime(logger);

    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    context = engine->createExecutionContext();

    delete network;
    delete config;
    delete parser;
    delete plan;
}

bool DLModel::saveEngine(const std::string& onnxpath)
{
    // find engine path from onnx path
    std::string engine_path;
    size_t dot_index = onnxpath.find_last_of(".");
    if (dot_index != std::string::npos) {
        engine_path = onnxpath.substr(0, dot_index) + ".engine";
    } 
    else
    {
        return false;
    }

    // save the engine to the path
    if (engine)
    {
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            std::cout << "Create engine file" << engine_path << "failed." << std::endl;
            return 0;
        }
        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}