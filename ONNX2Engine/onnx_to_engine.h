#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

class DLModel
{
public:
    DLModel();
    void init(std::string model_path, nvinfer1::ILogger& logger, bool use_fp16 = false);
    ~DLModel();

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    nvinfer1::INetworkDefinition* network;

    void build(std::string onnxPath, nvinfer1::ILogger& logger, bool use_fp16);
    bool saveEngine(const std::string& filename, bool use_fp16);
};