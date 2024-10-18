#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

enum MODEL_TYPE
{
    // FLOAT32 MODEL
    YOLO_POSE_V11 = 1,
    YOLO_DETECT_V11 = 2, 

    // FLOAT16 MODEL
    YOLO_POSE_V11_HALF = 3,
    YOLO_DETECT_V11_HALF = 4,
};

typedef struct _DL_CONFIG_PARAM
{
    MODEL_TYPE modelType;
    std::vector<int> imgSize = {640, 640}; // input image size for yolo
    int seg_h = 160;
    int seg_w = 160;
    int seg_channels = 32;
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    int	topk = 100;
} DL_CONFIG_PARAM;



typedef struct _DL_RESULT
{   
    cv::Rect_<float> rect;
    int classId;
    float conf;
    cv::Mat boxMask;
} DL_RESULT;

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

class YOLO_V11
{
public:
    YOLO_V11(_DL_CONFIG_PARAM& cfg);
    void init(std::string model_path, nvinfer1::ILogger& logger, bool use_fp16=false);
    void predict(cv::Mat& img, std::vector<_DL_RESULT>& prediction_results, bool use_fp16);
    void draw_objects(const cv::Mat&                                image,
                                cv::Mat&                                        result_img,
                                const std::vector<_DL_RESULT>&                      res,
                                const std::vector<std::string>&                 CLASS_NAMES,
                                const std::vector<std::vector<unsigned int>>&   COLORS,
                                const std::vector<std::vector<unsigned int>>&   MASK_COLORS);
    ~YOLO_V11();
    float img_height;   // camera image height
    float img_width;    // camera image width

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    nvinfer1::INetworkDefinition* network;

    void* buffer[3];
    cudaStream_t stream;

    float input_h;      // input height for YOLO
    float input_w;      // input width for YOLO
    float ratio;        // resize ratio
    float dw;
    float dh;
    nvinfer1::Dims output0_dims;
    nvinfer1::Dims output1_dims;

    std::vector<int> imgSize;
    float confThresh;
    float iouThreshold;
    int topk;
    int seg_h;
    int seg_w;
    int seg_channels;

    void build(std::string onnxPath, nvinfer1::ILogger& logger, bool use_fp16);
    bool saveEngine(const std::string& filename);
    void preprocess(const cv::Mat& iImage, cv::Mat& oImage, bool use_fp16);
    void postprocess(float* outputData0, float* outputData1, std::vector<_DL_RESULT>& res, int topk, int seg_channels, int seg_h, int seg_w);
};