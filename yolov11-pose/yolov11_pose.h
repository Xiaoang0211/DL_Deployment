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

typedef struct _DL_CONFIG_PARAM
{
    std::vector<int> imgSize = {640, 640}; // input image size for yolo
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    int	topk = 100;
} DL_CONFIG_PARAM;



typedef struct _DL_RESULT
{   
    cv::Rect_<float> rect;
    int classId;
    float confidence;
    std::vector<float> kps;
} DL_RESULT;

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

class YOLO_V11POSE
{
public:
    YOLO_V11POSE(_DL_CONFIG_PARAM& cfg);
    void init(std::string model_path, nvinfer1::ILogger& logger, bool use_fp16=false);
    std::vector<_DL_RESULT> predict(cv::Mat& img, std::vector<_DL_RESULT>& prediction_results, bool use_fp16);
    void drawBbox(cv::Mat& img, std::vector<_DL_RESULT>& results);
    void draw_objects(const cv::Mat&                                         image,
                               cv::Mat&                                      result,
                               const std::vector<_DL_RESULT>&                res,
                               const std::vector<std::vector<unsigned int>>& SKELETON,
                               const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                               const std::vector<std::vector<unsigned int>>& LIMB_COLORS);
    ~YOLO_V11POSE();
    float img_height;   // camera image height
    float img_width;    // camera image width

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    nvinfer1::INetworkDefinition* network;

    void* buffer[2];
    cudaStream_t stream;

    float input_h;      // input height for YOLO
    float input_w;      // input width for YOLO
    float ratio;        // resize ratio
    float dw;
    float dh;
    nvinfer1::Dims output_dims;

    std::vector<int> imgSize;
    float confThresh;
    float iouThreshold;
    int topk;

    void build(std::string onnxPath, nvinfer1::ILogger& logger, bool use_fp16);
    bool saveEngine(const std::string& filename);
    void preprocess(const cv::Mat& iImage, cv::Mat& oImage, bool use_fp16);
    void postprocess(float* outputData, std::vector<_DL_RESULT>& res, int topk);
};