#include <NvOnnxParser.h>
#include "NvInferPlugin.h"
#include <fstream>
#include "yolov11_pose.h"

#define RET_OK 0

using namespace nvinfer1;

/**
 * @brief converting .onnx model to .engine
 * @param model_path .onnx model path and output path for .engine
 * @param logger Nvinfer ILogger 
 */

YOLO_V11POSE::YOLO_V11POSE(_DL_CONFIG_PARAM& cfg)
{
    imgSize = cfg.imgSize;
    confThresh = cfg.rectConfidenceThreshold;
    iouThreshold = cfg.iouThreshold;
    topk = cfg.topk;
}

void YOLO_V11POSE::init(std::string model_path, nvinfer1::ILogger& logger, bool use_fp16)
{
    // Deserialize an engine
    if (model_path.find(".engine") != std::string::npos)
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
    else if  (model_path.find(".onnx") != std::string::npos){
        if (use_fp16)
        {
            build(model_path, logger, true);
            saveEngine(model_path);
        }
        else{
            build(model_path, logger, false);
            saveEngine(model_path);
        }
    }
    else{
        std::cerr << "Unsupported model file format. Provide .onnx or .engine file." << std::endl;
        return;
    }

#if NV_TENSORRT_MAJOR < 10
    // Define input dimensions
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    output_dims = engine->getBindingDimensions(1);
#else 
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0)); // images (1, 3, 640, 640)
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];

    output_dims = engine->getTensorShape(engine->getIOTensorName(1)); // output0 (1, 300, 6)
#endif
    // create CUDA stream
    cudaStreamCreate(&stream);

    cudaError_t state;
    size_t inputSize = 3 * input_h * input_w * sizeof(float);
    size_t outputSize =  output_dims.d[1] * output_dims.d[2] * sizeof(float);

    state = cudaMalloc(&buffer[0], inputSize); // input name: images
    if (state) {
        std::cout << "allocate input memory failed\n";
        std::abort();
    }

    state = cudaMalloc(&buffer[1], outputSize); // output name: output0
    if (state) {
        std::cout << "allocate output memory failed\n";
        std::abort();
    }
}


/**
 * @brief Destroy the DLModel::DLModel object
 * 
 */
YOLO_V11POSE::~YOLO_V11POSE(){
    if (stream)
        cudaStreamDestroy(stream);
    if (buffer[0])
        cudaFree(buffer[0]);
    if (buffer[1])
        cudaFree(buffer[1]);
    // if (context)
    //     context->destroy();
    // if (engine)
    //     engine->destroy();
    // if (runtime)
    //     runtime->destroy();
}

/**
 * @brief build .engine for inference
 * 
 * @param onnxPath 
 * @param logger 
 */
void YOLO_V11POSE::build(std::string onnxPath, nvinfer1::ILogger& logger, bool use_fp16)
{
    auto builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig* config = builder->createBuilderConfig();
    if (use_fp16)
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
    delete builder;
}

bool YOLO_V11POSE::saveEngine(const std::string& onnxpath)
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

/**
 * @brief 
 * 
 * @param iImage raw image
 * @param data preprocessed image
 */
void YOLO_V11POSE::preprocess(const cv::Mat& img, cv::Mat& out, bool use_fp16=false)
{
    cv::Mat mat;

    img_height = img.rows;
    img_width = img.cols;

    float r = std::min(input_h / img_height, input_w / img_width);
    int padw = std::round(img_width * r);
    int padh = std::round(img_height * r);

    cv::Mat tmp;
    if ((int)img_width != padw || (int)img_height != padh) {
        cv::resize(img, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = img.clone();
    }

    dw = input_w - padw;
    dh = input_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;

    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);

    if (use_fp16)
    {
        out.convertTo(out, CV_16F);
    }
    // update the ratio for resizing
    ratio  = 1 / r;
}

/**
 * @brief object detection prediction, only for tensorrt > 10
 * 
 * @param img input image
 */
std::vector<_DL_RESULT> YOLO_V11POSE::predict(cv::Mat& img, std::vector<_DL_RESULT>& prediction_results, bool use_fp16)
{
    // Preprocessing
    cv::Mat inputData;
    size_t inputSize;
    std::vector<float> outputData(output_dims.d[1] * output_dims.d[2]);

    if (!use_fp16) // float32 model
    {   
        // preprocessing
        preprocess(img, inputData, false);
        inputSize = inputData.total() * sizeof(float);

        // copying input tensor to cuda device
        cudaMemcpyAsync(buffer[0], inputData.ptr<float>(), inputSize, cudaMemcpyHostToDevice, stream);

        // inference
        context->executeV2(buffer);

        cudaStreamSynchronize(stream);

        // copying model prediction to host
        size_t outputSize = output_dims.d[1] * output_dims.d[2] * sizeof(float);
        cudaMemcpyAsync(outputData.data(), buffer[1], outputSize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // postprocessing
        postprocess(outputData.data(), prediction_results, topk);
    }
    else // float16 model
    {   
        // preprocessing
        preprocess(img, inputData, true);
        inputSize = inputData.total() * sizeof(__half);
        std::vector<__half> inputData_half(inputData.total());

        // converting input tp fp 16, might be unnecessary
        for (int i = 0; i < inputData.total(); ++i)
        {
            inputData_half[i] = __float2half(inputData.ptr<float>()[i]);
        }

        // copying input tensor to cuda device
        cudaMemcpyAsync(buffer[0], inputData.ptr<float>(), inputData.total() * inputData.elemSize(), cudaMemcpyHostToDevice, stream);

        // inference model with fp16 prediction
        context->executeV2(buffer);

        cudaStreamSynchronize(stream);

        // copying prediction results to host
        size_t outputSize = output_dims.d[1] * output_dims.d[2] * sizeof(__half);
        cudaMemcpyAsync(outputData.data(), buffer[1], outputSize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // postprocessing
        postprocess(outputData.data(), prediction_results, topk);
    }

    return prediction_results;
}

/**
 * @brief postprocessing
 * 
 */
void YOLO_V11POSE::postprocess(float* outputData, std::vector<_DL_RESULT>& res, int topk)
{   
    res.clear();

    std::vector<cv::Rect> positionBoxes; // bounding box
    std::vector<int> classIds;           // class IDs for detected objects
    std::vector<float> confidences;      // confidence scores for detected objects
    std::vector<std::vector<float>> kpss;
    std::vector<int> indices;

    auto num_channels = output_dims.d[1];
    auto num_anchors = output_dims.d[2];


    cv::Mat outputMat = cv::Mat(num_channels, num_anchors, CV_32F, outputData);
    outputMat = outputMat.t();

    for (int i=0; i < num_anchors; i++)
    {
        auto row_ptr = outputMat.row(i).ptr<float>();
        auto boxes_ptr = row_ptr;
        auto conf_ptr = row_ptr + 4;
        auto kps_ptr = row_ptr + 5;

        float confidence = *conf_ptr;

        if (confidence > confThresh) 
        {
            float x = *boxes_ptr++ - dw;
            float y = *boxes_ptr++ - dh;
            float w = *boxes_ptr++;
            float h = *boxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, img_width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, img_height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, img_width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, img_height);

            cv::Rect_<float> box;
            box.x      = x0;
            box.y      = y0;
            box.width  = x1 - x0;
            box.height = y1 - y0;

            std::vector<float> kps;
            for (int k = 0; k < 17; k++) 
            {
                float kps_x = (*(kps_ptr + 3 * k) - dw) * ratio;
                float kps_y = (*(kps_ptr + 3 * k + 1) - dh) * ratio;
                float kps_s = *(kps_ptr + 3 * k + 2);
                kps_x = clamp(kps_x, 0.f, img_width);
                kps_y = clamp(kps_y, 0.f, img_height);
                kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
            }

            positionBoxes.push_back(box);
            classIds.push_back(0); // Class ID is stored at position s + 5 in the 'result' array
            confidences.push_back(confidence); // Confidence score is stored at position s + 4 in the 'result' array
            kpss.push_back(kps);
        }
    }
#ifdef BATCHED_NMS
    cv::dnn:NMSBoxesBatched(positionBoxes, confidences, classIds, confThresh, iouThreshold, indices);
#else
    cv::dnn::NMSBoxes(positionBoxes, confidences, confThresh, iouThreshold, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) 
    {
        if (cnt >= topk)
        {
            break;
        }
        _DL_RESULT re;
        re.rect = positionBoxes[i];
        re.confidence = confidences[i];
        re.classId = classIds[i];
        re.kps = kpss[i];
        res.push_back(re);
        cnt += 1;
    }
}


void YOLO_V11POSE::draw_objects(const cv::Mat&                                image,
                               cv::Mat&                                      result,
                               const std::vector<_DL_RESULT>&                res,
                               const std::vector<std::vector<unsigned int>>& SKELETON,
                               const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                               const std::vector<std::vector<unsigned int>>& LIMB_COLORS)
{
    result = image.clone();
    const int num_point = 17;
    for (auto& re : res) {
        cv::rectangle(result, re.rect, {0, 0, 255}, 5);

        char text[256];
        sprintf(text, "person %.1f%%", re.confidence * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 2, 2, &baseLine);

        int x = (int)re.rect.x;
        int y = (int)re.rect.y + 1;

        if (y > result.rows)
            y = result.rows;

        cv::rectangle(result, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(result, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 2, {255, 255, 255}, 2);

        auto& kps = re.kps;
        for (int k = 0; k < num_point + 2; k++) {
            if (k < num_point) {
                int   kps_x = std::round(kps[k * 3]);
                int   kps_y = std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.5f) {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(result, {kps_x, kps_y}, 5, kps_color, -1);
                }
            }
            auto& ske    = SKELETON[k];
            int   pos1_x = std::round(kps[(ske[0] - 1) * 3]);
            int   pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.5f && pos2_s > 0.5f) {
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(result, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
            }
        }
    }
}