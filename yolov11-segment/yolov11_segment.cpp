#include <NvOnnxParser.h>
#include "NvInferPlugin.h"
#include <fstream>
#include "yolov11_segment.h"

#define RET_OK 0

using namespace nvinfer1;

/**
 * @brief converting .onnx model to .engine
 * @param model_path .onnx model path and output path for .engine
 * @param logger Nvinfer ILogger 
 */

YOLO_V11::YOLO_V11(_DL_CONFIG_PARAM& cfg)
{
    imgSize = cfg.imgSize;
    confThresh = cfg.rectConfidenceThreshold;
    iouThreshold = cfg.iouThreshold;
    topk = cfg.topk;
    seg_h = cfg.seg_h;
    seg_w = cfg.seg_w;
    seg_channels = cfg.seg_channels;
}

void YOLO_V11::init(std::string model_path, nvinfer1::ILogger& logger, bool use_fp16)
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
    output0_dims = engine->getBindingDimensions(1);
    output1_dims = engine->getBindingDimensions(2);
#else // when using tensorrt 10
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0)); // images (1, 3, 640, 640)
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];

    output0_dims = engine->getTensorShape(engine->getIOTensorName(1)); // output0 () detection
    output1_dims = engine->getTensorShape(engine->getIOTensorName(2)); // output1 () segmentation
#endif
    // create CUDA stream
    cudaStreamCreate(&stream);

    cudaError_t state;
    size_t inputSize = 3 * input_h * input_w * sizeof(float);
    size_t output0Size =  output0_dims.d[1] * output0_dims.d[2] * sizeof(float);
    size_t output1Size =  output1_dims.d[1] * output1_dims.d[2]* output1_dims.d[3] * sizeof(float);

    state = cudaMalloc(&buffer[0], inputSize); // input name: images
    if (state) {
        std::cout << "allocate input memory failed\n";
        std::abort();
    }

    state = cudaMalloc(&buffer[1], output0Size); // output name: output0
    if (state) {
        std::cout << "allocate output0 memory failed\n";
        std::abort();
    }

    state = cudaMalloc(&buffer[2], output1Size); // output name: output0
    if (state) {
        std::cout << "allocate output1 memory failed\n";
        std::abort();
    }
}


/**
 * @brief Destroy the DLModel::DLModel object
 * 
 */
YOLO_V11::~YOLO_V11(){
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
void YOLO_V11::build(std::string onnxPath, nvinfer1::ILogger& logger, bool use_fp16)
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

bool YOLO_V11::saveEngine(const std::string& onnxpath)
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
void YOLO_V11::preprocess(const cv::Mat& img, cv::Mat& out, bool use_fp16=false)
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
void YOLO_V11::predict(cv::Mat& img, std::vector<_DL_RESULT>& prediction_results, bool use_fp16)
{
    // Preprocessing
    cv::Mat inputData;
    size_t inputSize;
    std::vector<float> output0Data(output0_dims.d[1] * output0_dims.d[2]);
    std::vector<float> output1Data(output1_dims.d[1] * output1_dims.d[2] * output1_dims.d[3]);

    if (!use_fp16) // float32 model
    {   
        // preprocessing
        preprocess(img, inputData, false);
        inputSize = inputData.total() * sizeof(float);
        size_t output0Size = output0_dims.d[1] * output0_dims.d[2] * sizeof(float);
        size_t output1Size = output1_dims.d[1] * output1_dims.d[2] * output1_dims.d[3] * sizeof(float);

        // copying input tensor to cuda device
        cudaMemcpyAsync(buffer[0], inputData.ptr<float>(), inputSize, cudaMemcpyHostToDevice, stream);

        // inference
        context->executeV2(buffer);

        cudaStreamSynchronize(stream);

        // copying model prediction to host
        cudaMemcpyAsync(output0Data.data(), buffer[1], output0Size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(output1Data.data(), buffer[2], output1Size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // postprocessing
        postprocess(output0Data.data(), output1Data.data(), prediction_results, topk, seg_channels, seg_h, seg_w);
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
        size_t output0Size = output0_dims.d[1] * output0_dims.d[2] * sizeof(__half);
        size_t output1Size = output1_dims.d[1] * output1_dims.d[2] * output1_dims.d[3] * sizeof(__half);
        cudaMemcpyAsync(output0Data.data(), buffer[1], output0Size, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(output1Data.data(), buffer[2], output1Size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // postprocessing
        postprocess(output0Data.data(), output1Data.data(), prediction_results, topk, seg_channels, seg_h, seg_w);
    }
}

/**
 * @brief postprocessing
 * 
 */
void YOLO_V11::postprocess(float* output0Data, float* output1Data, std::vector<_DL_RESULT>& res, int topk, int seg_channels, int seg_h, int seg_w)
{   
    res.clear();

    std::vector<cv::Rect> positionBoxes; // bounding box
    std::vector<int> classIds;           // class IDs for detected objects
    std::vector<float> confs;      // confidence scores for detected objects
    std::vector<cv::Mat> mask_confids; // confidence scores for segmenation mask
    std::vector<int> indices;

    auto num_channels = output0_dims.d[1];
    auto num_anchors = output0_dims.d[2];
    int num_classes = num_channels - seg_channels - 4;

    cv::Mat outputMat = cv::Mat(num_channels, num_anchors, CV_32F, output0Data);
    outputMat = outputMat.t();

    cv::Mat protos = cv::Mat(seg_channels, seg_h * seg_w, CV_32F, output1Data);
    for (int i=0; i < num_anchors; i++)
    {
        auto row_ptr = outputMat.row(i).ptr<float>();
        auto boxes_ptr = row_ptr;
        auto conf_ptr = row_ptr + 4;
        auto mask_confid_ptr = row_ptr + 4 + num_classes;
        auto max_conf_ptr = std::max_element(conf_ptr, conf_ptr + num_classes);

        float conf = *max_conf_ptr;

        if (conf > confThresh) 
        {
            float x = *boxes_ptr++ - dw;
            float y = *boxes_ptr++ - dh;
            float w = *boxes_ptr++;
            float h = *boxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, img_width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, img_height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, img_width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, img_height);

            int classId = max_conf_ptr - conf_ptr;
            cv::Rect_<float> box;
            box.x      = x0;
            box.y      = y0;
            box.width  = x1 - x0;
            box.height = y1 - y0;

            cv::Mat mask_confid = cv::Mat(1, seg_channels, CV_32F, mask_confid_ptr);

            positionBoxes.push_back(box);
            classIds.push_back(classId); // Class ID is stored at position s + 5 in the 'result' array
            confs.push_back(conf); // Confidence score is stored at position s + 4 in the 'result' array
            mask_confids.push_back(mask_confid);
        }
    }
#ifdef BATCHED_NMS
    cv::dnn:NMSBoxesBatched(positionBoxes, confs, classIds, confThresh, iouThreshold, indices);
#else
    cv::dnn::NMSBoxes(positionBoxes, confs, confThresh, iouThreshold, indices);
#endif

    cv::Mat masks;
    int cnt = 0;
    for (auto& i : indices)
    {
        if (cnt >= topk)
        {
            break;
        }
        cv::Rect tmp = positionBoxes[i];
        _DL_RESULT re;
        re.classId = classIds[i];
        re.rect = tmp;
        re.conf = confs[i];
        masks.push_back(mask_confids[i]);
        res.push_back(re);
        cnt += 1;
    }

    if (masks.empty()) {

    }
    else
    {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat   = matmulRes.reshape(indices.size(), {seg_h, seg_w});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / input_w * seg_w;
        int scale_dh = dh / input_h * seg_h;

        cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

        for (int i = 0; i < indices.size(); i++) {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size((int)img_width, (int)img_height), cv::INTER_LINEAR);
            res[i].boxMask = mask(res[i].rect) > 0.5f;
        }
    }  
}

void YOLO_V11::draw_objects(const cv::Mat&                                  image,
                              cv::Mat&                                      result_img,
                              const std::vector<_DL_RESULT>&                res,
                              const std::vector<std::string>&               CLASS_NAMES,
                              const std::vector<std::vector<unsigned int>>& COLORS,
                              const std::vector<std::vector<unsigned int>>& MASK_COLORS)
{
    result_img          = image.clone();
    cv::Mat mask = image.clone();
    for (auto& re : res) {
        int        idx   = re.classId;
        cv::Scalar color = cv::Scalar(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);
        cv::Scalar mask_color =
            cv::Scalar(MASK_COLORS[idx % 20][0], MASK_COLORS[idx % 20][1], MASK_COLORS[idx % 20][2]);
        cv::rectangle(result_img, re.rect, color, 10);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[idx].c_str(), re.conf * 100);
        mask(re.rect).setTo(mask_color, re.boxMask);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 2, 2, &baseLine);

        int x = (int)re.rect.x;
        int y = (int)re.rect.y + 1;

        if (y > result_img.rows)
            y = result_img.rows;

        cv::rectangle(result_img, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(result_img, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 2, {255, 255, 255}, 4);
    }
    cv::addWeighted(result_img, 0.5, mask, 0.8, 1, result_img);
}