#include <iostream>
#include <string>
#include <tuple>
#include <cmath>
#include <unordered_map>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "yolov11_segment.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace std;

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49},  {72, 249, 10}, {146, 204, 23},
    {61, 219, 134}, {26, 147, 52},   {0, 212, 187},  {44, 153, 168}, {0, 194, 255},   {52, 69, 147}, {100, 115, 255},
    {0, 24, 236},   {132, 56, 255},  {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};




bool IsPathExist(const std::string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const std::string& path) {
    if (!IsPathExist(path)) {
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}


class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            cout << msg << endl;
    }
}logger;

int main(int argc, char** argv) {
    bool model_loaded = false;
    bool use_fp16 = false;

    unordered_map<string, string> options;
    string previous_option = "";
    string current_argument = "";
    DL_CONFIG_PARAM config;
    YOLO_V11 model(config);
    int cutoff = 0;
    for (int i = 0; i < argc; i++) {
        cutoff = 0;
        current_argument = argv[i];
        if (current_argument[0] == '-') {
            if (current_argument[1] == '-') {
                cutoff = 2;
            } else {
                cutoff = 1;
            }
            previous_option = current_argument.substr(cutoff);
            options[previous_option] = "1";
        } else if (!previous_option.empty()) {
            options[previous_option] = current_argument.substr(cutoff);
            previous_option = "";
        }
    }

    // load selected model
    if (!options["model"].empty()) {
        string model_path = options["model"];

        if (!IsFile(model_path)) {
            cout << "Model not found!" << endl;
            abort();
        }

        if (model_path.size() >= 10 && model_path.substr(model_path.size() - 10) == "-half.engine") 
        {
            use_fp16 = true;
            cout << "Using model with FP16 precision." << endl;
        }

        cout << "Loading model: \"" << model_path << "\"" <<  endl;
        bool use_fp16 = (model_path.find("half") != string::npos);
        string alternate_path = model_path.substr(0, model_path.length() - 5) + ".engine";
        if (model_path.substr(model_path.find_last_of('.') + 1) == "onnx" && IsFile(alternate_path)) {
            if (options["find-engine"].empty()) {
                string confirm_engine = "";
                cout << "\"" << alternate_path << "\" found, Override existing .engine file? (Y/N): ";
                cin >> confirm_engine;
                if (confirm_engine != "Y" && confirm_engine != "y"){
                    model_path = alternate_path;
                }
            } else {
                model_path = alternate_path;
            }
        }
        if (use_fp16){
            model.init(model_path, logger, true);
        }
        else
        {
            model.init(model_path, logger);
        }
        cout << "Model successfully loaded." << endl << endl;
        model_loaded = true;
    }

    if (!options["camera"].empty())
    {
        if (!model_loaded) {
            cout << "Model not loaded. Please specify a modelpath using --model." << endl;
            return -1;
        }

        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open camera." << endl;
            return -1;
        }

        cout << "Camera opened successfully." << endl;
        std::vector<_DL_RESULT> prediction_results;
        cv::Mat  result_img;
        while (true) 
        {
            cv::Mat frame;
            cap >> frame;

            if (frame.empty())
            {
                cerr << "Error: Captured empty frame." << endl;
                break;
            }

            auto start = chrono::system_clock::now();
            model.predict(frame, prediction_results, use_fp16);
            auto end = chrono::system_clock::now();
            double tpf = chrono::duration_cast<chrono::duration<double, std::milli>>(end - start).count();
            cout << "Time per frame: " << setw(9) << tpf << " ms, FPS:" << setw(4) << floor(100 / (tpf / 1000)) / 100.0 << endl;
            // << ", height: " << model.img_height << ", width: " << model.img_width << endl;

            model.draw_objects(frame, result_img, prediction_results, CLASS_NAMES, COLORS, MASK_COLORS);
            
            int target_width = model.img_width/2; // Keep original width or reduce it further if desired
            int target_height = model.img_height/2; // Reduce height to half for a smaller window
            cv::resize(result_img, result_img, cv::Size(target_width, target_height));  // Resize to show both images
            cv::imshow("YOLO V10 Object Detection", result_img);
            if (cv::waitKey(1) == 'q')
            {
                break;
            }
        }

        // Release resources
        cap.release();
        cv::destroyAllWindows();
        cout << "Camera closed and resources released." << endl; 
    }
    return 0;
}