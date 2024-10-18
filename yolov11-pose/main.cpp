#include <iostream>
#include <string>
#include <tuple>
#include <cmath>
#include <unordered_map>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "yolov11_pose.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace std;

const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14},
                                                         {14, 12},
                                                         {17, 15},
                                                         {15, 13},
                                                         {12, 13},
                                                         {6, 12},
                                                         {7, 13},
                                                         {6, 7},
                                                         {6, 8},
                                                         {7, 9},
                                                         {8, 10},
                                                         {9, 11},
                                                         {2, 3},
                                                         {1, 2},
                                                         {1, 3},
                                                         {2, 4},
                                                         {3, 5},
                                                         {4, 6},
                                                         {5, 7}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0}};



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
    DL_CONFIG_PARAM config;

    unordered_map<string, string> options;
    string previous_option = "";
    string current_argument = "";
    YOLO_V11POSE model(config);
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
        cv::Mat  res;
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
            std::vector<_DL_RESULT> results = model.predict(frame, prediction_results, use_fp16);
            auto end = chrono::system_clock::now();
            double tpf = chrono::duration_cast<chrono::duration<double, std::milli>>(end - start).count();
            cout << "Time per frame: " << setw(9) << tpf << " ms, FPS:" << setw(4) << floor(100 / (tpf / 1000)) / 100.0 
            << ", height: " << model.img_height << ", width: " << model.img_width << endl;

            model.draw_objects(frame, res, results, SKELETON, KPS_COLORS, LIMB_COLORS);
            // model.drawBbox(frame, prediction_results);
            
            int target_width = model.img_width/2; // Keep original width or reduce it further if desired
            int target_height = model.img_height/2; // Reduce height to half for a smaller window
            cv::resize(res, res, cv::Size(target_width, target_height));  // Resize to show both images
            cv::imshow("YOLO V10 Object Detection", res);
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