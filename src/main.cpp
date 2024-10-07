#include <iostream>
#include <string>
#include <tuple>
#include <cmath>
#include <unordered_map>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "onnx_to_engine.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace std;

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

    unordered_map<string, string> options;
    string previous_option = "";
    string current_argument = "";
    DLModel model;
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
        cout << "Loading model: \"" << model_path << "\"" <<  endl;

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
        model.init(model_path, logger);
        cout << "Model successfully loaded." << endl << endl;
        model_loaded = true;
    }
}