# DL_Deployment
Goal of this project is to develop a software for an UAV that can do the following tasks:
1. The UAV explores a new environment with OKVIS 2 and obstcale avoidance algorithm.
2. A 3D Gaussian 

## Converting `.onnx` to `.engine`

In the `/ONNX2Engine` directory, you will find the files for converting `.onnx` models to `.engine` files. These files can be customized to suit other deployment needs.

### Compilation Instructions

Before compiling, you may need to install the following dependencies:
- CUDA
- OpenCV
- TensorRT

Additionally, ensure that you adjust the directory paths to point to the correct TensorRT installation location on your system.

To compile the C++ files, follow these steps taking yolov11-segment as example:
   ```bash
   cd yolov11-segment
   mkdir build
   cd build
   cmake ..
   make
   ```
### Pre-trained weights

You can download the pre-trained weights in .engine format in the following link. Currently, Depth-Anything-Large (vitl) and Yolov10 are available.

https://drive.google.com/drive/folders/1mT8IovtHt9k0CL8TIHs4qFea9LiXGdSW?usp=drive_link