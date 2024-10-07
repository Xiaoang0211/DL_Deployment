# DL_Deployment
This repository is focused on preparing deep learning models that are potentially going to be deployed on robots, such as UAVs (Unmanned Aerial Vehicles).

## Converting `.onnx` to `.engine`

In the `/src` directory, you will find C++ files designed for converting `.onnx` models to `.engine` files. These files can be customized to suit other deployment needs.

### Compilation Instructions

Before compiling the C++ files in the `/src` directory, you may need to install the following dependencies:
- CUDA
- OpenCV
- TensorRT

Additionally, ensure that you adjust the directory paths to point to the correct TensorRT installation location on your system.

To compile the C++ files, follow these steps:

1. Create a build directory:
   ```bash
   mkdir build

