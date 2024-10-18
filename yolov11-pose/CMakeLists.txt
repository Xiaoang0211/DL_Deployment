cmake_minimum_required(VERSION 3.12)
project(yolov11-pose)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add source files
set(SOURCES
    yolov11_pose.cpp
    main.cpp
    # ... add more source files here if needed
)

# Add headers
set(HEADERS
    yolov11_pose.h
    # ... add more header files here if needed
)

# Find and include CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find and include OpenCV 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set TensorRT paths
set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")

# Include TensorRT
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIB_DIR})

# Read TensorRT version from NvInferVersion.h
file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" tensorrt_version REGEX "#define NV_TENSORRT_MAJOR +[0-9]+")
string(REGEX MATCH "[0-9]+" tensorrt_version_major ${tensorrt_version})

# Check TensorRT version and set libraries accordingly
if (tensorrt_version_major EQUAL 10)
    set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
else()
    set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
endif()

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)
