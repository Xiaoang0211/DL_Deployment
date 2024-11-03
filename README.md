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

## Random Walk of Drone in Gazebo Sim

### Overview

This setup enables a drone to perform a random walk within a **Gazebo Sim Garden** environment using **ROS 2 Humble**. The drone navigates randomly while avoiding obstacles with a collision prevention algorithm adapted from the [PX4 Autopilot](https://github.com/PX4/PX4-Autopilot) repository.

### Drone Model

The simulation uses a custom drone model in PX4 Autopilot, designated as `"x500_mono_cam_2dliadr"`, which is equipped with:
- **Mono Camera**: For visual input.
- **2D LiDAR**: Publishes obstacle information on the uORB topic `ObstacleDistance` to assist with obstacle detection and avoidance.

### Components of the Simulation

To build a Visual-Inertial SLAM (VI-SLAM) simulation in Gazebo Sim, the following components are required:

1. **Collision Prevention**  
   - Adapted from the official [PX4 Autopilot](https://github.com/PX4/PX4-Autopilot) repository, this component enables the drone to detect and avoid obstacles in its surroundings, ensuring safe navigation during the random walk.

2. **Random Walk Algorithm**  
   - This component implements a random movement pattern for the drone within the simulated environment, while leveraging the collision prevention algorithm to maintain obstacle-free navigation.

3. **Visual-Inertial SLAM (VI-SLAM)**  
   - For SLAM, the open-source **okvis** VI-SLAM framework is integrated. This provides the necessary functionality for real-time localization and mapping using visual and inertial inputs. The [okvis2 repository](https://github.com/smartroboticslab/okvis2) provides further implementation details.

### Requirements and Setup

1. **Gazebo Sim Garden** (or any other compatible environment).
2. **ROS 2 Humble** to handle communication and control.
3. **PX4 Autopilot Integration** with collision prevention for obstacle avoidance.

### Steps

1. **Configure Collision Prevention**  
   Follow the [PX4 Autopilot](https://github.com/PX4/PX4-Autopilot) repository guidelines to adapt collision prevention for Gazebo Sim. Ensure that `ObstacleDistance` data from the 2D LiDAR is correctly published and subscribed to by the collision prevention module.

2. **Implement Random Walk**  
   Develop a random walk algorithm for the drone, making use of collision prevention to adjust the drone’s path in real-time to avoid collisions with obstacles.

3. **Integrate VI-SLAM (okvis)**  
   Set up the [okvis2](https://github.com/smartroboticslab/okvis2) VI-SLAM framework to process data from the mono camera and IMU, enabling real-time visual-inertial mapping within the Gazebo environment.

By following these steps, the drone will autonomously navigate in a random walk while actively preventing collisions and creating a SLAM-based map of its environment.
