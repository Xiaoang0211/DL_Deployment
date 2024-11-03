/**
 * @brief drone topic advertiser example, which 
 * @file drone_advertiser.cpp
 * @addtogroup examples
 * @author Xiaoang Zhang <jesse1008611@gmail.com>
 */

#include <px4_msgs/msg/obstacle_distance.hpp>
#include <gz/msgs/laserscan.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/camera_info.pb.h>
#include <gz/transport/Node.hh>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>


class DroneAdvertiser : public rclcpp::Node
{
public:
	DroneAdvertiser(const std::string &cameraInfoTopic, 
                    const std::string &imageTopic, 
                    const std::string &laserScanTopic) 
    : Node("gz_drone_advertiser")
    {
        ros_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/gz_camera/image", 10);
        ros_camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/gz_camera/camera_info", 10);
        ros_obstacle_distance_pub_ = this->create_publisher<px4_msgs::msg::ObstacleDistance>("/fmu/in/obstacle_distance", 10);

        gz_node_ = std::make_shared<gz::transport::Node>();

        // subscription to camera info
        if (!gz_node_->Subscribe(cameraInfoTopic, &DroneAdvertiser::cameraInfoCallback, this))
        {
            RCLCPP_ERROR(this->get_logger(), "Error subscribing to Gazebo Sim topic: %s", cameraInfoTopic.c_str());
        }

        // subscription to image captured by camera
        if (!gz_node_->Subscribe(imageTopic, &DroneAdvertiser::imageCallback, this))
        {
            RCLCPP_ERROR(this->get_logger(), "Error subscribing to Gazebo Sim topic: %s", imageTopic.c_str());
        }

        // subscription to 2d lidar, which serves as the distance sensor
        if (!gz_node_->Subscribe(laserScanTopic, &DroneAdvertiser::laserScanCallback, this))
        {
            RCLCPP_ERROR(this->get_logger(), "Error subscribing to Gazebo Sim topic [%s]", laserScanTopic.c_str());
        }
    }
private:
    std::shared_ptr<gz::transport::Node> gz_node_;
    std::string cameraInfoTopic;
    std::string imageTopic;

    // ROS 2 publisher: image, camera info, and obstacle distance
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ros_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr ros_camera_info_pub_;
    rclcpp::Publisher<px4_msgs::msg::ObstacleDistance>::SharedPtr ros_obstacle_distance_pub_;

    // Initialize ROS 2 CameraInfo
    sensor_msgs::msg::CameraInfo camera_info_msg_;

    void cameraInfoCallback(const gz::msgs::CameraInfo &msg)
    {
        // populate the ROS 2 CameraInfo message
        camera_info_msg_.header.stamp = this->get_clock()->now(); //change the timestamp to the original ones of the gz sim topics
        // int64_t total_nanoseconds = static_cast<int64_t>(camera_info_msg_.header.stamp.sec) * 1000000000LL + static_cast<int64_t>(camera_info_msg_.header.stamp.nanosec);
        camera_info_msg_.header.frame_id = "camera_frame";
        camera_info_msg_.width = msg.width();
        camera_info_msg_.height = msg.height();
        camera_info_msg_.distortion_model = "plumb_bob";

        // Copy distortion coefficients
        camera_info_msg_.d.assign(msg.distortion().k().begin(), msg.distortion().k().end());

        // Copy intrinsic matrix
        for (int i = 0; i < 9; ++i)
        {
            camera_info_msg_.k[i] = msg.intrinsics().k(i);
        }

        // Copy projection matrix
        for (int i = 0; i < 12; ++i) {
            camera_info_msg_.p[i] = msg.projection().p(i);
        }

        // Copy rectification matrix
        for (int i = 0; i < 9; ++i) {
            camera_info_msg_.r[i] = msg.rectification_matrix(i);
        }

        ros_camera_info_pub_->publish(camera_info_msg_);
    }

    void imageCallback(const gz::msgs::Image &msg)
    {
        int cv_type;

        if (msg.pixel_format_type() == gz::msgs::PixelFormatType::RGB_INT8)
        {
            cv_type = CV_8UC3;
        } else if (msg.pixel_format_type() == gz::msgs::PixelFormatType::L_INT8)
        {
            cv_type = CV_8UC1;
        } else
        {
            RCLCPP_ERROR(this->get_logger(), "Unsupported image format!");
            return;
        }

        cv::Mat image(msg.height(), msg.width(), cv_type, const_cast<char*>(msg.data().data()));
        
        // Convert RGB to BGR for OpenCV if necessary
        if (msg.pixel_format_type() == gz::msgs::PixelFormatType::RGB_INT8) {
            cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        }

        // Display the image 
        cv::imshow("Gazebo Camera Stream", image);
        cv::waitKey(1);

        // Convert OpenCV Mat to ROS 2 Image message using cv_bridge
        std_msgs::msg::Header header;
        header.stamp = this->get_clock()->now();
        header.frame_id = "camera_frame";

        sensor_msgs::msg::Image::SharedPtr ros_image = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
        ros_image_pub_->publish(*ros_image);
    }

    void laserScanCallback(const gz::msgs::LaserScan &scan)
    {
        static constexpr int SECTOR_SIZE_DEG = 10; // PX4 Collision Prevention uses 36 sectors of 10 degrees each

        double angle_min_deg = scan.angle_min() * 180 / M_PI;
        double angle_step_deg = scan.angle_step() * 180 / M_PI;

        int samples_per_sector = std::round(SECTOR_SIZE_DEG / angle_step_deg);
        int number_of_sectors = scan.ranges_size() / samples_per_sector;

        std::vector<double> ds_array(number_of_sectors, std::numeric_limits<uint16_t>::max());

        // Downsample -- take average of samples per sector
        for (int i = 0; i < number_of_sectors; i++) {
        double sum = 0;
        int samples_used_in_sector = 0;

        for (int j = 0; j < samples_per_sector; j++) {
            int index = i * samples_per_sector + j;

            if (index >= scan.ranges_size()) {
            break;
            }

            double distance = scan.ranges(index);

            // Inf values mean no object
            if (std::isinf(distance)) {
            continue;
            }

            sum += distance;
            samples_used_in_sector++;
        }

        // If all samples in a sector are inf then it means the sector is clear
        if (samples_used_in_sector == 0) {
            ds_array[i] = scan.range_max();

        } else {
            ds_array[i] = sum / samples_used_in_sector;
        }
        }

        // Publish to ObstacleDistance
        px4_msgs::msg::ObstacleDistance obs{};

        // Initialize distances with unknown values
        std::fill(std::begin(obs.distances), std::end(obs.distances), UINT16_MAX);

        obs.timestamp = this->get_clock()->now().nanoseconds() / 1000; // timestamp in microseconds, since it is the convention of PX4 autopilot
        obs.frame = px4_msgs::msg::ObstacleDistance::MAV_FRAME_BODY_FRD;
        obs.sensor_type = px4_msgs::msg::ObstacleDistance::MAV_DISTANCE_SENSOR_LASER;
        obs.min_distance = static_cast<uint16_t>(scan.range_min() * 100); // in cm
        obs.max_distance = static_cast<uint16_t>(scan.range_max() * 100); // in cm
        obs.angle_offset = static_cast<float>(angle_min_deg * M_PI / 180.0); // in radians
        obs.increment = static_cast<float>(SECTOR_SIZE_DEG * M_PI / 180.0); // in radians

        // Map samples in FOV into sectors in ObstacleDistance
        int index = 0;

        // Iterate in reverse because array is FLU and we need FRD
        for (auto it = ds_array.rbegin(); it != ds_array.rend(); ++it) {
        uint16_t distance_cm = static_cast<uint16_t>((*it) * 100);

        if (distance_cm >= obs.max_distance) {
            obs.distances[index] = obs.max_distance + 1;

        } else if (distance_cm < obs.min_distance) {
            obs.distances[index] = 0;

        } else {
            obs.distances[index] = distance_cm;
        }

        index++;
        }

        // Publish the ObstacleDistance message
        ros_obstacle_distance_pub_->publish(obs);
    }
};

int main(int argc, char *argv[])
{
	std::cout << "Starting drone advertiser node..." << std::endl;
	setvbuf(stdout, NULL, _IONBF, BUFSIZ);
	rclcpp::init(argc, argv);

    // Gazebo Sim Topics to be subscribed
    std::string cameraInfoTopic = "/camera_info";
    std::string imageTopic = "/camera";
    std::string laserScanTopic = "/world/baylands/model/x500_cam_2dlidar_0/model/lidar/link/link/sensor/lidar_2d_v2/scan";

    // Instantiate GZROSCameraAdavertiser and start listening
    auto drone_advertiser = std::make_shared<DroneAdvertiser>(cameraInfoTopic, imageTopic, laserScanTopic);
	rclcpp::spin(drone_advertiser);
	rclcpp::shutdown();

	return 0;
}
