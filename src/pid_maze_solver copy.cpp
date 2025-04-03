#include "ament_index_cpp/get_package_share_directory.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp" // Include this header
#include "yaml-cpp/yaml.h" // include the yaml library
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem> // Include the filesystem library
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <unordered_map>
#include <vector>

using namespace std::chrono_literals;

class PID {
public:
  PID() : kp_(0.0), ki_(0.0), kd_(0.0), dt_(0.0) {}

  PID(double kp, double ki, double kd, double dt)
      : kp_(kp), ki_(ki), kd_(kd), dt_(dt), integral_(0.0), prev_error_(0.0) {}

  double compute(double error) {
    integral_ += error * dt_;
    integral_ = std::clamp(integral_, -0.5, 0.5); // Anti-windup
    double derivative = (error - prev_error_) / dt_;
    double output = kp_ * error + ki_ * integral_ + kd_ * derivative;
    prev_error_ = error;
    return output;
  }

  void reset_() {
    integral_ = 0.0;
    prev_error_ = 0.0;
  }

private:
  double kp_, ki_, kd_, dt_;
  double integral_;
  double prev_error_;
};

struct DirectionIndices {
  int front;
  int left;
  int right;
  int behind;
};

DirectionIndices
getDirectionIndices(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg) {
  DirectionIndices indices;

  // Calculate index for front (0 degrees)
  indices.front =
      static_cast<int>((0.0 - scan_msg->angle_min) / scan_msg->angle_increment);

  // Calculate index for left (90 degrees, π/2 radians)
  indices.left = static_cast<int>((M_PI / 2 - scan_msg->angle_min) /
                                  scan_msg->angle_increment);

  // Calculate index for right (-90 degrees, -π/2 radians)
  indices.right = static_cast<int>((-M_PI / 2 - scan_msg->angle_min) /
                                   scan_msg->angle_increment);

  // Calculate index for behind (180 degrees, π radians)
  indices.behind = static_cast<int>((M_PI - scan_msg->angle_min) /
                                    scan_msg->angle_increment);

  // Handle wrap-around for circular LIDAR scans
  const int total_points = scan_msg->ranges.size();
  indices.front = (indices.front % total_points + total_points) % total_points;
  indices.left = (indices.left % total_points + total_points) % total_points;
  indices.right = (indices.right % total_points + total_points) % total_points;
  indices.behind =
      (indices.behind % total_points + total_points) % total_points;

  return indices;
}

DirectionIndices getRosbotXLDirectionIndices(
    const sensor_msgs::msg::LaserScan::SharedPtr scan_msg) {
  DirectionIndices indices;
  const int total_points = scan_msg->ranges.size();

  // Adjustments for upside-down mounting
  indices.front = static_cast<int>((M_PI - scan_msg->angle_min) /
                                   scan_msg->angle_increment);
  indices.left = static_cast<int>((3 * M_PI / 2 - scan_msg->angle_min) /
                                  scan_msg->angle_increment);
  indices.right = static_cast<int>((M_PI / 2 - scan_msg->angle_min) /
                                   scan_msg->angle_increment);
  indices.behind =
      static_cast<int>((0 - scan_msg->angle_min) / scan_msg->angle_increment);

  // Wrap-around handling
  indices.front %= total_points;
  indices.left %= total_points;
  indices.right %= total_points;
  indices.behind %= total_points;

  // Ensure positive indices
  indices.front = (indices.front + total_points) % total_points;
  indices.left = (indices.left + total_points) % total_points;
  indices.right = (indices.right + total_points) % total_points;
  indices.behind = (indices.behind + total_points) % total_points;

  return indices;
}

class PIDMazeSolver : public rclcpp::Node {
private:
  int scene_number_;
  double max_velocity_;
  double max_ang_velocity_;
  std::vector<std::vector<double>> waypoints_; //{dx, dy, dphi}

  void readWaypointsYAML();
  void pid_controller();
  std::array<double, 3> local2globalframe(double vx, double vy, double avz);
  std::array<double, 3> global2localframe(double vx, double vy, double avz);

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::CallbackGroup::SharedPtr timer_cb_grp_;
  rclcpp::CallbackGroup::SharedPtr odom_cb_grp_;

  geometry_msgs::msg::Point current_position_;
  double phi; // current_yaw_

  PID pid_linear_, pid_angular_;
  double time_step = 0.01; // in milliseconds

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr
      laser_subscriber_;
  void laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  bool check_obstacles_pid(int index);
  std::vector<float> laser_ranges_;
  double min_safe_distance_ = 0.20;
  double angle_increment_, range_min_, range_max_;
  DirectionIndices direction_indices_;
  bool first_run_ = true;
  bool is_simulation = true;
  void publish_vel_(double linx, double liny, double angz) {
    geometry_msgs::msg::Twist msg;
    msg.linear.x = linx;
    msg.linear.y = liny;
    msg.angular.z = angz;
    cmd_vel_publisher_->publish(msg);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

public:
  PIDMazeSolver(int scene_number);
  ~PIDMazeSolver();
};

PIDMazeSolver::~PIDMazeSolver() {
  RCLCPP_INFO(this->get_logger(), "Maze Solver Terminated.");
}

// Function to normalize angle to the range -pi to pi
double normalize_angle(double angle) {
  while (angle > M_PI)
    angle -= 2.0 * M_PI;
  while (angle < -M_PI)
    angle += 2.0 * M_PI;
  return angle;
}

double avg_distance(const std::vector<float> &ranges, int start, int end,
                    float min_valid_range, float max_valid_range) {
  int count = 0;
  double distance = 0.0;
  for (int i = start; i <= end; ++i) {
    const float range = ranges[i];
    if (!std::isnan(range) && !std::isinf(range) && range >= min_valid_range &&
        range <= max_valid_range) {
      distance += range;
      count++;
    }
  }
  return round(distance / count, 2);
}

bool find_wall_distance(
    const std::vector<float> &ranges, int half_window_size, int center_index,
    float min_valid_range = 0.0f,
    float max_valid_range = std::numeric_limits<float>::infinity()) {
  // Validate inputs
  const int length = ranges.size();
  if (length == 0 || half_window_size <= 0 || center_index < 0 ||
      center_index >= length) {
    return false;
  }

  // Calculate window bounds with wrap-around handling
  const int start_index = (center_index - half_window_size + length) % length;
  const int end_index = (center_index + half_window_size) % length;
  double distance = 0.0;

  if (start_index <= end_index) {
    // Normal case: window doesn't wrap around
    distance = avg_distance(ranges, start_index, end_index, min_valid_range,
                            max_valid_range);
  } else {
    // Window wraps around the end of the array
    distance = avg_distance(ranges, start_index, length - 1, min_valid_range,
                            max_valid_range);
    distance +=
        avg_distance(ranges, 0, end_index, min_valid_range, max_valid_range);
    distance /= 2.0;
  }

  return distance;
}

void PIDMazeSolver::odom_callback(
    const nav_msgs::msg::Odometry::SharedPtr msg) {
  // Extract position
  current_position_ = msg->pose.pose.position;

  // Extract orientation (quaternion)
  tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                    msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);

  // Convert quaternion to Euler angles (roll, pitch, yaw)
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  phi = yaw;

  // Log the position and orientation for debugging
  RCLCPP_DEBUG(this->get_logger(),
               "Position: [x: %f, y: %f, z: %f], Orientation (yaw): %f",
               current_position_.x, current_position_.y, current_position_.z,
               phi);
}

void PIDMazeSolver::laser_callback(
    const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  laser_ranges_ = msg->ranges;
  if (first_run_) {
    angle_increment_ = msg->angle_increment;
    range_min_ = msg->range_min;
    range_max_ = msg->range_max;
    if (is_simulation) {
      direction_indices_ = getDirectionIndices(msg);
    } else {
      direction_indices_ = getRosbotXLDirectionIndices(msg);
    }

    first_run_ = false;
    // dir_indices[0] = 0;
    // dir_indices[1] = 178;
    // dir_indices[2] = 358;
    // dir_indices[3] = 540;
    RCLCPP_INFO(this->get_logger(), "Dir indices: %d, %d, %d, %d",
                direction_indices_.front, direction_indices_.left,
                direction_indices_.behind, direction_indices_.right);
    // Position the robot
  }
}

PIDMazeSolver::PIDMazeSolver(int scene_number)
    : Node("maze_solver_node"), scene_number_(scene_number) {
  timer_cb_grp_ =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  odom_cb_grp_ =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions options;
  options.callback_group = odom_cb_grp_;

  cmd_vel_publisher_ =
      this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odometry/filtered", 10,
      std::bind(&PIDMazeSolver::odom_callback, this, std::placeholders::_1),
      options);

  laser_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan_filtered", 10,
      std::bind(&PIDMazeSolver::laser_callback, this, std::placeholders::_1),
      options);

  // Based on simulation or real bot
  // Read waypoints from YAML file & update waypoints vector
  // And configure PID controllers
  readWaypointsYAML();

  RCLCPP_INFO(this->get_logger(), "Maze Solver Initialized.");

  timer_ = this->create_wall_timer(
      1s, std::bind(&PIDMazeSolver::pid_controller, this), timer_cb_grp_);
}

bool PIDMazeSolver::check_obstacles_pid(int ind) {
  return check_obstacles(laser_ranges_, 3, ind, min_safe_distance_, range_min_,
                         range_max_, 5);
}

void PIDMazeSolver::pid_controller() {
  geometry_msgs::msg::Twist cmd_vel;
  RCLCPP_INFO(this->get_logger(), "Trajectory started.");

  // Loop through each waypoint
  int index = 0;
  for (const auto &waypoint : waypoints_) {
    pid_linear_.reset_();
    pid_angular_.reset_();

    rclcpp::Rate rate(int(1 / time_step)); // Control loop frequency

    RCLCPP_INFO(this->get_logger(), "WP%u: [%.2f, %.2f, %.2f]", ++index,
                waypoint[0], waypoint[1], waypoint[2]);

    // Step 1. Turn only
    double error_z = std::numeric_limits<double>::max();
    cmd_vel.linear.x = 0.0;
    cmd_vel.linear.y = 0.0;
    double target_z = normalize_angle(phi + waypoint[2]);
    while (fabs(error_z) > 0.007) {
      if (!rclcpp::ok()) { // Check if ROS is still running
        RCLCPP_WARN(this->get_logger(), "Trajectory Canceled.");
        timer_->cancel();         // Stop the timer
        odom_subscriber_.reset(); // Kill the odometry subscription
        rclcpp::shutdown();
        return;
      }

      // Calculate error
      error_z = target_z - phi;
      error_z =
          atan2(sin(error_z), cos(error_z)); // Normalize error to [-pi, pi]
      RCLCPP_DEBUG(this->get_logger(), "Angle to target: %.2f", error_z);

      // PID control
      double angular_vel = pid_angular_.compute(error_z);
      angular_vel =
          std::clamp(angular_vel, -max_ang_velocity_, max_ang_velocity_);
      cmd_vel.angular.z = angular_vel;
      cmd_vel_publisher_->publish(cmd_vel);
      RCLCPP_DEBUG(this->get_logger(), "Angular vel: %.3f", cmd_vel.angular.z);

      rate.sleep(); // Maintain loop frequency
    }
    // Now stop the bot
    cmd_vel.angular.z = 0.0;
    cmd_vel_publisher_->publish(cmd_vel);

    // Step 2. Move while adjusting direction
    // Transform waypoint to global frame
    auto [dx, dy, dphi] =
        local2globalframe(waypoint[0], waypoint[1], waypoint[2]);
    double target_x = current_position_.x + dx;
    double target_y = current_position_.y + dy;
    double distance = std::numeric_limits<double>::max();
    while (distance > 0.01) {
      if (!rclcpp::ok()) { // Check if ROS is still running
        RCLCPP_WARN(this->get_logger(), "Trajectory Canceled.");
        timer_->cancel();         // Stop the timer
        odom_subscriber_.reset(); // Kill the odometry subscription
        rclcpp::shutdown();
        return;
      }

      // Calculate error in global frame
      error_z = target_z - phi;
      error_z =
          atan2(sin(error_z), cos(error_z)); // Normalize error to [-pi, pi]
      double error_x = target_x - current_position_.x;
      double error_y = target_y - current_position_.y;
      distance = std::hypot(error_x, error_y);

      // Transform error to robot frame
      auto [robot_frame_x, robot_frame_y, robot_frame_z] =
          global2localframe(error_x, error_y, 0.0);

      // Normalize direction
      double direction_x = robot_frame_x / distance;
      double direction_y = robot_frame_y / distance;
      RCLCPP_DEBUG(this->get_logger(), "Distance to target: %.2f", distance);
      RCLCPP_DEBUG(this->get_logger(), "Angle to target: %.2f", error_z);

      // PID control
      double linear_vel = pid_linear_.compute(distance);
      linear_vel = std::clamp(linear_vel, -max_velocity_, max_velocity_);

      double angular_vel = 0.0;
      if (error_z > 0.007) {
        angular_vel = pid_angular_.compute(error_z);
        angular_vel =
            std::clamp(angular_vel, -max_ang_velocity_, max_ang_velocity_);
        cmd_vel.linear.x = 0.0;
        cmd_vel.angular.z = angular_vel;
        cmd_vel_publisher_->publish(cmd_vel);
      }
      // Scale down linear velocity when we need to make significant turns
      if (fabs(angular_vel) > 0.1) {
        linear_vel *= 0.7; // Reduce speed when correcting course
      }

      cmd_vel.linear.x = direction_x * linear_vel;
      cmd_vel.linear.y = direction_y * linear_vel;
      cmd_vel.angular.z = 0;

      // Check for obstacles
      //   if (check_obstacles_pid(direction_indices_.front) &&
      //       cmd_vel.linear.x > 0) {
      //     cmd_vel.linear.x = 0.0;
      //     publish_vel_(-0.1, 0.0, 0.0);
      //     RCLCPP_INFO(this->get_logger(), "Obstacle detected: Front");
      //   } else if (check_obstacles_pid(direction_indices_.left) &&
      //              cmd_vel.linear.y > 0) {
      //     cmd_vel.linear.y = 0.0;
      //     publish_vel_(0.0, -0.1, 0.0);
      //     RCLCPP_INFO(this->get_logger(), "Obstacle detected: Left");
      //   } else if (check_obstacles_pid(direction_indices_.behind) &&
      //              cmd_vel.linear.x < 0) {
      //     cmd_vel.linear.x = 0.0;
      //     publish_vel_(0.1, 0.0, 0.0);
      //     RCLCPP_INFO(this->get_logger(), "Obstacle detected: Behind");
      //   } else if (check_obstacles_pid(direction_indices_.right) &&
      //              cmd_vel.linear.y < 0) {
      //     cmd_vel.linear.y = 0.0;
      //     publish_vel_(0.0, 0.1, 0.0);
      //     RCLCPP_INFO(this->get_logger(), "Obstacle detected: Right");
      //   }

      cmd_vel_publisher_->publish(cmd_vel);

      rate.sleep();
    }

    // Now stop the bot
    cmd_vel.linear.x = 0.0;
    cmd_vel.linear.y = 0.0;
    cmd_vel.angular.z = 0.0;
    cmd_vel_publisher_->publish(cmd_vel);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  RCLCPP_INFO(this->get_logger(), "Trajectory completed.");
  timer_->cancel();         // Stop the timer
  odom_subscriber_.reset(); // Kill the odometry subscription
  rclcpp::shutdown();
}

std::array<double, 3> PIDMazeSolver::global2localframe(double vx, double vy,
                                                       double avz) {
  Eigen::Vector3d velocity(avz, vx, vy);

  Eigen::Matrix3d R;
  R << 1, 0, 0, 0, std::cos(phi), std::sin(phi), 0, -std::sin(phi),
      std::cos(phi);

  Eigen::Vector3d twist = R * velocity;

  return {twist(1), twist(2), twist(0)}; // dx, dy, dphi
}

std::array<double, 3> PIDMazeSolver::local2globalframe(double vx, double vy,
                                                       double avz) {
  Eigen::Vector3d velocity(avz, vx, vy);

  Eigen::MatrixXd R(3, 3);                      // 3x3 matrix
  R.row(0) << 1, 0, 0;                          // Row 0
  R.row(1) << 0, std::cos(phi), -std::sin(phi); // Row 1
  R.row(2) << 0, std::sin(phi), std::cos(phi);  // Row 2

  Eigen::Vector3d twist = R * velocity;

  return {twist(1), twist(2), twist(0)};
}

void PIDMazeSolver::readWaypointsYAML() {
  /* Based on simulation or real bot
    Read waypoints from YAML file & update waypoints vector
    And configure PID controllers
  */

  // Get the package's share directory and append the YAML file path
  std::string package_share_directory =
      ament_index_cpp::get_package_share_directory("pid_maze_solver");

  std::string waypoint_file_name = "";

  switch (scene_number_) {
  case 1: // Simulation
    RCLCPP_INFO(this->get_logger(), "Welcome to Simulation!");
    waypoint_file_name = "waypoints_sim.yaml";

    /* https://husarion.com/manuals/rosbot-xl/
    Maximum translational velocity = 0.8 m/s
    Maximum rotational velocity = 180 deg/s (3.14 rad/s)
    */
    max_velocity_ = 0.8;
    max_ang_velocity_ = 3.14;
    pid_linear_ = PID(1.5, 0.01, 0.1, time_step);
    pid_angular_ = PID(2.0, 0.01, 0.30, time_step);
    break;

  case 2: // CyberWorld
    RCLCPP_INFO(this->get_logger(), "Welcome to CyberWorld!");
    waypoint_file_name = "waypoints_real.yaml";

    /* https://husarion.com/manuals/rosbot-xl/
    Maximum translational velocity = 0.8 m/s
    Maximum rotational velocity = 180 deg/s (3.14 rad/s)
    */
    max_velocity_ = 0.20;
    max_ang_velocity_ = 0.30;
    pid_linear_ = PID(2.0, 0.05, 0.3, time_step);
    pid_angular_ = PID(2.0, 0.01, 0.30, time_step);
    is_simulation = false;
    break;

  case 3: // Simulation Reverse
    RCLCPP_INFO(this->get_logger(), "Welcome to Simulation Reverse!");
    waypoint_file_name = "reverse_waypoints_sim.yaml";
    /* https://husarion.com/manuals/rosbot-xl/
    Maximum translational velocity = 0.8 m/s
    Maximum rotational velocity = 180 deg/s (3.14 rad/s)
    */
    max_velocity_ = 0.8;
    max_ang_velocity_ = 3.14;
    pid_linear_ = PID(1.5, 0.01, 0.1, time_step);
    pid_angular_ = PID(2.0, 0.01, 0.30, time_step);
    break;

  case 4: // CyberWorld Reverse
    RCLCPP_INFO(this->get_logger(), "Welcome to CyberWorld!");
    waypoint_file_name = "reverse_waypoints_real.yaml";

    /* https://husarion.com/manuals/rosbot-xl/
    Maximum translational velocity = 0.8 m/s
    Maximum rotational velocity = 180 deg/s (3.14 rad/s)
    */
    max_velocity_ = 0.20;
    max_ang_velocity_ = 0.30;
    pid_linear_ = PID(2.0, 0.05, 0.3, time_step);
    pid_angular_ = PID(2.0, 0.01, 0.30, time_step);
    is_simulation = false;
    break;

  default:
    RCLCPP_ERROR(this->get_logger(), "Invalid Scene Number: %d", scene_number_);
  }

  RCLCPP_DEBUG(this->get_logger(), "Waypoint file loaded: %s",
               waypoint_file_name.c_str());

  std::string yaml_file_path =
      package_share_directory + "/waypoints/" + waypoint_file_name;

  // Read points from the YAML file
  try {
    YAML::Node config = YAML::LoadFile(yaml_file_path);

    if (config["waypoints"]) {
      for (const auto &wp : config["waypoints"]) {
        if (wp.size() == 3) { // Ensure it's a valid waypoint
          waypoints_.push_back(
              {wp[0].as<double>(), wp[1].as<double>(), wp[2].as<double>()});
        } else {
          RCLCPP_WARN(rclcpp::get_logger("WaypointsLoader"),
                      "Skipping invalid waypoint with incorrect size.");
        }
      }
      RCLCPP_INFO(rclcpp::get_logger("WaypointsLoader"),
                  "Successfully loaded %zu waypoints.", waypoints_.size());
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("WaypointsLoader"),
                   "No 'waypoints' key found in the YAML file.");
    }
  } catch (const YAML::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load YAML file: %s", e.what());
  }
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  // Check if a scene number argument is provided
  int scene_number = 1; // Default scene number to simulation
  if (argc > 1) {
    scene_number = std::atoi(argv[1]);
  }
  // Check if the scene number is valid before creating the node
  if (scene_number < 1 && scene_number > 4) {
    std::cerr << "Error: Invalid Scene Number -- " << scene_number << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  // Add as input variable the scene number
  auto node = std::make_shared<PIDMazeSolver>(scene_number);
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
