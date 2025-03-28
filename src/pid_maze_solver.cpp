#include "ament_index_cpp/get_package_share_directory.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp" // Include this header
#include "yaml-cpp/yaml.h" // include the yaml library
#include <Eigen/Dense>
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
  bool check_obstacles();
  std::vector<float> laser_ranges_;
  std::array<int, 4> dir_indices;
  double min_safe_distance_ = 0.20;
  double angle_increment_;

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

// Function to find index for a given angle and specific points
int find_index_scan_msg(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg,
                        double angle) {
  if (!scan_msg) {
    RCLCPP_ERROR(rclcpp::get_logger("find_index_scan_msg"),
                 "Invalid LaserScan message.");
    return -1;
  }

  if (angle < scan_msg->angle_min || angle > scan_msg->angle_max) {
    std::ostringstream error_msg;
    error_msg << "Angle out of bounds, [" << scan_msg->angle_min << " ,"
              << scan_msg->angle_max << "]";
    throw std::invalid_argument(error_msg.str());
  }

  auto angle_to_index = [&](double ang) -> int {
    return static_cast<int>((ang - scan_msg->angle_min) /
                            scan_msg->angle_increment);
  };

  int temp = angle_to_index(angle);
  RCLCPP_DEBUG(rclcpp::get_logger("find_index_scan_msg"),
               "Angle: %f, Index: %d", angle, temp);
  return temp;
}

void PIDMazeSolver::laser_callback(
    const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  laser_ranges_ = msg->ranges;
  angle_increment_ = msg->angle_increment;
  dir_indices[0] = find_index_scan_msg(msg, 0.0);
  dir_indices[1] = find_index_scan_msg(msg, M_PI / 2);
  dir_indices[2] = find_index_scan_msg(msg, M_PI);
  dir_indices[3] = find_index_scan_msg(msg, -M_PI / 2);
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

bool PIDMazeSolver::check_obstacles() {
  if (laser_ranges_.empty())
    return false;

  // Check front
  int center_index = dir_indices[0];

  int cone_width = 10;

  for (int i = center_index - cone_width; i <= center_index + cone_width; i++) {
    if (i >= 0 && i < int(laser_ranges_.size()) &&
        !std::isinf(laser_ranges_[i]) &&
        laser_ranges_[i] < min_safe_distance_) {
      RCLCPP_DEBUG(this->get_logger(),
                   "center_index: %d, index: %d, distance: %.2f", center_index,
                   i, laser_ranges_[i]);
      return true;
    }
  }
  return false;
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
      // Check for obstacles first
      if (check_obstacles()) {
        // Stop and avoid obstacle
        cmd_vel.linear.x = 0.0;
        cmd_vel.linear.y = 0.0;
        cmd_vel.angular.z = 0.0; // Simple turn right
        cmd_vel_publisher_->publish(cmd_vel);
        rate.sleep();
        break;
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
      }
      // Scale down linear velocity when we need to make significant turns
      if (fabs(angular_vel) > 0.1) {
        linear_vel *= 0.7; // Reduce speed when correcting course
      }

      cmd_vel.linear.x = direction_x * linear_vel;
      cmd_vel.linear.y = direction_y * linear_vel;
      cmd_vel.angular.z = angular_vel;
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
