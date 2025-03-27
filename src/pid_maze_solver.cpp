#include "ament_index_cpp/get_package_share_directory.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp" // Include this header
#include "yaml-cpp/yaml.h" // include the yaml library
#include <Eigen/Dense>
#include <filesystem> // Include the filesystem library
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
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
  std::unordered_map<std::string, std::vector<float>> motions;

  void readWaypointsYAML();
  void pid_controller();
  std::vector<double> global2localvelocity(double vx, double vy, double avz);
  std::vector<double> velocity2wheelspeed(double wz, double vx, double vy);

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::CallbackGroup::SharedPtr timer_cb_grp_;
  rclcpp::CallbackGroup::SharedPtr odom_cb_grp_;

  geometry_msgs::msg::Point current_position_;
  double phi; // current_yaw_

  PID pid_x_, pid_y_, pid_z_;
  double time_step = 0.01; // in milliseconds

  // Robot parameters
  double l;
  double r;
  double w;

  // Transformation matrix H_4x3
  Eigen::MatrixXd H;
  // Pseudo-inverse of H
  Eigen::MatrixXd H_pseudo_inverse;
  // vector wheel speeds
  Eigen::Vector4d u;

public:
  PIDMazeSolver(int scene_number);
  ~PIDMazeSolver();
  std::vector<double> cap_velocities(double u_x, double u_y, double u_z);
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
  phi = normalize_angle(yaw);

  // Log the position and orientation for debugging
  RCLCPP_DEBUG(this->get_logger(),
               "Position: [x: %f, y: %f, z: %f], Orientation (yaw): %f",
               current_position_.x, current_position_.y, current_position_.z,
               phi);
}

PIDMazeSolver::PIDMazeSolver(int scene_number)
    : Node("maze_solver_node"), scene_number_(scene_number) {
  motions = {
      {"forward", {1.0, 1.0, 1.0, 1.0}},            // All wheels forward
      {"backward", {-1.0, -1.0, -1.0, -1.0}},       // All wheels backward
      {"left", {-1.0, 1.0, -1.0, 1.0}},             // Sideways left
      {"right", {1.0, -1.0, 1.0, -1.0}},            // Sideways right
      {"clockwise", {1.0, -1.0, -1.0, 1.0}},        // Clockwise rotation
      {"counterclockwise", {-1.0, 1.0, 1.0, -1.0}}, // Counter-clockwise
      {"stop", {0.0, 0.0, 0.0, 0.0}}                // Stop all wheels
  };
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

  // Based on simulation or real bot
  // Read waypoints from YAML file & update waypoints vector
  // And configure PID controllers
  readWaypointsYAML();

  // Robot parameters
  l = 0.170 / 2; // Half of the wheel base distance
  r = 0.100 / 2; // Radius of the wheels
  w = 0.270 / 2; // Half of track width

  // Initialize the transformation matrix H_4x3 with the correct size
  H = Eigen::MatrixXd(4, 3);                   // Explicitly size the matrix
  H.row(0) << (-l - w) / r, 1.0 / r, -1.0 / r; // Row 0
  H.row(1) << (l + w) / r, 1.0 / r, 1.0 / r;   // Row 1
  H.row(2) << (l + w) / r, 1.0 / r, -1.0 / r;  // Row 2
  H.row(3) << (-l - w) / r, 1.0 / r, 1.0 / r;  // Row 3

  // Compute the pseudo-inverse of H
  H_pseudo_inverse = H.completeOrthogonalDecomposition().pseudoInverse();

  RCLCPP_INFO(this->get_logger(), "Maze Solver Initialized.");

  timer_ = this->create_wall_timer(
      1s, std::bind(&PIDMazeSolver::pid_controller, this), timer_cb_grp_);
}

std::vector<double> PIDMazeSolver::velocity2wheelspeed(double wz, double vx,
                                                       double vy) {
  // Twist vector
  Eigen::Vector3d twist(wz, vx, vy);

  // Compute wheel speeds: u = H * twist
  Eigen::Vector4d u = H * twist;

  // Convert Eigen vector to std::vector<double>
  std::vector<double> wheel_speeds(u.data(), u.data() + u.size());

  return wheel_speeds;
}

std::vector<double> PIDMazeSolver::global2localvelocity(double vx, double vy,
                                                        double avz) {
  // Create input vector
  Eigen::Vector3d velocity(vx, vy, avz);

  // Define the transformation matrix R row-wise
  Eigen::MatrixXd R(3, 3);                      // 3x3 matrix
  R.row(0) << 1, 0, 0;                          // Row 0
  R.row(1) << 0, std::cos(phi), std::sin(phi);  // Row 1
  R.row(2) << 0, -std::sin(phi), std::cos(phi); // Row 2

  // Perform matrix-vector multiplication
  Eigen::Vector3d twist = R * velocity;

  // Convert Eigen::Vector3d to std::vector<double>
  return std::vector<double>{twist(0), twist(1), twist(2)};
}
void PIDMazeSolver::pid_controller() {
  double u_x, u_y, u_z;
  std::vector<double> capped_velocities;
  std::vector<double> wheel_speeds, u;
  double dx, dy, dphi;
  double sp_x, sp_y, sp_phi;
  double error, distance;
  geometry_msgs::msg::Twist twist;
  RCLCPP_INFO(this->get_logger(), "Trajectory started.");

  // Loop through each waypoint
  int index = 0;
  for (const auto &waypoint : waypoints_) {

    dx = waypoint[0];
    dy = waypoint[1];
    dphi = waypoint[2];
    RCLCPP_INFO(this->get_logger(), "WP%u: [%.2f, %.2f, %.2f]", ++index, dx, dy,
                dphi);

    // 1. Change to local waypoints wrt Robot frame
    auto local_vels = global2localvelocity(dphi, dx, dy);
    // 2. Compute the wheel speeds
    auto wheel_speeds =
        velocity2wheelspeed(local_vels[0], local_vels[1], local_vels[2]);
    // Extract wheel speeds (in rad/s) and assign to vector u

    u << wheel_speeds[0], wheel_speeds[1], wheel_speeds[2], wheel_speeds[3];
    // 3. Compute the twist vector
    Eigen::Vector3d twist_v = H_pseudo_inverse * u;

    // Create and publish Twist message
    geometry_msgs::msg::Twist twist;
    twist.linear.x = twist_v(1);
    twist.linear.y = twist_v(2);
    twist.angular.z = twist_v(0);

    sp_x = dx;
    sp_y = dy;
    sp_phi = normalize_angle(dphi);

    rclcpp::Rate rate(int(1 / time_step)); // Control loop frequency

    // 1. Turn only
    error = 0.0;
    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    do {
      if (!rclcpp::ok()) { // Check if ROS is still running
        RCLCPP_WARN(this->get_logger(), "Trajectory Canceled.");
        timer_->cancel();         // Stop the timer
        odom_subscriber_.reset(); // Kill the odometry subscription
        rclcpp::shutdown();
        return;
      }

      // Calculate error for yaw, accounting for wrap-around
      double error_yaw = normalize_angle(sp_phi - phi);

      // PID calculation for Angle
      u_z = pid_z_.compute(sp_phi, phi, true);

      // Use the normalized error for logging and control
      error = error_yaw;
      RCLCPP_DEBUG(this->get_logger(),
                   "phi: %.3f, sp_phi, %.3f, target: %.3f rad", phi, sp_phi,
                   error);
      RCLCPP_DEBUG(this->get_logger(), "Position: %.3f,%.3f,%.3f",
                   current_position_.x, current_position_.y, phi);

      // Prepare and publish the twist message
      capped_velocities = cap_velocities(0.0, 0.0, u_z);
      twist.angular.z = capped_velocities[2];
      cmd_vel_publisher_->publish(twist);
      RCLCPP_DEBUG(this->get_logger(), "Angular vel: %.3f", twist.angular.z);

      rate.sleep();                // Maintain loop frequency
    } while (fabs(error) > 0.007); // Run until error is within tolerance
    pid_z_.reset_();

    // 2. Move
    distance = 0.0;
    do {
      if (!rclcpp::ok()) { // Check if ROS is still running
        RCLCPP_WARN(this->get_logger(), "Trajectory Canceled.");
        timer_->cancel();         // Stop the timer
        odom_subscriber_.reset(); // Kill the odometry subscription
        rclcpp::shutdown();
        return;
      }
      // PID calculation
      u_x = pid_x_.compute(sp_x, current_position_.x);
      u_y = pid_y_.compute(sp_y, current_position_.y);
      u_z = pid_z_.compute(sp_phi, phi, true);

      // Calculate distance to the target
      distance = std::sqrt(std::pow(pid_x_.getError(), 2) +
                           std::pow(pid_y_.getError(), 2));

      // Prepare and publish the twist message
      auto twist_v = global2localvelocity(u_z, u_x, u_y);
      capped_velocities = cap_velocities(twist_v[1], twist_v[2], twist_v[0]);
      twist.linear.x = capped_velocities[0];
      twist.linear.y = capped_velocities[1];
      twist.angular.z = capped_velocities[2];
      cmd_vel_publisher_->publish(twist);
      RCLCPP_DEBUG(this->get_logger(), "Distance to target: %.2f", distance);
      RCLCPP_DEBUG(this->get_logger(), "Vel: %.2f,%.2f,%.2f", u_x, u_y, u_z);
      RCLCPP_DEBUG(this->get_logger(), "Twist: %.2f,%.2f,%.2f",
                   capped_velocities[0], capped_velocities[1],
                   capped_velocities[2]);

      rate.sleep();            // Maintain loop frequency
    } while (distance > 0.01); // Run until distance is within tolerance

    // Now stop the bot
    RCLCPP_INFO(this->get_logger(), "Stopping for 1 seconds");
    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    twist.angular.z = 0.0;
    cmd_vel_publisher_->publish(twist);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Reset PID for new waypoints
    pid_x_.reset_();
    pid_y_.reset_();
    pid_z_.reset_();
  }

  RCLCPP_INFO(this->get_logger(), "Trajectory completed.");
  timer_->cancel();         // Stop the timer
  odom_subscriber_.reset(); // Kill the odometry subscription
  rclcpp::shutdown();
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
    pid_x_ = PID(2.0, 0.05, 0.3, time_step);
    pid_y_ = PID(2.0, 0.05, 0.5, time_step);
    pid_z_ = PID(2.0, 0.01, 0.30, time_step);
    break;

  case 2: // CyberWorld
    RCLCPP_INFO(this->get_logger(), "Welcome to CyberWorld!");
    waypoint_file_name = "waypoints_real.yaml";

    /* https://husarion.com/manuals/rosbot-xl/
    Maximum translational velocity = 0.8 m/s
    Maximum rotational velocity = 180 deg/s (3.14 rad/s)
    */
    max_velocity_ = 0.35;
    max_ang_velocity_ = 1.5;
    pid_x_ = PID(2.0, 0.05, 0.3, time_step);
    pid_y_ = PID(2.0, 0.05, 0.5, time_step);
    pid_z_ = PID(2.0, 0.01, 0.30, time_step);
    break;

  case 3: // Simulation Reverse
    RCLCPP_INFO(this->get_logger(), "Welcome to Simulation Reverse!");
    waypoint_file_name = "reverse_waypoints_sim.yaml";
    max_velocity_ = 0.8;
    max_ang_velocity_ = 3.14;
    pid_x_ = PID(2.0, 0.05, 0.3, time_step);
    pid_y_ = PID(2.0, 0.05, 0.5, time_step);
    pid_z_ = PID(2.0, 0.01, 0.30, time_step);
    break;

  case 4: // CyberWorld Reverse
    RCLCPP_INFO(this->get_logger(), "Welcome to CyberWorld Reverse!");
    waypoint_file_name = "reverse_waypoints_real.yaml";
    max_velocity_ = 0.35;
    max_ang_velocity_ = 1.5;
    pid_x_ = PID(2.0, 0.05, 0.3, time_step);
    pid_y_ = PID(2.0, 0.05, 0.5, time_step);
    pid_z_ = PID(2.0, 0.01, 0.30, time_step);
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

std::vector<double> PIDMazeSolver::cap_velocities(double u_x, double u_y,
                                                  double u_z) {
  // Cap linear velocities
  double linear_velocity_magnitude = std::sqrt(u_x * u_x + u_y * u_y);
  if (linear_velocity_magnitude > max_velocity_) {
    double scale_factor = max_velocity_ / linear_velocity_magnitude;
    u_x *= scale_factor;
    u_y *= scale_factor;
  }

  // Cap angular velocity
  if (std::abs(u_z) > max_ang_velocity_) {
    u_z = (u_z > 0 ? max_ang_velocity_ : -max_ang_velocity_);
  }

  // Return capped velocities
  return {u_x, u_y, u_z};
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
