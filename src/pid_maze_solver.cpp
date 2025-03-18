#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <vector>

class PID {
public:
  PID() : kp_(0.0), ki_(0.0), kd_(0.0), dt_(0.0) {}

  PID(double kp, double ki, double kd, double dt)
      : kp_(kp), ki_(ki), kd_(kd), dt_(dt), integral_(0.0), prev_error_(0.0) {}

  double compute(double setpoint, double measurement) {
    double error = setpoint - measurement;
    integral_ += error * dt_;
    double derivative = (error - prev_error_) / dt_;
    double output = kp_ * error + ki_ * integral_ + kd_ * derivative;

    prev_error_ = error;
    return output;
  }

  double getError() { return prev_error_; }

private:
  double kp_;
  double ki_;
  double kd_;
  double dt_;
  double integral_;
  double prev_error_;
};

using namespace std::chrono_literals;

class PIDMazeSolver : public rclcpp::Node {
private:
  int scene_number_;
  double max_velocity_;
  double max_ang_velocity_;
  std::vector<std::vector<double>> waypoints_; //{dx, dy, dphi}

  void SelectWaypoints();
  void pid_controller();

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

public:
  PIDMazeSolver(int scene_number);
  ~PIDMazeSolver();
  std::vector<double> cap_velocities(double u_x, double u_y, double u_z);
};

PIDMazeSolver::~PIDMazeSolver() {
  RCLCPP_INFO(this->get_logger(), "Turn Controller Terminated.");
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

  SelectWaypoints();
  /* https://husarion.com/manuals/rosbot-xl/
  Maximum translational velocity = 0.8 m/s
  Maximum rotational velocity = 180 deg/s (3.14 rad/s)
  */
  max_velocity_ = 0.8;
  max_ang_velocity_ = 3.14;
  pid_x_ = PID(1.0, 0.01, 0.20, time_step);
  pid_y_ = PID(1.0, 0.01, 0.20, time_step);
  pid_z_ = PID(2.0, 0.01, 0.30, time_step);
  RCLCPP_INFO(this->get_logger(), "Maze Solver Initialized.");

  timer_ = this->create_wall_timer(
      1s, std::bind(&PIDMazeSolver::pid_controller, this), timer_cb_grp_);
}

void PIDMazeSolver::pid_controller() {
  double u_x, u_y, u_z;
  std::vector<double> capped_velocities;
  double dx, dy, dphi;
  double sp_x, sp_y, sp_phi;
  double error, distance;
  geometry_msgs::msg::Twist twist;
  RCLCPP_INFO(this->get_logger(), "Trajectory started.");

  // Loop through each waypoint
  for (const auto &waypoint : waypoints_) {

    dx = waypoint[0];
    dy = waypoint[1];
    dphi = waypoint[2];
    RCLCPP_INFO(this->get_logger(), "WP: [%.2f, %.2f, %.2f]", dx, dy, dphi);

    sp_x = current_position_.x + dx;
    sp_y = current_position_.y + dy;
    sp_phi = phi + dphi;

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
      // PID calculation for Angle
      u_z = pid_z_.compute(sp_phi, phi);

      // Calculate error to the target
      error = pid_z_.getError();
      RCLCPP_DEBUG(this->get_logger(), "phi: %.3f, angle to target: %.3f rads",
                   phi, error);
      RCLCPP_DEBUG(this->get_logger(), "Position: %.3f,%.3f,%.3f",
                   current_position_.x, current_position_.y, phi);

      // Prepare and publish the twist message
      capped_velocities = cap_velocities(0.0, 0.0, u_z);
      twist.angular.z = capped_velocities[2];
      cmd_vel_publisher_->publish(twist);
      RCLCPP_DEBUG(this->get_logger(), "Angular vel: %.3f", twist.angular.z);

      rate.sleep();                // Maintain loop frequency
    } while (fabs(error) > 0.007); // Run until error is within tolerance

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
      u_z = pid_z_.compute(sp_phi, phi);

      // Calculate distance to the target
      distance = std::sqrt(std::pow(pid_x_.getError(), 2) +
                           std::pow(pid_y_.getError(), 2));
      RCLCPP_DEBUG(this->get_logger(), "Distance to target: %.2f", distance);
      RCLCPP_DEBUG(this->get_logger(), "Position: %.2f,%.2f,%.2f", u_x, u_y,
                   u_z);

      // Prepare and publish the twist message
      capped_velocities = cap_velocities(u_x, u_y, u_z);
      twist.linear.x = capped_velocities[0];
      twist.linear.y = capped_velocities[1];
      twist.angular.z = capped_velocities[2];
      cmd_vel_publisher_->publish(twist);

      rate.sleep();            // Maintain loop frequency
    } while (distance > 0.01); // Run until distance is within tolerance

    // Now stop the bot
    RCLCPP_INFO(this->get_logger(), "Stopping for 1 seconds");
    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    twist.angular.z = 0.0;
    cmd_vel_publisher_->publish(twist);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  RCLCPP_INFO(this->get_logger(), "Trajectory completed.");
  timer_->cancel();         // Stop the timer
  odom_subscriber_.reset(); // Kill the odometry subscription
  rclcpp::shutdown();
}

void PIDMazeSolver::SelectWaypoints() {
  switch (scene_number_) {
  case 1: // Simulation
    // Assign waypoints for Simulation
    RCLCPP_INFO(this->get_logger(), "Welcome to Simulation!");
    // Waypoints: {dx,dy,dphi}
    waypoints_ = {
        {0.27, 0.00, +0.0},      // w1
        {0.23, -0.074, -0.7854}, // w2
        {0.0, 0.0, -0.7854},     // w3
                                 // {0.0, 0.0, -1.0472}      // w4
    };
    break;

  case 2: // CyberWorld
    // Assign waypoints for CyberWorld
    RCLCPP_INFO(this->get_logger(), "Welcome to CyberWorld!");
    waypoints_ = {
        {0.0, 1.0, -1.0},      // w1
        {0.0, 1.0, 1.0},       // w2
        {0.0, 1.0, 1.0},       // w3
        {-1.5708, 1.0, -1.0},  // w4
        {-1.5708, -1.0, -1.0}, // w5
        {0.0, -1.0, 1.0},      // w6
        {0.0, -1.0, 1.0},      // w7
        {0.0, -1.0, -1.0},     // w8
        {0.0, 0.0, 0.0}        // Stop
    };
    break;

  default:
    RCLCPP_ERROR(this->get_logger(), "Invalid Scene Number: %d", scene_number_);
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
  if (scene_number != 1 && scene_number != 2) {
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
