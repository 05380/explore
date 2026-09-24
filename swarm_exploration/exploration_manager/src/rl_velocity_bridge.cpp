#include <algorithm>
#include <cmath>
#include <cstdint>

#include <geometry_msgs/TwistStamped.h>
#include <exploration_manager/RLTarget.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <ros/ros.h>

namespace {

double clamp(double value, double low, double high) {
  return std::max(low, std::min(value, high));
}

double yawFromQuaternion(const geometry_msgs::Quaternion& q) {
  const double siny = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny, cosy);
}

class RLVelocityBridge {
public:
  RLVelocityBridge() : nh_(), pnh_("~") {
    pnh_.param("control_rate", control_rate_, 20.0);
    pnh_.param("command_timeout", command_timeout_, 0.30);
    pnh_.param("lookahead_time", lookahead_time_, 0.05);
    pnh_.param("normalized_action", normalized_action_, true);
    pnh_.param("require_active_target", require_active_target_, true);
    pnh_.param("max_forward_speed", max_forward_speed_, 2.0);
    pnh_.param("max_backward_speed", max_backward_speed_, 0.5);
    pnh_.param("max_lateral_speed", max_lateral_speed_, 0.8);
    pnh_.param("max_vertical_speed", max_vertical_speed_, 0.5);
    pnh_.param("max_speed_norm", max_speed_norm_, 2.0);
    pnh_.param("max_yaw_rate", max_yaw_rate_, 1.0);
    pnh_.param("min_x", min_x_, -74.5);
    pnh_.param("max_x", max_x_, 74.5);
    pnh_.param("min_y", min_y_, -74.5);
    pnh_.param("max_y", max_y_, 74.5);
    pnh_.param("min_z", min_z_, 0.5);
    pnh_.param("max_z", max_z_, 4.5);

    control_rate_ = std::max(1.0, control_rate_);
    command_timeout_ = std::max(0.05, command_timeout_);
    lookahead_time_ = std::max(0.01, lookahead_time_);
    max_speed_norm_ = std::max(0.0, max_speed_norm_);

    cmd_sub_ = nh_.subscribe(
        "/rl_navigation/cmd_vel_body", 1, &RLVelocityBridge::commandCallback, this,
        ros::TransportHints().tcpNoDelay());
    odom_sub_ = nh_.subscribe("/odom_world", 10, &RLVelocityBridge::odometryCallback, this,
        ros::TransportHints().tcpNoDelay());
    target_sub_ = nh_.subscribe("/rl_navigation/target", 1,
        &RLVelocityBridge::targetCallback, this, ros::TransportHints().tcpNoDelay());
    position_cmd_pub_ =
        nh_.advertise<quadrotor_msgs::PositionCommand>("/rl_navigation/position_cmd", 10);
    timer_ = nh_.createTimer(ros::Duration(1.0 / control_rate_), &RLVelocityBridge::timerCallback,
        this);
  }

private:
  void commandCallback(const geometry_msgs::TwistStampedConstPtr& msg) {
    command_ = *msg;
    last_command_time_ = ros::Time::now();
    have_command_ = true;
  }

  void odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
    odometry_ = *msg;
    have_odometry_ = true;
  }

  void targetCallback(const exploration_manager::RLTargetConstPtr& msg) {
    target_active_ = msg->active;
    if (!target_active_) have_command_ = false;
  }

  double scaleSigned(double action, double positive_limit, double negative_limit) const {
    action = clamp(action, -1.0, 1.0);
    return action >= 0.0 ? action * positive_limit : action * negative_limit;
  }

  void timerCallback(const ros::TimerEvent&) {
    if (!have_odometry_) return;

    const bool inactive = require_active_target_ && !target_active_;
    const bool stale = inactive || !have_command_ ||
        (ros::Time::now() - last_command_time_).toSec() > command_timeout_;
    double vx_body = 0.0;
    double vy_body = 0.0;
    double vz = 0.0;
    double yaw_rate = 0.0;

    if (!stale) {
      if (normalized_action_) {
        vx_body = scaleSigned(command_.twist.linear.x, max_forward_speed_, max_backward_speed_);
        vy_body = clamp(command_.twist.linear.y, -1.0, 1.0) * max_lateral_speed_;
        vz = clamp(command_.twist.linear.z, -1.0, 1.0) * max_vertical_speed_;
        yaw_rate = clamp(command_.twist.angular.z, -1.0, 1.0) * max_yaw_rate_;
      } else {
        vx_body = clamp(command_.twist.linear.x, -max_backward_speed_, max_forward_speed_);
        vy_body = clamp(command_.twist.linear.y, -max_lateral_speed_, max_lateral_speed_);
        vz = clamp(command_.twist.linear.z, -max_vertical_speed_, max_vertical_speed_);
        yaw_rate = clamp(command_.twist.angular.z, -max_yaw_rate_, max_yaw_rate_);
      }
    } else {
      if (inactive)
        ROS_WARN_THROTTLE(1.0, "No active RL target: publishing hover command");
      else
        ROS_WARN_THROTTLE(1.0, "RL action timeout: publishing hover command");
    }

    const double speed_norm = std::sqrt(
        vx_body * vx_body + vy_body * vy_body + vz * vz);
    if (max_speed_norm_ > 0.0 && speed_norm > max_speed_norm_) {
      const double scale = max_speed_norm_ / speed_norm;
      vx_body *= scale;
      vy_body *= scale;
      vz *= scale;
    }

    const double yaw = yawFromQuaternion(odometry_.pose.pose.orientation);
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    const double vx_world = c * vx_body - s * vy_body;
    const double vy_world = s * vx_body + c * vy_body;

    quadrotor_msgs::PositionCommand output;
    output.header.stamp = ros::Time::now();
    output.header.frame_id = "world";
    output.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
    output.trajectory_id = command_id_++;
    output.kx = { 5.7, 5.7, 6.2 };
    output.kv = { 3.4, 3.4, 4.0 };

    output.position.x = clamp(
        odometry_.pose.pose.position.x + vx_world * lookahead_time_, min_x_, max_x_);
    output.position.y = clamp(
        odometry_.pose.pose.position.y + vy_world * lookahead_time_, min_y_, max_y_);
    output.position.z =
        clamp(odometry_.pose.pose.position.z + vz * lookahead_time_, min_z_, max_z_);
    output.velocity.x = vx_world;
    output.velocity.y = vy_world;
    output.velocity.z = vz;
    output.acceleration.x = 0.0;
    output.acceleration.y = 0.0;
    output.acceleration.z = 0.0;
    output.yaw = yaw + yaw_rate * lookahead_time_;
    output.yaw_dot = yaw_rate;
    position_cmd_pub_.publish(output);
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber cmd_sub_, odom_sub_, target_sub_;
  ros::Publisher position_cmd_pub_;
  ros::Timer timer_;

  geometry_msgs::TwistStamped command_;
  nav_msgs::Odometry odometry_;
  ros::Time last_command_time_;
  bool have_command_ = false;
  bool have_odometry_ = false;
  bool require_active_target_ = true;
  bool target_active_ = false;
  uint32_t command_id_ = 1;

  bool normalized_action_;
  double control_rate_, command_timeout_, lookahead_time_;
  double max_forward_speed_, max_backward_speed_, max_lateral_speed_, max_vertical_speed_;
  double max_speed_norm_;
  double max_yaw_rate_;
  double min_x_, max_x_, min_y_, max_y_, min_z_, max_z_;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "rl_velocity_bridge");
  RLVelocityBridge bridge;
  ros::spin();
  return 0;
}
