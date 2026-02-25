# UWB Obstacle-Aware SLAM — ObsAwarePlan

A ROS 2 Python package for UWB-based simultaneous localization, anchor mapping and obstacle-aware trajectory planning for PX4 drones.

## Overview

`ObsAwarePlan` fuses **UWB ranging** with an **EKF-SLAM** back-end to estimate both the drone position and unknown UWB anchor locations on the fly. An **Artificial Potential Field (APF)** planner steers the drone along a goal sequence while actively avoiding obstacles whose positions are being discovered during flight.

### Key modules

| File | Description |
|------|-------------|
| `postion_UWB.py` | Main ROS 2 node — EKF prediction/update, WLS anchor init, APF setpoint publishing |
| `EKF_uwb.py` | Extended Kalman Filter for 3-D UWB-SLAM (CA dynamics, acc pseudo-measurement) |
| `WLS_Est.py` | Weighted Least-Squares estimator for bootstrapping new anchor positions |
| `APF_class.py` | Artificial Potential Field planner with direction smoothing |
| `RVIZ_visualizer.py` | RViz marker helpers (trajectory, anchors, direction arrow) |
| `slam_planner_vel.py` | Velocity-setpoint controller node — arms, takes off, follows APF waypoints |
| `utils.py` | Math utilities (rotation matrices, quaternion helpers, ENU↔NED conversions) |

## Dependencies

- ROS 2 (Humble or newer)
- [`px4_msgs`](https://github.com/PX4/px4_msgs)
- `numpy`

## Installation

```bash
# inside your ROS 2 workspace src/
git clone https://github.com/davidenascivera/UWB-Obs-Aware-SLAM.git ObsAwarePlan
cd ../..
colcon build --packages-select ObsAwarePlan
source install/setup.bash
```

## Running

```bash
# EKF-SLAM + planner node
ros2 run ObsAwarePlan position_ekf

# Velocity setpoint / offboard controller
ros2 run ObsAwarePlan slam_planner

# Monte Carlo run with custom seed and start position
ros2 run ObsAwarePlan position_ekf --ros-args -p mc_seed:=42 -p x0:=1.0 -p y0:=2.0
```

## ROS 2 Topics

| Topic | Type | Direction |
|-------|------|-----------|
| `/fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | publish |
| `/fmu/in/offboard_control_mode` | `px4_msgs/OffboardControlMode` | publish |
| `/fmu/in/vehicle_command` | `px4_msgs/VehicleCommand` | publish |
| `/fmu/out/sensor_combined` | `px4_msgs/SensorCombined` | subscribe |
| `/model/x500_0/odometry` | `nav_msgs/Odometry` | subscribe (ground truth) |
| `/uwb_ekf/odometry` | `nav_msgs/Odometry` | publish (EKF estimate) |
| `/apf/waypoint` | `px4_msgs/TrajectorySetpoint` | publish / subscribe |

## License

MIT
