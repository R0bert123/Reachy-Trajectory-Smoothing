# Setup and Usage Guide

This document explains the requirements and configuration steps for the trajectory smoothing project developed for the humanoid robot Reachy 2023.

## 1. Requirements

The project was developed and tested using the following environment:

  - Operating System: Ubuntu 22.04
  - Programming Language: Python 3.10
  - Robotics Framework: ROS 2 Humble
  - Robot: Reachy 2023

### Software Dependencies

Relevant dependencies were in this project:

  - NumPy
  - SciPy
  - ROS 2 Python Client Library (`rclpy`)
  - ROS 2 Control (`ros2_control, ros2_controllers`)
  - RViz
  - IKPy
  - ReachySDK

## 2. Project Structure

The repository contains the following main components:

  - `reachy_description/`: Contains the URDF models, ROS 2 configuration files, and launch files.
  - `reachy_scripts/YAML-Files/`: Contains predefined robot movement sequences and waypoints.
  - `reachy_scripts/RVIZ_Nodes/`: Contains additional ROS 2 visualization scripts.
  - `reachy_trajectory_smoothing.py`: Main Python script implementing the six interpolation methods.

## 3. ROS 2 Setup

The robot description package must be available in a configured ROS 2 environment.

Initialize ROS 2 Humble in a terminal:

`source /opt/ros/humble/setup.bash`

### Building the Robot Description Package

The `reachy_description` directory contains a ROS 2 package with its own package.xml and CMakeLists.txt.

Place the package in the src directory of a ROS 2 workspace.

For example, create a workspace:

`mkdir -p ~/reachy_ws/src`

After placing the package inside `~/reachy_ws/src/`, build the workspace:

`cd ~/reachy_ws`

`colcon build --symlink-install`

After a successful build, source the workspace:

`source ~/reachy_ws/install/setup.bash`

Depending on the installed ROS 2 packages, additional dependencies may be required.


## 4. URDF Configuration

The reachy_description directory contains the robot description files required for visualization and robot control.

The URDF files define the robot's kinematic structure, including its joints and links.

For this project, robot descriptions were adapted to support the left and right arms of Reachy.

The directory also contains configuration files for ROS 2 Control and launch files for initializing the robot environment.

### Important Notes

  - Ensure that the correct URDF files are loaded.
  - Verfiy the joint names and joint limits.
  - Check that the ROS 2 controllers are configured correctly.
  - Ensure that the required TF frames are availabe when using RViz.

## 5. YAML Movement Files

The `reachy_scripts/YAML-Files/` directory contains predefined movement configurations.

Each YAML file defines joint positions that the robot should pass through during a movement sequence.

The main Python script loads these configurations and generates interpolated movements between the defined waypoints.

### Available Movement Configurations

Depending on the configuration, YAML files can include:

  - Joint positions
  - Number of interpolation steps
  - Delay between steps
  - Movement speed parameters
  - Gripper configurations
  - Optional checkpoints

The same movement configuration can be evaluated using different interpolation methods.

For further information, see the `README.md` in the YAML-Files directory.

## 6. Running the Main Program

The main implementation is located in: `reachy_trajectory_smoothing.py`

Before executing the program:
  1. Initialize the ROS 2 environment.
  2. Load the required robot description and controllers.
  3. Verify that the robot is connected or that the simulation is running.
  4. Select the required YAML movement configuration.
  5. Select the desired interpolation method.

The program implements six interpolation methods:
  - Linear Interpolation (No Interpolator)
  - Cubic Polynomial Interpolation
  - Quintic Polynomial Interpolation
  - Cubic B-Spline Interpolation
  - Cubic Bézier Interpolation
  - Minimum-Jerk Interpolation

Execution command: To be documented according to the final script's actual arguments and configuration.

## 7. RViz Visualization

The `reachy_scripts/RVIZ_Nodes/` directory contains an additional ROS 2 node for visualizing joint positions in RViz.

The node subscribes to the `/joint_states` topic and converts joint positions from radians to degrees.

The calculated angles are published as text markers using the `/joint_state_text_array` topic.

To display these values:
  1. Start the ROS 2 environment.
  2. Launch RViz with the Reachy robot description.
  3. Start the joint state visualization node.
  4. Add a MarkerArray display in RViz.
  5. Select the `/joint_state_text_array` topic.

The required TF frames must be available for the text markers to appear at the correct positions.

## 8. Safety Nodes

When executing movement sequences on the physical Reachy robot, the following precautions should be considered:

  - Verify that all target joint positions remain within the robot's joint limits.
  - Ensure that the robot has sufficient space to perform the movement.
  - Check the movement configuration before execution.
  - Avoid unexpected movements when switching between interpolation methods.
  - Ensure that the robot can be stopped safely if unexpected behavior occurs.

The project was developed in a research environment and should not be considered a production-ready robot control system.

