# RViz Joint State Visualization

This directory contains a custom ROS 2 node for visualizing the joint positions of the robot Reachy in the Software RViz.

The Python script "reachy_rviz_show_JointState.py" converts joint positions from radiant to degrees. In the following step publishes the values as text markers using the joint_state_text_array topic.

## Requirements
  - Python3
  - ROS 2 Humble
  - RViz
  - rclpy
  - sensor_msgs
  - visualization_msgs
