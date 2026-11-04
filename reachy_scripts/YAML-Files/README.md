# YAML Movement Configurations

This directory contains YAML configuration files that define predefined movement sequences for the humanoid robot Reachy.

Each YAML file specifies joint positions (waypoints) that the robot should move through during the execution of a movement sequence.

The configurations are loaded by the main Python script (reachy_trajectory_smoothing.py), which uses the selected interpolation method to generate smooth transitions between the defined waypoints.

## Features

  - Define movement sequences using joint positions
  - Specify waypoints for the robot's arms
  - Configure movement parameters such as the number of steps and delay between movements
  - Reuse movement sequences with different interpolation methods

## Available Configurations

The directory contains several example movement sequences:

  - 6DK_demo.yaml
  - demo.yaml
  - grab.yaml
  - point.yaml
  - wave.yaml

These configurations can be used to demonstrate and compare the implemented interpolation methods without modifying the main Python code.
