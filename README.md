# Autonomous Vehicle Simulation using PyBullet

## Overview
This project simulates a four-wheel vehicle in PyBullet that follows a figure-eight trajectory using **Ackermann Steering** and **Pure Pursuit Path Tracking**. The goal is to implement realistic vehicle steering and motion control.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install pybullet numpy
```

## Ackermann Steering Model
Ackermann steering geometry is used to control the front wheels, ensuring they follow a circular path without slipping.

### Equations
The turning radius \( R \) is given by:
\[
R = \frac{L}{\tan(\delta)}
\]
where:
- \( L \) is the wheelbase (distance between front and rear axles)
- \( \delta \) is the steering angle

The inner and outer wheel angles are:
\[
\delta_{inner} = \tan^{-1} \left( \frac{L}{R_{inner}} \right)
\]
\[
\delta_{outer} = \tan^{-1} \left( \frac{L}{R_{outer}} \right)
\]

where:
\[
R_{inner} = \frac{L}{\tan(|\delta|)}
\]
\[
R_{outer} = R_{inner} \pm track_{width}
\]

Wheel speeds are adjusted based on the radii:
\[
v_{inner} = v \times \frac{R_{inner}}{R_{inner} + track_{width}}
\]
\[
v_{outer} = v \times \frac{R_{outer}}{R_{outer} + track_{width}}
\]

## Pure Pursuit Path Tracking
Pure Pursuit is a geometric path-tracking algorithm that determines the required steering angle to reach a **look-ahead point** on the trajectory.

### Steering Angle Calculation
Given the vehicle's current position \((x, y)\) and look-ahead point \((x_{LA}, y_{LA})\), we compute:

1. The local coordinates:
\[
x' = (x_{LA} - x) \cos(-\theta) - (y_{LA} - y) \sin(-\theta)
\]
\[
y' = (x_{LA} - x) \sin(-\theta) + (y_{LA} - y) \cos(-\theta)
\]
where \( \theta \) is the vehicle's yaw angle.

2. The look-ahead radius:
\[
R = \frac{x'^2 + y'^2}{2y'}
\]

3. The required steering angle:
\[
\delta = \tan^{-1} \left( \frac{L}{R} \right)
\]

## Simulation
To run the simulation, execute:
```bash
python simulation.py
```

### Features:
- **Real-time trajectory tracking** with debug lines
- **Smooth vehicle steering** using Ackermann geometry
- **Pure Pursuit algorithm** for accurate path following

## Acknowledgments
This implementation is inspired by autonomous vehicle motion control techniques used in robotics and self-driving cars.
