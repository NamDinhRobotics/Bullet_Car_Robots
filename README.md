# README: Ackermann Steering and Pure Pursuit Path Tracking

## Introduction
This project simulates a race car in PyBullet using **Ackermann Steering** and **Pure Pursuit Path Tracking** to follow a figure-eight trajectory. The simulation includes real-time control of the vehicle's steering and velocity while drawing its path.

## Requirements
Ensure you have the necessary Python packages installed:
```bash
pip install pybullet numpy
```

## Ackermann Steering
Ackermann steering geometry ensures that all four wheels of a vehicle follow circular paths with a common center. The steering angles for the inner and outer wheels are calculated using:

### Formulas:
- **Turning radius of the inner and outer wheels:**
  \[
  R_{\text{inner}} = \frac{L}{\tan(|\delta|)}
  \]
  \[
  R_{\text{outer}} = R_{\text{inner}} \pm \text{track width}
  \]
  where:
  - \(L\) is the wheelbase (distance between front and rear axle)
  - \(\delta\) is the steering angle
  - \(\text{track width}\) is the distance between left and right wheels

- **Individual steering angles for front wheels:**
  \[
  \delta_{\text{inner}} = \tan^{-1}\left(\frac{L}{R_{\text{inner}}}\right)
  \]
  \[
  \delta_{\text{outer}} = \tan^{-1}\left(\frac{L}{R_{\text{outer}}}\right)
  \]

- **Wheel velocities (to maintain correct turning ratio):**
  \[
  v_{\text{inner}} = v \times \frac{R_{\text{inner}}}{R_{\text{inner}} + \text{track width}}
  \]
  \[
  v_{\text{outer}} = v \times \frac{R_{\text{outer}}}{R_{\text{outer}} + \text{track width}}
  \]

## Pure Pursuit Algorithm
The **Pure Pursuit** controller determines the steering angle required to follow a given path by choosing a look-ahead point and computing the turning radius needed to reach it.

### Steps:
1. Find the closest waypoint to the vehicle's position.
2. Locate a **look-ahead point** at a specified distance ahead of the vehicle.
3. Compute the radius \(R\) of the circle passing through the vehicle and look-ahead point:
   \[
   R = \frac{x^2 + y^2}{2y}
   \]
   where \((x, y)\) is the look-ahead point in the vehicle's coordinate frame.
4. Calculate the required steering angle:
   \[
   \delta = \tan^{-1}\left(\frac{L}{R}\right)
   \]

## Implementation in PyBullet
- The **car's steering** is controlled using PyBullet’s `setJointMotorControl2()` function.
- The **figure-eight trajectory** is generated using a Lissajous curve.
- The **vehicle follows the trajectory** by computing steering angles dynamically and adjusting wheel velocities.

## Running the Simulation
Execute the script with:
```bash
python simulation.py
```

## Visualization
- **Waypoints** are drawn in blue.
- **Look-ahead points** are highlighted in red.
- **The vehicle’s path trace** is shown in yellow.

## Future Improvements
- Implement PID control for smoother tracking.
- Add obstacles and collision avoidance.
- Improve camera view to track the vehicle dynamically.

## References
- Ackermann Steering: https://en.wikipedia.org/wiki/Ackermann_steering_geometry
- Pure Pursuit: https://www.ri.cmu.edu/pub_files/2000/0/1995_BAE_Following.pdf

