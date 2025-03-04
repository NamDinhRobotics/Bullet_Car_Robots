# Ackermann Steering & Pure Pursuit Simulation in PyBullet

This project simulates Ackermann steering and Pure Pursuit path tracking using PyBullet. The simulation features a racecar model that follows a figure-eight trajectory while applying realistic steering dynamics.

## Features
- **Ackermann Steering Geometry** for realistic vehicle turning.
- **Pure Pursuit Path Tracking** for trajectory following.
- **Dynamic Friction Adjustment** for road grip control.
- **Real-Time Visualization** with waypoints, path traces, and camera tracking.

## Requirements
Make sure you have the following dependencies installed:

```bash
pip install pybullet numpy
```

## Usage
Run the Python script to start the simulation:

```bash
python simulation.py
```

## Ackermann Steering
Ackermann steering ensures that the inner and outer wheels of a vehicle follow different turning radii to avoid tire slippage.

### Steering Angle Calculation
The steering angle is given by:

![Ackermann Steering](https://latex.codecogs.com/png.latex?\theta_%7Bin%7D%20=%20tan^{-1}\left(%5Cfrac%7BL%7D%7BR_%7Bin%7D%7D%5Cright))

where:
- \( \theta_{in} \) is the inner wheel steering angle,
- \( L \) is the wheelbase,
- \( R_{in} \) is the inner turning radius.

## Pure Pursuit Algorithm
The Pure Pursuit algorithm is used to calculate the steering angle based on a lookahead waypoint.

### Steering Angle Calculation
The lookahead radius \( R \) is determined by:

![Pure Pursuit](https://latex.codecogs.com/png.latex?R=%5Cfrac%7B%5Ctext%7Blocal_x%7D^2%20+%20%5Ctext%7Blocal_y%7D^2%7D%7B2%5Ctext%7Blocal_y%7D%7D)

The steering angle \( \delta \) is then computed as:

![Steering Angle](https://latex.codecogs.com/png.latex?%5Cdelta%20=%20tan^{-1}%5Cleft(%5Cfrac%7BL%7D%7BR%7D%5Cright))

where:
- \( R \) is the lookahead radius,
- \( L \) is the wheelbase,
- \( \delta \) is the steering angle.

## Simulation Environment
- The vehicle follows a **figure-eight path** defined using waypoints.
- The path and vehicle trajectory are visualized in real-time.
- Adjustable camera tracking enhances visualization.

## Example Output
Simulation showing the car following a figure-eight path with real-time Ackermann steering and Pure Pursuit tracking.

![Simulation Screenshot](simulation_screenshot.png)

## License
This project is open-source under the MIT License. Feel free to modify and experiment!

