# Ackermann Steering and Pure Pursuit Simulation

This project demonstrates **Ackermann steering** and **Pure Pursuit path tracking** using **PyBullet** for a simulated racecar. The vehicle follows a figure-eight trajectory while dynamically adjusting steering based on Pure Pursuit control.

## Features
- Ackermann steering model for realistic turning.
- Pure Pursuit algorithm for path tracking.
- 4-wheel independent velocity control.
- Figure-eight trajectory visualization.
- Real-time simulation using PyBullet.

## Installation
```bash
pip install pybullet numpy
```

## Run the Simulation
```bash
python simulation.py
```

## Equations

### 1. Ackermann Steering Geometry
The turning radius **R** for the inner and outer wheels is determined as:

![Ackermann Formula](https://latex.codecogs.com/png.latex?\dpi{150}\color{White}R_{inner} = \frac{L}{\tan(|\delta|)},\quad R_{outer} = R_{inner} \pm W)

where:
- \(L\) is the wheelbase.
- \(W\) is the track width.
- \(\delta\) is the steering angle.

The wheel angles are calculated as:

![Wheel Angle Formula](https://latex.codecogs.com/png.latex?\dpi{150}\color{White}\theta_{inner} = \tan^{-1}\left(\frac{L}{R_{inner}}\right),\quad \theta_{outer} = \tan^{-1}\left(\frac{L}{R_{outer}}\right))

### 2. Pure Pursuit Steering Control
The lookahead point is chosen based on the **lookahead distance (L_d)**. The required steering angle is computed using:

![Pure Pursuit Formula](https://latex.codecogs.com/png.latex?\dpi{150}\color{White}\delta = \tan^{-1}\left(\frac{2L y}{L_d^2}\right))

where:
- \(y\) is the perpendicular distance to the lookahead point.
- \(L_d\) is the lookahead distance.

## Example Screenshot
![Simulation](simulation_screenshot.png)

## File Structure
```
├── simulation.py            # Main simulation script
├── README.md                # Documentation
├── simulation_screenshot.png # Placeholder image
```

## License
This project is open-source under the MIT License.

