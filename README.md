# Ackermann Steering & Pure Pursuit Simulation

This project simulates an Ackermann steering model combined with a Pure Pursuit path-following algorithm using PyBullet.
![image](https://github.com/user-attachments/assets/572b1e54-9699-4bf5-944d-34bd2041ad23)

## Features
- Simulates a vehicle with realistic Ackermann steering.
- Uses the Pure Pursuit algorithm to follow a figure-eight trajectory.
- Visualizes waypoints, the vehicle path, and lookahead points.

## Installation
Ensure you have Python installed, then install the required dependencies:

```sh
pip install pybullet numpy
```

## Running the Simulation
Run the following command:

```sh
python simulation.py
```

## Mathematical Background

### Ackermann Steering Model
Ackermann steering ensures that all wheels follow a circular path with different turning radii. The inner and outer wheels turn at different angles to minimize tire slip.

#### Steering Angle Computation
$$\delta_{inner} = \tan^{-1}\left(\frac{L}{R_{inner}}\right)$$ <br />
$$\delta_{outer} = \tan^{-1}\left(\frac{L}{R_{outer}}\right)$$ <br />
where:
- $$L$$ is the wheelbase
- $$R_{inner}$$ and $$R_{outer}$$ are the inner and outer turning radii

#### Turning Radius
$$ R_{inner} = \frac{L}{\tan(|\delta|)} $$
$$ R_{outer} = R_{inner} \pm trackWidth $$

### Pure Pursuit Algorithm
Pure Pursuit calculates the required steering angle to follow a given path based on a look-ahead point.

#### Steering Angle Computation
$$ \gamma = \tan^{-1}\left(\frac{2 L y}{d^2}\right) $$
where:
- $$L$$ is the wheelbase
- $$y$$ is the perpendicular distance to the look-ahead point
- $$d$$ is the Euclidean distance to the look-ahead point

## File Structure
```
.
├── simulation.py   # Python script for PyBullet simulation
├── README.md       # Project documentation
```

## References
- [Ackermann Steering](https://en.wikipedia.org/wiki/Ackermann_steering_geometry)
- [Pure Pursuit](https://www.ri.cmu.edu/pub_files/2009/6/PurePursuit.pdf)

## License
This project is licensed under the MIT License.
