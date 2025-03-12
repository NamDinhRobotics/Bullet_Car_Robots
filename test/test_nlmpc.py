import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import casadi as ca

# ### Step 1: Set Up PyBullet Simulation

# Connect to PyBullet
physicsClient = p.connect(p.GUI)  # Use p.DIRECT for non-GUI mode
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity and time step
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 120.0)
p.setRealTimeSimulation(1)  # Real-time simulation

# Load environment and car
planeId = p.loadURDF("plane.urdf")
car = p.loadURDF("racecar/racecar.urdf", basePosition=[0, -3, 0.1])

# Set friction properties
p.changeDynamics(planeId, -1, lateralFriction=0.1)
wheels = [2, 3, 5, 7]  # Wheel link indices
for wheel in wheels:
    p.changeDynamics(car, wheel, lateralFriction=0.1)

# Disable velocity control for steering joints
p.setJointMotorControl2(car, 4, p.VELOCITY_CONTROL, force=0)  # Left steering
p.setJointMotorControl2(car, 6, p.VELOCITY_CONTROL, force=0)  # Right steering

# ### Step 2: Define Vehicle Parameters

# Get wheel positions
wheel_positions = {
    "front_left": np.array(p.getLinkState(car, 2)[0]),
    "front_right": np.array(p.getLinkState(car, 3)[0]),
    "rear_left": np.array(p.getLinkState(car, 5)[0]),
    "rear_right": np.array(p.getLinkState(car, 7)[0])
}

# Calculate wheelbase and wheeltrack
wheelbase = np.linalg.norm(wheel_positions["front_left"] - wheel_positions["rear_left"])
wheeltrack = np.linalg.norm(wheel_positions["front_left"] - wheel_positions["front_right"])

# ### Step 3: Generate Reference Path

# Path Parameters
radius = 3.0  # Radius of circular path
num_waypoints = 2000  # Number of waypoints

# Generate circular path
waypoints = np.array([[radius * math.cos(2 * math.pi * i / num_waypoints),
                       radius * math.sin(2 * math.pi * i / num_waypoints)]
                      for i in range(num_waypoints)])

# Draw circular trajectory in PyBullet
for i in range(len(waypoints) - 1):
    p.addUserDebugLine([waypoints[i][0], waypoints[i][1], 0.1],
                       [waypoints[i + 1][0], waypoints[i + 1][1], 0.1],
                       [0, 0, 1], 2)

# ### Step 4: Define Helper Functions

def get_closest_waypoint(pos):
    """Find the index of the closest waypoint to the car's position."""
    distances = np.linalg.norm(waypoints - np.array([pos[0], pos[1]]), axis=1)
    return np.argmin(distances)

def compute_reference_orientations(waypoints, ref_indices):
    """Compute reference orientations (yaw angles) for the given waypoints."""
    theta_ref = []
    for i in ref_indices:
        idx = i % len(waypoints)
        next_idx = (idx + 1) % len(waypoints)
        dx = waypoints[next_idx][0] - waypoints[idx][0]
        dy = waypoints[next_idx][1] - waypoints[idx][1]
        theta_ref.append(math.atan2(dy, dx))
    return theta_ref

# ### Step 5: Set Up NMPC with CasADi

# NMPC Parameters
N = 10  # Prediction horizon
dt = 1 / 120.0  # Time step matching PyBullet
L = wheelbase  # Wheelbase of the car

# Optimization object
opti = ca.Opti()

# Variables
X = opti.variable(3, N + 1)  # State: [x, y, theta]
U = opti.variable(2, N)      # Control: [delta (steering angle), v (velocity)]
x0 = opti.parameter(3)       # Initial state parameter
p_ref = opti.parameter(2, N) # Reference positions [x_ref, y_ref]
theta_ref = opti.parameter(N) # Reference orientations

# Cost function
cost = 0
w_theta = 1.0  # Weight for orientation error
w_u = 0.01     # Weight for control effort

for k in range(1, N + 1):
    cost += ca.sumsqr(X[0:2, k] - p_ref[:, k-1])  # Position error
    cost += w_theta * (X[2, k] - theta_ref[k-1])**2  # Orientation error
for k in range(N):
    cost += w_u * ca.sumsqr(U[:, k])  # Control effort

opti.minimize(cost)

# Dynamics constraints (kinematic bicycle model)
def f(x, u):
    v = u[1]
    delta = u[0]
    theta = x[2]
    return ca.vertcat(
        x[0] + dt * v * ca.cos(theta),
        x[1] + dt * v * ca.sin(theta),
        x[2] + dt * (v / L) * ca.tan(delta)
    )

for k in range(N):
    opti.subject_to(X[:, k+1] == f(X[:, k], U[:, k]))

# Initial state constraint
opti.subject_to(X[:, 0] == x0)

# Input constraints
opti.subject_to(opti.bounded(-0.5, U[0, :], 0.5))  # Steering angle limits (radians)
opti.subject_to(opti.bounded(0, U[1, :], 15))      # Velocity limits (m/s)

# Solver options
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100}
opti.solver('ipopt', opts)

# ### Step 6: Implement Control Application

def apply_ackermann_control(steering_angle, velocity):
    """Apply Ackermann steering and velocity control to the racecar."""
    if abs(steering_angle) < 1e-3:  # Avoid division by zero
        delta_left = delta_right = 0
    else:
        R = wheelbase / math.tan(abs(steering_angle))  # Turning radius
        if steering_angle > 0:  # Left turn
            delta_left = math.atan(wheelbase / (R - wheeltrack/2))
            delta_right = math.atan(wheelbase / (R + wheeltrack/2))
        else:  # Right turn
            delta_left = -math.atan(wheelbase / (R + wheeltrack/2))
            delta_right = -math.atan(wheelbase / (R - wheeltrack/2))

    # Set steering angles
    p.setJointMotorControl2(car, 4, p.POSITION_CONTROL, targetPosition=delta_left)
    p.setJointMotorControl2(car, 6, p.POSITION_CONTROL, targetPosition=delta_right)

    # Set uniform wheel velocities
    for wheel in wheels:
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=velocity, force=10)

# ### Step 7: Simulation Loop with NMPC

# Store vehicle trace points for visualization
trace_points = []

# Main simulation loop
while True:
    # Get current state
    pos, ori = p.getBasePositionAndOrientation(car)
    yaw = p.getEulerFromQuaternion(ori)[2]
    current_state = [pos[0], pos[1], yaw]

    # Find closest waypoint
    closest_idx = get_closest_waypoint(current_state)

    # Generate local reference path
    ref_indices = [(closest_idx + k) % num_waypoints for k in range(1, N + 1)]
    p_ref_val = waypoints[ref_indices].T  # [2, N]

    # Compute reference orientations
    theta_ref_val = compute_reference_orientations(waypoints, ref_indices)

    # Set NMPC parameters
    opti.set_value(x0, current_state)
    opti.set_value(p_ref, p_ref_val)
    opti.set_value(theta_ref, theta_ref_val)

    # Solve NMPC
    try:
        sol = opti.solve()
        u_opt = sol.value(U[:, 0])  # First control input
        steering_angle = u_opt[0]
        velocity = u_opt[1]
    except Exception as e:
        print(f"NMPC solver failed: {e}")
        steering_angle = 0.0
        velocity = 0.0  # Emergency stop

    # Apply control
    apply_ackermann_control(steering_angle, velocity)

    # Store and draw vehicle trace
    trace_points.append((pos[0], pos[1]))
    if len(trace_points) > 1:
        for i in range(len(trace_points) - 1):
            p.addUserDebugLine([trace_points[i][0], trace_points[i][1], 0.05],
                               [trace_points[i + 1][0], trace_points[i + 1][1], 0.05],
                               [1, 1, 0], 2, lifeTime=10)
    if len(trace_points) > 500:
        trace_points.pop(0)  # Limit trace length

    # Step simulation
    p.stepSimulation()
    time.sleep(1 / 240.0)  # Control frequency higher than simulation step