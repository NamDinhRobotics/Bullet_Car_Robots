import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity and time step
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 120.0)
p.setRealTimeSimulation(1)

# Load environment
planeId = p.loadURDF("plane.urdf")
car = p.loadURDF("racecar/racecar.urdf", basePosition=[0, -3, 0.1])

# Set friction properties
p.changeDynamics(planeId, -1, lateralFriction=0.1)
wheels = [2, 3, 5, 7]
for wheel in wheels:
    p.changeDynamics(car, wheel, lateralFriction=0.1)

# Get wheel joint indices
wheel_joints = {"front_left": 2, "front_right": 3, "rear_left": 5, "rear_right": 7}

# Get wheel positions
wheel_positions = {wheel: np.array(p.getLinkState(car, joint)[0]) for wheel, joint in wheel_joints.items()}

# Calculate wheelbase and wheeltrack
wheelbase = np.linalg.norm(wheel_positions["front_left"] - wheel_positions["rear_left"])
wheeltrack = np.linalg.norm(wheel_positions["front_left"] - wheel_positions["front_right"])

# Disable velocity control for steering joints
p.setJointMotorControl2(car, 4, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(car, 6, p.VELOCITY_CONTROL, force=0)

# Path Parameters
radius = 3.0
num_waypoints = 2000
speed = 10.0  # Target speed

# Generate circular path
waypoints = np.array([[radius * math.cos(2 * math.pi * i / num_waypoints),
                       radius * math.sin(2 * math.pi * i / num_waypoints)]
                      for i in range(num_waypoints)])

# Draw circular trajectory
for i in range(len(waypoints) - 1):
    p.addUserDebugLine([waypoints[i][0], waypoints[i][1], 0.1],
                       [waypoints[i + 1][0], waypoints[i + 1][1], 0.1],
                       [0, 0, 1], 2)

# Store vehicle trace points
trace_points = []

# Function to get closest waypoint
def get_closest_waypoint(pos):
    distances = np.linalg.norm(waypoints - np.array([pos[0], pos[1]]), axis=1)
    return np.argmin(distances)

# Function to get look-ahead waypoint
def get_lookahead_waypoint(pos, closest_idx, look_ahead_dist):
    for i in range(closest_idx, len(waypoints)):
        if np.linalg.norm(waypoints[i] - np.array([pos[0], pos[1]])) >= look_ahead_dist:
            return waypoints[i]
    return waypoints[-1]

# Steering Calculation
def compute_pure_pursuit_steering(pos, yaw, look_ahead_dist):
    closest_idx = get_closest_waypoint(pos)
    lookahead_wp = get_lookahead_waypoint(pos, closest_idx, look_ahead_dist)

    # Visualize look-ahead point
    lookahead_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.5])
    lookahead_id = p.createMultiBody(baseVisualShapeIndex=lookahead_sphere,
                                     basePosition=[lookahead_wp[0], lookahead_wp[1], 0.2])
    p.removeBody(lookahead_id)  # Remove immediately after one frame

    p.addUserDebugLine([pos[0], pos[1], 0.1], [lookahead_wp[0], lookahead_wp[1], 0.1], [0, 1, 0], 2, lifeTime=0.1)

    # Corrected local coordinates (y positive to the left)
    dx, dy = lookahead_wp[0] - pos[0], lookahead_wp[1] - pos[1]
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

    if abs(local_x**2 + local_y**2) < 1e-6:  # Avoid division by near-zero
        return 0
    # Standard pure pursuit steering angle
    steering_angle = math.atan(2 * wheelbase * local_y / (local_x**2 + local_y**2))
    return steering_angle

# Apply Ackermann steering
def apply_ackermann_control(steering_angle, velocity):
    if abs(steering_angle) < 1e-3:
        delta_left = delta_right = 0
    else:
        R = wheelbase / math.tan(abs(steering_angle))
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

# Simulation Loop
while True:
    pos, ori = p.getBasePositionAndOrientation(car)
    yaw = p.getEulerFromQuaternion(ori)[2]

    # Calculate current speed
    linear_velocity, _ = p.getBaseVelocity(car)
    current_speed = math.sqrt(linear_velocity[0]**2 + linear_velocity[1]**2)

    # Novel look-ahead distance functions (uncomment one to test)
    look_ahead_dist = max(0.8, 1.0 * current_speed)
    # 1. Exponential
    #look_ahead_dist = 2.0 * (1 - math.exp(-0.5 * current_speed)) + 0.5
    # 2. Quadratic with cap
    # look_ahead_dist = min(0.02 * current_speed**2 + 0.1, 3.0)
    # 3. Speed and curvature adaptive (uses previous steering_angle, initialize first)
    #steering_angle_prev = compute_pure_pursuit_steering(pos, yaw, 0.1)  # Initial estimate
    #look_ahead_dist = max(0.1, 0.2 * current_speed / (1 + abs(steering_angle_prev)))

    # Compute steering and apply control
    steering_angle = compute_pure_pursuit_steering(pos, yaw, look_ahead_dist)
    apply_ackermann_control(steering_angle, speed)

    # Store and draw vehicle trace
    trace_points.append((pos[0], pos[1]))
    if len(trace_points) > 1:
        for i in range(len(trace_points) - 1):
            p.addUserDebugLine([trace_points[i][0], trace_points[i][1], 0.05],
                               [trace_points[i + 1][0], trace_points[i + 1][1], 0.05],
                               [1, 1, 0], 2, lifeTime=10)
    if len(trace_points) > 500:
        trace_points.pop(0)

    # Step simulation
    p.stepSimulation()
    time.sleep(1 / 240.0)  # Adjust for real-time stability