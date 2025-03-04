import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Add default data path

# Set gravity and time step
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/50.0)
p.setRealTimeSimulation(1)
# Load environment
planeId = p.loadURDF("plane.urdf")
car = p.loadURDF("racecar/racecar.urdf", basePosition=[0, 0, 0.1])  # Start on the circle

# Disable velocity control for front wheels (steering control)
p.setJointMotorControl2(car, 4, p.VELOCITY_CONTROL, force=0)  # Left front
p.setJointMotorControl2(car, 6, p.VELOCITY_CONTROL, force=0)  # Right front

# Ackermann & Pure Pursuit Parameters
L = 0.5         # Wheelbase
track_width = 0.4  # Distance between left and right wheels
speed = 15       # Base speed
look_ahead_dist = 1.5  # Look-ahead distance for Pure Pursuit
circle_radius = 5  # Radius of the circular track
num_waypoints = 400  # Number of waypoints in the circle

# Generate waypoints for the circle
waypoints = np.array([
    [circle_radius * math.cos(2 * math.pi * i / num_waypoints),
     circle_radius * math.sin(2 * math.pi * i / num_waypoints)]
    for i in range(num_waypoints)
])

# Draw the circular trajectory (blue)
for i in range(len(waypoints) - 1):
    p.addUserDebugLine([waypoints[i][0], waypoints[i][1], 0.1],
                       [waypoints[i + 1][0], waypoints[i + 1][1], 0.1],
                       [0, 0, 1], 2)

# Store previous positions for trace
trace_points = []

def get_closest_waypoint(pos):
    """Finds the closest waypoint to the car's current position."""
    distances = np.linalg.norm(waypoints - np.array([pos[0], pos[1]]), axis=1)
    return np.argmin(distances)

def get_lookahead_waypoint(pos, closest_idx):
    """Finds the waypoint ahead of the vehicle based on look-ahead distance."""
    for i in range(closest_idx, len(waypoints)):
        if np.linalg.norm(waypoints[i] - np.array([pos[0], pos[1]])) >= look_ahead_dist:
            return waypoints[i]
    return waypoints[-1]  # Default to last waypoint if none ahead

def compute_pure_pursuit_steering(pos, yaw):
    """Computes the steering angle using Pure Pursuit."""
    closest_idx = get_closest_waypoint(pos)
    lookahead_wp = get_lookahead_waypoint(pos, closest_idx)

    # Draw lookahead waypoint as a red ball
    p.addUserDebugText("o", [lookahead_wp[0], lookahead_wp[1], 0.1], textColorRGB=[1, 0, 0], textSize=2)

    # Draw a small green line connecting car to lookahead point with transparency
    p.addUserDebugLine([pos[0], pos[1], 0.1], 
                   [lookahead_wp[0], lookahead_wp[1], 0.1], 
                   [0, 1, 0], 2, lifeTime=0.1)

    # Transform lookahead waypoint to vehicle coordinate frame
    dx = lookahead_wp[0] - pos[0]
    dy = lookahead_wp[1] - pos[1]
    local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
    local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

    # Calculate steering using Pure Pursuit formula
    if local_y == 0:
        return 0  # Going straight
    radius = (local_x ** 2 + local_y ** 2) / (2 * local_y)
    steering_angle = math.atan(L / radius)

    return steering_angle

def apply_ackermann_control(steering_angle, velocity):
    """Applies Ackermann steering and velocity control."""
    if abs(steering_angle) < 1e-3:
        R_inner = R_outer = float('inf')  # Go straight
    else:
        R_inner = L / math.tan(abs(steering_angle))
        R_outer = R_inner + track_width if steering_angle > 0 else R_inner - track_width

    # Compute individual wheel angles
    inner_angle = math.atan(L / R_inner) if R_inner != float('inf') else 0
    outer_angle = math.atan(L / R_outer) if R_outer != float('inf') else 0

    # Apply calculated steering angles
    if steering_angle > 0:  # Turning left
        p.setJointMotorControl2(car, 4, p.POSITION_CONTROL, targetPosition=inner_angle)  # Left front
        p.setJointMotorControl2(car, 6, p.POSITION_CONTROL, targetPosition=outer_angle)  # Right front
    else:  # Turning right or straight
        p.setJointMotorControl2(car, 4, p.POSITION_CONTROL, targetPosition=-inner_angle)  # Left front
        p.setJointMotorControl2(car, 6, p.POSITION_CONTROL, targetPosition=-outer_angle)  # Right front

    # Compute wheel speeds
    inner_speed = velocity * (R_inner / (R_inner + track_width)) if R_inner != float('inf') else velocity
    outer_speed = velocity * (R_outer / (R_outer + track_width)) if R_outer != float('inf') else velocity

    # Apply speed to all wheels (4AWD)
    p.setJointMotorControl2(car, 2, p.VELOCITY_CONTROL, targetVelocity=inner_speed, force=10)  # Left rear
    p.setJointMotorControl2(car, 3, p.VELOCITY_CONTROL, targetVelocity=outer_speed, force=10)  # Right rear
    p.setJointMotorControl2(car, 5, p.VELOCITY_CONTROL, targetVelocity=inner_speed, force=10)  # Left front drive
    p.setJointMotorControl2(car, 7, p.VELOCITY_CONTROL, targetVelocity=outer_speed, force=10)  # Right front drive

# Simulation Loop
while True:
    pos, ori = p.getBasePositionAndOrientation(car)
    yaw = p.getEulerFromQuaternion(ori)[2]  # Get yaw (heading)

    # Compute steering angle using Pure Pursuit
    steering_angle = compute_pure_pursuit_steering(pos, yaw)

    # Apply Ackermann steering and velocity
    apply_ackermann_control(steering_angle, speed)

    # Store vehicle tracking point
    trace_points.append((pos[0], pos[1]))

    # Draw trace in yellow
    if len(trace_points) > 1:
        for i in range(len(trace_points) - 1):
            p.addUserDebugLine([trace_points[i][0], trace_points[i][1], 0.05], 
                               [trace_points[i + 1][0], trace_points[i + 1][1], 0.05], 
                               [1, 1, 0], 2, lifeTime=10)

    # Limit number of points in trace to avoid performance issues
    if len(trace_points) > 500:
        trace_points.pop(0)

    # Step simulation
    p.stepSimulation()
    time.sleep(1./50.0)

p.disconnect()

