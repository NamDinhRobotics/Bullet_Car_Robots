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
p.setTimeStep(1/120.0)
p.setRealTimeSimulation(1)

# Load environment
planeId = p.loadURDF("plane.urdf")
car = p.loadURDF("racecar/racecar.urdf", basePosition=[0, -0, 0.1])

# Set friction properties
p.changeDynamics(planeId, -1, lateralFriction=0.8)
wheels = [2, 3, 5, 7]
for wheel in wheels:
    p.changeDynamics(car, wheel, lateralFriction=0.8)

# Get wheel joint indices
wheel_joints = {"front_left": 2, "front_right": 3, "rear_left": 5, "rear_right": 7}

# Get wheel positions
wheel_positions = {wheel: np.array(p.getLinkState(car, joint)[0]) for wheel, joint in wheel_joints.items()}

# Calculate wheelbase and wheeltrack
wheelbase = np.linalg.norm(wheel_positions["front_left"] - wheel_positions["rear_left"])
wheeltrack = np.linalg.norm(wheel_positions["front_left"] - wheel_positions["front_right"])

# Disable velocity control for steering wheels
p.setJointMotorControl2(car, 4, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(car, 6, p.VELOCITY_CONTROL, force=0)

# Path Parameters
radius = 3.0  # Radius of the circular path
num_waypoints = 200
look_ahead_dist = 1.0
speed = 10.0

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
def get_lookahead_waypoint(pos, closest_idx):
    for i in range(closest_idx, len(waypoints)):
        if np.linalg.norm(waypoints[i] - np.array([pos[0], pos[1]])) >= look_ahead_dist:
            return waypoints[i]
    return waypoints[-1]

# Steering Calculation
def compute_pure_pursuit_steering(pos, yaw):
    closest_idx = get_closest_waypoint(pos)
    lookahead_wp = get_lookahead_waypoint(pos, closest_idx)
    
    lookahead_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.5])
    lookahead_id = p.createMultiBody(baseVisualShapeIndex=lookahead_sphere, 
                                 basePosition=[lookahead_wp[0], lookahead_wp[1], 0.2])

    # Tự động xóa sau 1 giây
    p.addUserDebugText("", [0, 0, 0], lifeTime=1)  # Tạo một object ảo để trigger update
    p.removeBody(lookahead_id)
    

    # Create ahead ball (visualization)
    p.addUserDebugLine([pos[0], pos[1], 0.1], [lookahead_wp[0], lookahead_wp[1], 0.1], [0, 1, 0], 2, lifeTime=0.1)
    
    dx, dy = lookahead_wp[0] - pos[0], lookahead_wp[1] - pos[1]
    local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
    local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

    if local_y == 0:
        return 0
    radius = (local_x ** 2 + local_y ** 2) / (2 * local_y)
    return math.atan(wheelbase / radius)

# Apply Ackermann steering
def apply_ackermann_control(steering_angle, velocity):
    if abs(steering_angle) < 1e-3:
        R_inner = R_outer = float('inf')
    else:
        R_inner = wheelbase / math.tan(abs(steering_angle))
        R_outer = R_inner + wheeltrack if steering_angle > 0 else R_inner - wheeltrack

    inner_angle = math.atan(wheelbase / R_inner) if R_inner != float('inf') else 0
    outer_angle = math.atan(wheelbase / R_outer) if R_outer != float('inf') else 0

    if steering_angle > 0:
        p.setJointMotorControl2(car, 4, p.POSITION_CONTROL, targetPosition=inner_angle)
        p.setJointMotorControl2(car, 6, p.POSITION_CONTROL, targetPosition=outer_angle)
    else:
        p.setJointMotorControl2(car, 4, p.POSITION_CONTROL, targetPosition=-inner_angle)
        p.setJointMotorControl2(car, 6, p.POSITION_CONTROL, targetPosition=-outer_angle)

    inner_speed = velocity * (R_inner / (R_inner + wheeltrack)) if R_inner != float('inf') else velocity
    outer_speed = velocity * (R_outer / (R_outer + wheeltrack)) if R_outer != float('inf') else velocity

    for wheel in [2, 5]:
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=inner_speed, force=10)
    for wheel in [3, 7]:
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=outer_speed, force=10)

# Simulation Loop
while True:
    pos, ori = p.getBasePositionAndOrientation(car)
    yaw = p.getEulerFromQuaternion(ori)[2]

    steering_angle = compute_pure_pursuit_steering(pos, yaw)
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
    for _ in range(3):
        p.stepSimulation()

