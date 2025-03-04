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
p.setTimeStep(1/120.0)  # Faster simulation
p.setRealTimeSimulation(1)

# Load environment
planeId = p.loadURDF("plane.urdf")
car = p.loadURDF("racecar/racecar.urdf", basePosition=[0, 0, 0.1])

# Đặt hệ số ma sát của sàn (giảm để làm trơn hơn, tăng để có độ bám tốt hơn)
p.changeDynamics(planeId, -1, lateralFriction=0.5)  # Giá trị thấp làm cho sàn trơn hơn

# Đặt hệ số ma sát cho bánh xe để kiểm soát độ bám đường
wheels = [2, 3, 5, 7]  # Các joint ID của bánh xe
for wheel in wheels:
    p.changeDynamics(car, wheel, lateralFriction=0.5)  # Tăng/giảm để điều chỉnh độ bám



# print params
# Get wheel joint indices
wheel_joints = {
    "front_left": 2,  # Left front wheel
    "front_right": 3, # Right front wheel
    "rear_left": 5,   # Left rear wheel
    "rear_right": 7   # Right rear wheel
}

# Get wheel positions
wheel_positions = {}
for wheel, joint_index in wheel_joints.items():
    wheel_positions[wheel] = np.array(p.getLinkState(car, joint_index)[0])  # Extract position

# Calculate wheelbase (distance between front and rear wheels)
wheelbase = np.linalg.norm(wheel_positions["front_left"] - wheel_positions["rear_left"])

# Calculate wheeltrack (distance between left and right wheels)
wheeltrack = np.linalg.norm(wheel_positions["front_left"] - wheel_positions["front_right"])

print(f"Wheelbase: {wheelbase:.3f} meters")
print(f"Wheeltrack: {wheeltrack:.3f} meters")



# Disable velocity control for front wheels (steering control)
p.setJointMotorControl2(car, 4, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(car, 6, p.VELOCITY_CONTROL, force=0)

# Ackermann & Pure Pursuit Parameters
L = wheelbase         # Wheelbase
track_width = wheeltrack # Distance between left and right wheels
speed = 10       # Base speed
look_ahead_dist = 1.0  # Look-ahead distance
a = 5           # Amplitude of the eight-shape (half-width)
b = 3           # Height of loops
num_waypoints = 500  # More waypoints for smoother tracking

# Generate waypoints for figure-eight path (Lissajous curve approach)
waypoints = np.array([
    [a * math.sin(2 * math.pi * i / num_waypoints),  
     b * math.sin(4 * math.pi * i / num_waypoints)]
    for i in range(num_waypoints)
])

# Draw the figure-eight trajectory (blue)
for i in range(len(waypoints) - 1):
    p.addUserDebugLine([waypoints[i][0], waypoints[i][1], 0.1],
                       [waypoints[i + 1][0], waypoints[i + 1][1], 0.1],
                       [0, 0, 1], 2)

# Store vehicle trace points
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

    lookahead_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.5])
    lookahead_id = p.createMultiBody(baseVisualShapeIndex=lookahead_sphere, 
                                 basePosition=[lookahead_wp[0], lookahead_wp[1], 0.2])

    # Tự động xóa sau 1 giây
    p.addUserDebugText("", [0, 0, 0], lifeTime=15)  # Tạo một object ảo để trigger update
    p.removeBody(lookahead_id) 


    # Draw a small green line connecting car to lookahead point
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
        p.setJointMotorControl2(car, 4, p.POSITION_CONTROL, targetPosition=inner_angle)
        p.setJointMotorControl2(car, 6, p.POSITION_CONTROL, targetPosition=outer_angle)
    else:  # Turning right or straight
        p.setJointMotorControl2(car, 4, p.POSITION_CONTROL, targetPosition=-inner_angle)
        p.setJointMotorControl2(car, 6, p.POSITION_CONTROL, targetPosition=-outer_angle)

    # Compute wheel speeds
    inner_speed = velocity * (R_inner / (R_inner + track_width)) if R_inner != float('inf') else velocity
    outer_speed = velocity * (R_outer / (R_outer + track_width)) if R_outer != float('inf') else velocity

    # Apply speed to all wheels (4AWD)
    p.setJointMotorControl2(car, 2, p.VELOCITY_CONTROL, targetVelocity=inner_speed, force=10)
    p.setJointMotorControl2(car, 3, p.VELOCITY_CONTROL, targetVelocity=outer_speed, force=10)
    p.setJointMotorControl2(car, 5, p.VELOCITY_CONTROL, targetVelocity=inner_speed, force=10)
    p.setJointMotorControl2(car, 7, p.VELOCITY_CONTROL, targetVelocity=outer_speed, force=10)

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
        
    pos, ori = p.getBasePositionAndOrientation(car)
    # Lấy vị trí xe
    car_x, car_y, car_z = pos
    
    # Thiết lập góc nhìn từ phía sau xe (camera view)
    camera_distance = 1.9   # Khoảng cách từ camera đến xe
    camera_yaw = -20        # Góc xoay ngang (0 nhìn thẳng, ±90 nhìn ngang)
    camera_pitch = -30    # Góc nhìn từ trên xuống (-30 độ)
    
    # p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, [car_x, car_y, car_z])


    # Step simulation faster
    for _ in range(3):  # Run multiple steps per loop iteration
        p.stepSimulation()

    #time.sleep(1./240.0)  # Faster refresh rate

p.disconnect()

