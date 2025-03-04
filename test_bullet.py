import pybullet as p
import pybullet_data
import time
import math

# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Add default data path

# Set gravity and time step
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/30.0)

# Load environment
planeId = p.loadURDF("plane.urdf")
car = p.loadURDF("racecar/racecar.urdf", basePosition=[0, 0, 0.1])

# Disable velocity control for front wheels (so we can set steering manually)
p.setJointMotorControl2(car, 4, p.VELOCITY_CONTROL, force=0)  # Left front
p.setJointMotorControl2(car, 6, p.VELOCITY_CONTROL, force=0)  # Right front

# Ackermann parameters
L = 0.5  # Wheelbase
track_width = 0.4  # Distance between left and right wheels
steering_angle = 0  # Initial steering angle
velocity = 0  # Initial speed

def apply_ackermann_control(steering_angle, velocity):
    """Applies Ackermann steering with the given steering angle and velocity."""
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

# Simulation loop
while True:
    keys = p.getKeyboardEvents()

    # Steering controls
    if ord('a') in keys:  # Steer Left
        steering_angle = max(steering_angle - 0.05, -0.5)
    elif ord('d') in keys:  # Steer Right
        steering_angle = min(steering_angle + 0.05, 0.5)
    elif ord('s') in keys:  # Reset steering
        steering_angle = 0

    # Speed controls
    if ord('w') in keys:  # Move Forward
        velocity = min(velocity + 0.5, 10)
    elif ord('x') in keys:  # Move Backward
        velocity = max(velocity - 0.5, -10)
    elif ord('z') in keys:  # Stop
        velocity = 0

    # Apply Ackermann control
    apply_ackermann_control(steering_angle, velocity)

    # Step simulation
    p.stepSimulation()
    time.sleep(1./30.0)

p.disconnect()

