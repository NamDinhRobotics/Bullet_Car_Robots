import numpy as np
import pybullet as p
import pybullet_data
import math

class PyBulletCar:
    def __init__(self, urdf_path, start_position, start_orientation=None):
        """Initialize the PyBullet car simulation."""
        self.start_orientation = start_orientation
        self.start_position = start_position
        self.physics_client = p.connect(p.GUI)

        # Create a persistent red sphere for the lookahead point
        sphere_radius = 0.1
        sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1, 0, 0, 0.5])
        lookahead_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_visual, basePosition=[0, 0, -10])
        self.lookahead_id = lookahead_id


        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/120.0)
        p.setRealTimeSimulation(1)
        self.plane_id = p.loadURDF("plane.urdf")
        self.car_id = p.loadURDF(urdf_path, basePosition=start_position, baseOrientation=start_orientation)
        p.changeDynamics(self.plane_id, -1, lateralFriction=0.8)
        wheels = [2, 3, 5, 7]
        for wheel in wheels:
            p.changeDynamics(self.car_id, wheel, lateralFriction=0.8)
        p.setJointMotorControl2(self.car_id, 4, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.car_id, 6, p.VELOCITY_CONTROL, force=0)
        self.wheel_joints = {"front_left": 2, "front_right": 3, "rear_left": 5, "rear_right": 7}
        wheel_positions = {wheel: np.array(p.getLinkState(self.car_id, joint)[0])
                           for wheel, joint in self.wheel_joints.items()}
        self.wheelbase = np.linalg.norm(wheel_positions["front_left"] - wheel_positions["rear_left"])
        self.wheeltrack = np.linalg.norm(wheel_positions["front_left"] - wheel_positions["front_right"])

    def apply_control(self, v, w):
        """Apply linear and angular velocity to the car."""
        steering_angle = math.atan((w * self.wheelbase) / v) if v != 0 else 0
        self._apply_ackermann_control(steering_angle, v)

    def _apply_ackermann_control(self, steering_angle, velocity):
        """Apply Ackermann steering control to the car."""
        if abs(steering_angle) < 1e-3:
            R_inner = R_outer = float('inf')
        else:
            R_inner = self.wheelbase / math.tan(abs(steering_angle))
            R_outer = R_inner + self.wheeltrack if steering_angle > 0 else R_inner - self.wheeltrack
        inner_angle = math.atan(self.wheelbase / R_inner) if R_inner != float('inf') else 0
        outer_angle = math.atan(self.wheelbase / R_outer) if R_outer != float('inf') else 0
        if steering_angle > 0:
            p.setJointMotorControl2(self.car_id, 4, p.POSITION_CONTROL, targetPosition=inner_angle)
            p.setJointMotorControl2(self.car_id, 6, p.POSITION_CONTROL, targetPosition=outer_angle)
        else:
            p.setJointMotorControl2(self.car_id, 4, p.POSITION_CONTROL, targetPosition=-inner_angle)
            p.setJointMotorControl2(self.car_id, 6, p.POSITION_CONTROL, targetPosition=-outer_angle)
        inner_speed = velocity * (R_inner / (R_inner + self.wheeltrack)) if R_inner != float('inf') else velocity
        outer_speed = velocity * (R_outer / (R_outer + self.wheeltrack)) if R_outer != float('inf') else velocity
        for wheel in [2, 5]:
            p.setJointMotorControl2(self.car_id, wheel, p.VELOCITY_CONTROL, targetVelocity=inner_speed, force=10)
        for wheel in [3, 7]:
            p.setJointMotorControl2(self.car_id, wheel, p.VELOCITY_CONTROL, targetVelocity=outer_speed, force=10)

    def get_state(self):
        """Get the current state of the car (x, y, yaw)."""
        pos, ori = p.getBasePositionAndOrientation(self.car_id)
        yaw = p.getEulerFromQuaternion(ori)[2]
        return np.array([pos[0], pos[1], yaw])

    def step_simulation(self, dt):
        """Step the simulation forward by dt seconds."""
        num_steps = int(dt / (1/120.0))
        for _ in range(num_steps):
            p.stepSimulation()