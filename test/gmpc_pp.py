import numpy as np
import pybullet as p
import pybullet_data
import math
import time
from enum import Enum
import casadi as ca
from manifpy import SE2, SE2Tangent


# Define trajectory types
class TrajType(Enum):
    CIRCLE = 1
    EIGHT = 2
    LINE = 3

# Reference Trajectory Generator
class RefTrajGenerator:
    def __init__(self, traj_config):
        """Initialize the trajectory generator with configuration."""
        self.traj_type = traj_config['type']
        self.param = traj_config['param']
        self.dt = self.param['dt']
        self.nTraj = self.param['nTraj']
        self.start_state = self.param['start_state']
        self.linear_vel = self.param['linear_vel']
        self.angular_vel = self.param['angular_vel']
        self.ref_control = self._compute_ref_control()

    def _compute_ref_control(self):
        """Compute reference control inputs based on trajectory type."""
        if self.traj_type == TrajType.LINE:
            return np.tile([self.linear_vel, 0], (self.nTraj, 1)).T
        elif self.traj_type == TrajType.CIRCLE:
            return np.tile([self.linear_vel, self.angular_vel], (self.nTraj, 1)).T
        elif self.traj_type == TrajType.EIGHT:
            t = np.arange(self.nTraj) * self.dt
            period = self.nTraj * self.dt
            w = self.angular_vel * np.sin(4 * np.pi * t / period)
            return np.vstack([np.full(self.nTraj, self.linear_vel), w])
        else:
            raise ValueError(f"Unsupported trajectory type: {self.traj_type}")

    def get_traj(self):
        """Generate the reference trajectory states and controls."""
        ref_state = np.zeros((3, self.nTraj))
        ref_state[:, 0] = self.start_state
        X = SE2(self.start_state[0], self.start_state[1], self.start_state[2])
        for k in range(self.nTraj - 1):
            v, w = self.ref_control[:, k]
            xi = np.array([v, 0.0, w])
            xi_scaled = xi * self.dt
            X = X + SE2Tangent(xi_scaled)
            ref_state[:, k + 1] = [X.x(), X.y(), X.angle()]
        return ref_state, self.ref_control, self.dt

# Geometric MPC Controller
class GeometricMPC:
    def __init__(self, linearization_type='ADJ'):
        """Initialize the Geometric MPC controller."""
        self.nState = 3
        self.nControl = 2
        self.nTwist = 3
        self.Q = None
        self.R = None
        self.N = None
        self.linearizationType = linearization_type
        self.solve_time = 0.0

    def setup_solver(self, Q=[20000, 20000, 2000], R=0.3, N=10):
        """Set up the MPC solver with cost matrices and horizon."""
        self.Q = np.diag(Q)
        self.R = R * np.eye(self.nControl)
        self.N = N

    def set_control_bound(self, v_min=-10, v_max=10, w_min=-np.pi, w_max=np.pi):
        """Set control input bounds."""
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max

    def vel_cmd_to_local_twist(self, vel_cmd):
        """Convert velocity command to local twist."""
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])

    def find_closest_index(self, current_state, ref_trajectory_path):
        """Find the index of the closest point on the reference path."""
        ref_state = ref_trajectory_path[0]
        distances = np.linalg.norm(ref_state[:2, :] - current_state[:2, np.newaxis], axis=0)
        return np.argmin(distances)

    def get_local_ref(self, current_state, ref_trajectory_path):
        """Extract local reference trajectory for the MPC horizon."""
        ref_state, ref_control, dt = ref_trajectory_path
        closest_idx = self.find_closest_index(current_state, ref_trajectory_path)
        local_ref_state = ref_state[:, closest_idx:closest_idx + self.N + 1]
        local_ref_control = ref_control[:, closest_idx:closest_idx + self.N]
        if local_ref_state.shape[1] < self.N + 1:
            pad_state = np.tile(ref_state[:, -1:], (1, self.N + 1 - local_ref_state.shape[1]))
            local_ref_state = np.hstack((local_ref_state, pad_state))
            pad_control = np.tile(ref_control[:, -1:], (1, self.N - local_ref_control.shape[1]))
            local_ref_control = np.hstack((local_ref_control, pad_control))
        return local_ref_state, local_ref_control, dt

    def solve(self, current_state, ref_trajectory_path):
        """Solve the MPC optimization problem."""
        start_time = time.time()
        local_ref_state, local_ref_control, dt = self.get_local_ref(current_state, ref_trajectory_path)
        opti = ca.Opti('conic')
        x_var = opti.variable(self.nState, self.N + 1)
        u_var = opti.variable(self.nControl, self.N)
        X_curr = SE2(current_state[0], current_state[1], current_state[2])
        X_ref = SE2(local_ref_state[0, 0], local_ref_state[1, 0], local_ref_state[2, 0])
        x_init = X_ref.between(X_curr).log().coeffs()
        opti.subject_to(x_var[:, 0] == x_init)
        for i in range(self.N):
            u_d = local_ref_control[:, i]
            twist_d = self.vel_cmd_to_local_twist(u_d)
            A = -SE2Tangent(twist_d).smallAdj() if self.linearizationType == 'ADJ' else -SE2Tangent(twist_d).hat()
            B = np.eye(self.nTwist)
            h = -twist_d
            x_next = x_var[:, i] + dt * (A @ x_var[:, i] + B @ self.vel_cmd_to_local_twist(u_var[:, i]) + h)
            opti.subject_to(x_var[:, i + 1] == x_next)
        cost = sum(ca.mtimes([x_var[:, i].T, self.Q, x_var[:, i]]) +
                   ca.mtimes([(u_var[:, i] - local_ref_control[:, i]).T, self.R, (u_var[:, i] - local_ref_control[:, i])])
                   for i in range(self.N))
        cost += ca.mtimes([x_var[:, self.N].T, 100 * self.Q, x_var[:, self.N]])
        opti.subject_to(opti.bounded(self.v_min, u_var[0, :], self.v_max))
        opti.subject_to(opti.bounded(self.w_min, u_var[1, :], self.w_max))
        opti.solver('qpoases', {'printLevel': 'none'})
        opti.minimize(cost)
        sol = opti.solve()
        u_opt = sol.value(u_var[:, 0])
        self.solve_time = time.time() - start_time
        return u_opt

# Pure Pursuit Controller
class PurePursuitController:
    def __init__(self, lookahead_distance, wheelbase, linear_vel):
        """Initialize the Pure Pursuit controller."""
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.linear_vel = linear_vel
        self.lookahead_point = None  # To store the lookahead point for visualization

    def find_closest_point(self, current_position, path):
        """Find the closest point on the path to the current position."""
        min_dist = float('inf')
        closest_point = None
        closest_segment_idx = None
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            ab = p2 - p1
            ap = current_position - p1
            ab_norm = np.dot(ab, ab)
            if ab_norm < 1e-6:
                continue
            t = np.dot(ap, ab) / ab_norm
            t = max(0, min(1, t))
            projection = p1 + t * ab
            dist = np.linalg.norm(current_position - projection)
            if dist < min_dist:
                min_dist = dist
                closest_point = projection
                closest_segment_idx = i
        if closest_point is None:
            closest_point = path[0]
            closest_segment_idx = 0
        return closest_point, closest_segment_idx

    def find_lookahead_point(self, closest_point, closest_segment_idx, path):
        """Find the lookahead point along the path."""
        current_point = closest_point
        accumulated_distance = 0.0
        i = closest_segment_idx
        while i < len(path) - 1:
            p1 = path[i]
            p2 = path[i + 1]
            segment_length = np.linalg.norm(p2 - p1)
            if i == closest_segment_idx:
                remaining_segment = np.linalg.norm(p2 - current_point)
            else:
                remaining_segment = segment_length
            if accumulated_distance + remaining_segment >= self.lookahead_distance:
                remaining_distance = self.lookahead_distance - accumulated_distance
                direction = (p2 - p1) / segment_length
                lookahead_point = current_point + remaining_distance * direction
                return lookahead_point
            else:
                accumulated_distance += remaining_segment
                current_point = p2
                i += 1
        return path[-1]

    def compute_steering_angle(self, current_position, current_heading, lookahead_point):
        """Compute the steering angle based on the lookahead point."""
        dx = lookahead_point[0] - current_position[0]
        dy = lookahead_point[1] - current_position[1]
        alpha = math.atan2(dy, dx) - current_heading
        delta = math.atan2(2 * self.wheelbase * math.sin(alpha), self.lookahead_distance)
        return delta

    def compute_control(self, current_state, path):
        """Compute control inputs (linear and angular velocity)."""
        current_position = current_state[:2]
        current_heading = current_state[2]
        closest_point, closest_segment_idx = self.find_closest_point(current_position, path)
        lookahead_point = self.find_lookahead_point(closest_point, closest_segment_idx, path)
        self.lookahead_point = lookahead_point  # Store for visualization
        delta = self.compute_steering_angle(current_position, current_heading, lookahead_point)
        v = self.linear_vel
        w = (v / self.wheelbase) * math.tan(delta)
        return v, w

# PyBullet Car Simulation
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
        p.changeDynamics(self.plane_id, -1, lateralFriction=0.1)
        wheels = [2, 3, 5, 7]
        for wheel in wheels:
            p.changeDynamics(self.car_id, wheel, lateralFriction=0.1)
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



# Main Simulation Loop
if __name__ == "__main__":
    # Trajectory configuration for a circular path
    radius = 5.0
    traj_config = {
        'type': TrajType.CIRCLE,
        'param': {
            'start_state': np.array([0, -3, 0]),
            'linear_vel': 10.0,
            'angular_vel': 5.0 / radius,
            'nTraj': 200,
            'dt': 0.05
        }
    }

    # Initialize components
    traj_generator = RefTrajGenerator(traj_config)
    ref_trajectory_path = traj_generator.get_traj()
    ref_state = ref_trajectory_path[0]

    start_point = [0, -0, 0.0]
    start_heading = 0
    start_orientation = p.getQuaternionFromEuler([0, 0, start_heading])
    car = PyBulletCar("racecar/racecar.urdf", start_point, start_orientation)

    mpc = GeometricMPC()
    mpc.setup_solver(Q=[20000, 20000, 2000], R=0.3, N=20)
    mpc.set_control_bound(v_min=-10, v_max=10, w_min=-np.pi, w_max=np.pi)

    pure_pursuit = PurePursuitController(
        lookahead_distance=1.0,
        wheelbase=car.wheelbase,
        linear_vel=traj_config['param']['linear_vel']
    )

    # Visualize reference trajectory (blue)
    for i in range(ref_state.shape[1] - 1):
        p.addUserDebugLine([ref_state[0, i], ref_state[1, i], 0.1],
                           [ref_state[0, i + 1], ref_state[1, i + 1], 0.1],
                           [0, 0, 1], 2)

    # Define the controller selector
    use_pure_pursuit = True  # Set to False to use Geometric MPC

    # Simulation loop
    trace_points = []
    while True:
        current_state = car.get_state()
        if use_pure_pursuit:
            # Use Pure Pursuit controller
            path = ref_state[:2, :].T  # Extract [x, y] points as (nTraj, 2) array
            v, w = pure_pursuit.compute_control(current_state, path)
            if pure_pursuit.lookahead_point is not None:
                # Update sphere position to lookahead point
                p.resetBasePositionAndOrientation(
                    car.lookahead_id,
                    [pure_pursuit.lookahead_point[0], pure_pursuit.lookahead_point[1], 0.1],
                    [0, 0, 0, 1]
                )
                # Draw green line from car to lookahead point
                p.addUserDebugLine(
                    [current_state[0], current_state[1], 0.1],
                    [pure_pursuit.lookahead_point[0], pure_pursuit.lookahead_point[1], 0.1],
                    [0, 1, 0], 2, lifeTime=0.1
                )
        else:
            # Hide sphere when using MPC
            p.resetBasePositionAndOrientation(car.lookahead_id, [0, 0, -10], [0, 0, 0, 1])
            v, w = mpc.solve(current_state, ref_trajectory_path)

        car.apply_control(v, w)
        car.step_simulation(dt=0.05)

        # Trace the car's path (yellow)
        trace_points.append((current_state[0], current_state[1]))
        if len(trace_points) > 1:
            for i in range(len(trace_points) - 1):
                p.addUserDebugLine([trace_points[i][0], trace_points[i][1], 0.05],
                                   [trace_points[i + 1][0], trace_points[i + 1][1], 0.05],
                                   [1, 1, 0], 2, lifeTime=10)
        if len(trace_points) > 500:
            trace_points.pop(0)

        time.sleep(0.001)