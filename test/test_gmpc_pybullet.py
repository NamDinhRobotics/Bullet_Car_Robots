import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import casadi as ca
from manifpy import SE2, SE2Tangent
from enum import Enum

# Define enumerations for trajectory types
class TrajType(Enum):
    CIRCLE = 1
    EIGHT = 2
    LINE = 3

# Reference Trajectory Generator
class RefTrajGenerator:
    def __init__(self, traj_config):
        """
        Initialize the trajectory generator with a configuration dictionary.

        Args:
            traj_config (dict): Configuration with 'type' (TrajType) and 'param' (dict).
                                'param' contains 'dt', 'nTraj', 'start_state', 'linear_vel', 'angular_vel'.
        """
        self.traj_type = traj_config['type']
        self.param = traj_config['param']
        self.dt = self.param['dt']  # Time step
        self.nTraj = self.param['nTraj']  # Number of trajectory points
        self.start_state = self.param['start_state']  # Initial state [x, y, theta]
        self.linear_vel = self.param['linear_vel']  # Linear velocity (v)
        self.angular_vel = self.param['angular_vel']  # Angular velocity (w)
        self.ref_control = self._compute_ref_control()

    def _compute_ref_control(self):
        """
        Compute the reference control sequence based on trajectory type.

        Returns:
            np.ndarray: Control sequence of shape (2, nTraj), [v, w] for each time step.
        """
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
        """
        Generate the reference trajectory by integrating the unicycle model.

        Returns:
            tuple: (ref_state, ref_control, dt)
                   - ref_state: np.ndarray of shape (3, nTraj), [x, y, theta]
                   - ref_control: np.ndarray of shape (2, nTraj), [v, w]
                   - dt: float, time step
        """
        ref_state = np.zeros((3, self.nTraj))
        ref_state[:, 0] = self.start_state
        X = SE2(self.start_state[0], self.start_state[1], self.start_state[2])

        for k in range(self.nTraj - 1):
            v, w = self.ref_control[:, k]
            xi = np.array([v, 0.0, w])  # Local twist [v_x, v_y, w]
            xi_scaled = xi * self.dt
            X = X + SE2Tangent(xi_scaled)
            ref_state[:, k + 1] = [X.x(), X.y(), X.angle()]

        return ref_state, self.ref_control, self.dt

# Geometric MPC Class
class GeometricMPC:
    def __init__(self, linearization_type='ADJ'):
        self.controllerType = 'GMPC'
        self.nState = 3  # SE(2) error state (x, y, theta) in Lie algebra
        self.nControl = 2  # Control inputs (v, w)
        self.nTwist = 3  # Twist vector in SE(2) (v_x, v_y, w)
        self.Q = None  # State cost weighting matrix
        self.R = None  # Control cost weighting matrix
        self.N = None  # Prediction horizon
        self.linearizationType = linearization_type
        self.solve_time = 0.0
        self.local_ref_state = None  # To store the local reference state
        self.setup_solver()
        self.set_control_bound()

    def setup_solver(self, Q=[20000, 20000, 2000], R=0.3, N=10):
        """Initialize the solver parameters."""
        self.Q = np.diag(Q)  # Emphasize position over orientation
        self.R = R * np.eye(self.nControl)  # Control effort penalty
        self.N = N  # Prediction horizon

    def set_control_bound(self, v_min=-10, v_max=10, w_min=-np.pi, w_max=np.pi):
        """Set bounds for control inputs."""
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max

    def vel_cmd_to_local_twist(self, vel_cmd):
        """Convert velocity command [v, w] to twist [v, 0, w]."""
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])

    def local_twist_to_vel_cmd(self, local_vel):
        """Convert twist [v_x, v_y, w] to velocity command [v, w]."""
        return ca.vertcat(local_vel[0], local_vel[2])

    def find_closest_index(self, current_state, ref_trajectory_path):
        """Find the index of the closest point on the reference trajectory."""
        ref_state = ref_trajectory_path[0]
        distances = np.linalg.norm(ref_state[:2, :] - current_state[:2, np.newaxis], axis=0)
        return np.argmin(distances)

    def get_local_ref(self, current_state, ref_trajectory_path):
        """Extract a local reference trajectory starting from the closest point."""
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
        """
        Solve the MPC optimization problem using SE(2) geometry.

        Args:
            current_state: Current state [x, y, theta]
            ref_trajectory_path: Tuple (ref_state, ref_control, dt)
        Returns:
            u: Optimal control input [v, w]
        """
        start_time = time.time()
        local_ref_state, local_ref_control, dt = self.get_local_ref(current_state, ref_trajectory_path)
        self.local_ref_state = local_ref_state  # Store the local reference state

        opti = ca.Opti('conic')
        x_var = opti.variable(self.nState, self.N + 1)  # States over horizon
        u_var = opti.variable(self.nControl, self.N)  # Controls over horizon

        X_curr = SE2(current_state[0], current_state[1], current_state[2])
        X_ref = SE2(local_ref_state[0, 0], local_ref_state[1, 0], local_ref_state[2, 0])
        x_init = X_ref.between(X_curr).log().coeffs()
        opti.subject_to(x_var[:, 0] == x_init)

        for i in range(self.N):
            u_d = local_ref_control[:, i]
            twist_d = self.vel_cmd_to_local_twist(u_d)
            if self.linearizationType == 'ADJ':
                A = -SE2Tangent(twist_d).smallAdj()
            else:
                A = -SE2Tangent(twist_d).hat()
            B = np.eye(self.nTwist)
            h = -twist_d
            x_next = x_var[:, i] + dt * (A @ x_var[:, i] + B @ self.vel_cmd_to_local_twist(u_var[:, i]) + h)
            opti.subject_to(x_var[:, i + 1] == x_next)

        cost = 0
        for i in range(self.N):
            u_d = local_ref_control[:, i]
            cost += ca.mtimes([x_var[:, i].T, self.Q, x_var[:, i]]) + \
                    ca.mtimes([(u_var[:, i] - u_d).T, self.R, (u_var[:, i] - u_d)])
        cost += ca.mtimes([x_var[:, self.N].T, 100 * self.Q, x_var[:, self.N]])  # Terminal cost

        opti.subject_to(opti.bounded(self.v_min, u_var[0, :], self.v_max))
        opti.subject_to(opti.bounded(self.w_min, u_var[1, :], self.w_max))

        opti.solver('qpoases', {'printLevel': 'none'})
        opti.minimize(cost)
        sol = opti.solve()

        u_opt = sol.value(u_var[:, 0])
        self.solve_time = time.time() - start_time
        return u_opt

# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity and time step
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/120.0)
p.setRealTimeSimulation(1)  # Enable real-time simulation

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

# Disable velocity control for steering wheels
p.setJointMotorControl2(car, 4, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(car, 6, p.VELOCITY_CONTROL, force=0)

# Trajectory Configuration
radius = 3.0
traj_config = {
    'type': TrajType.CIRCLE,
    'param': {
        'start_state': np.array([0, -3, 0]),  # Match initial car position
        'linear_vel': 5.0,
        'angular_vel': 5.0 / radius,  # w = v / r for circular motion
        'nTraj': 200,
        'dt': 0.05
    }
}
traj_generator = RefTrajGenerator(traj_config)
ref_trajectory_path = traj_generator.get_traj()
ref_state, ref_control, dt = ref_trajectory_path

# Visualize reference trajectory in PyBullet (blue)
for i in range(ref_state.shape[1] - 1):
    p.addUserDebugLine([ref_state[0, i], ref_state[1, i], 0.1],
                       [ref_state[0, i + 1], ref_state[1, i + 1], 0.1],
                       [0, 0, 1], 2)

# Initialize MPC
mpc = GeometricMPC()
mpc.setup_solver(Q=[20000, 20000, 2000], R=0.3, N=20)
mpc.set_control_bound(v_min=-10, v_max=10, w_min=-np.pi, w_max=np.pi)

# Apply Ackermann steering
def apply_ackermann_control(steering_angle, velocity):
    """
    Apply Ackermann steering to the car based on steering angle and velocity.

    Args:
        steering_angle (float): Steering angle in radians
        velocity (float): Linear velocity in m/s
    """
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
sim_dt = 1 / 120.0
num_steps = int(dt / sim_dt)  # Number of simulation steps per MPC update
trace_points = []
local_ref_ids = []  # List to store IDs of local reference debug lines

while True:
    # Get current state
    pos, ori = p.getBasePositionAndOrientation(car)
    yaw = p.getEulerFromQuaternion(ori)[2]
    current_state = np.array([pos[0], pos[1], yaw])

    # Compute optimal control inputs using MPC
    u_opt = mpc.solve(current_state, ref_trajectory_path)
    v, w = u_opt

    # Convert angular velocity to steering angle
    if v != 0:
        steering_angle = math.atan((w * wheelbase) / v)
    else:
        steering_angle = 0  # Default to straight if velocity is zero

    # Apply control to the car
    apply_ackermann_control(steering_angle, v)

    # Remove previous local reference lines
    for id in local_ref_ids:
        p.removeUserDebugItem(id)
    local_ref_ids = []

    # Draw new local reference lines (green)
    if mpc.local_ref_state is not None:
        local_ref = mpc.local_ref_state
        for i in range(local_ref.shape[1] - 1):
            id = p.addUserDebugLine(
                [local_ref[0, i], local_ref[1, i], 0.2],
                [local_ref[0, i + 1], local_ref[1, i + 1], 0.2],
                [0, 1, 0], 2
            )
            local_ref_ids.append(id)

    # Step simulation for dt=0.05 s
    for _ in range(num_steps):
        p.stepSimulation()

    # Store and draw trace (yellow)
    trace_points.append((pos[0], pos[1]))
    if len(trace_points) > 1:
        for i in range(len(trace_points) - 1):
            p.addUserDebugLine([trace_points[i][0], trace_points[i][1], 0.05],
                               [trace_points[i + 1][0], trace_points[i + 1][1], 0.05],
                               [1, 1, 0], 2, lifeTime=10)

    # Limit trace points to prevent performance issues
    if len(trace_points) > 500:
        trace_points.pop(0)

    # Small sleep to control loop rate
    time.sleep(0.001)