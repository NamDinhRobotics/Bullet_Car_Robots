import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent
import casadi as ca
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
            # Straight line: constant v, w = 0
            return np.tile([self.linear_vel, 0], (self.nTraj, 1)).T
        elif self.traj_type == TrajType.CIRCLE:
            # Circle: constant v and w
            return np.tile([self.linear_vel, self.angular_vel], (self.nTraj, 1)).T
        elif self.traj_type == TrajType.EIGHT:
            # Figure-eight: constant v, sinusoidally varying w
            t = np.arange(self.nTraj) * self.dt
            period = self.nTraj * self.dt
            w = self.angular_vel * np.sin(4 * np.pi * t / period)  # Two cycles for figure-eight
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
        self.setup_solver()
        self.set_control_bound()

    def setup_solver(self, Q=[20000, 20000, 2000], R=0.3, N=10):
        """Initialize the solver parameters."""
        self.Q = np.diag(Q)  # Emphasize position over orientation
        self.R = R * np.eye(self.nControl)  # Control effort penalty
        self.N = N  # Prediction horizon

    def set_control_bound(self, v_min=-100, v_max=100, w_min=-100, w_max=100):
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
        """Find the index of the closest point on the reference trajectory to the current state."""
        ref_state = ref_trajectory_path[0]  # Extract ref_state from tuple
        distances = np.linalg.norm(ref_state[:2, :] - current_state[:2, np.newaxis], axis=0)
        return np.argmin(distances)

    def get_local_ref(self, current_state, ref_trajectory_path):
        """Extract a local reference trajectory starting from the closest point."""
        ref_state, ref_control, dt = ref_trajectory_path
        closest_idx = self.find_closest_index(current_state, ref_trajectory_path)
        local_ref_state = ref_state[:, closest_idx:closest_idx + self.N + 1]
        local_ref_control = ref_control[:, closest_idx:closest_idx + self.N]
        if local_ref_state.shape[1] < self.N + 1:
            # Pad with the last state/control if necessary
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
            ref_trajectory_path: Tuple (ref_state, ref_control, dt) where
                                 - ref_state: np.ndarray of shape (3, nTraj), [x, y, theta]
                                 - ref_control: np.ndarray of shape (2, nTraj), [v, w]
                                 - dt: float, time step
        Returns:
            u: Optimal control input [v, w]
        """
        start_time = time.time()
        local_ref_state, local_ref_control, dt = self.get_local_ref(current_state, ref_trajectory_path)

        # Casadi optimization setup
        opti = ca.Opti('conic')
        x_var = opti.variable(self.nState, self.N + 1)  # States over horizon
        u_var = opti.variable(self.nControl, self.N)  # Controls over horizon

        # Initial condition in Lie algebra
        X_curr = SE2(current_state[0], current_state[1], current_state[2])
        X_ref = SE2(local_ref_state[0, 0], local_ref_state[1, 0], local_ref_state[2, 0])
        x_init = X_ref.between(X_curr).log().coeffs()
        opti.subject_to(x_var[:, 0] == x_init)

        # Dynamics constraints using SE(2) linearization
        for i in range(self.N):
            u_d = local_ref_control[:, i]
            twist_d = self.vel_cmd_to_local_twist(u_d)
            if self.linearizationType == 'ADJ':
                A = -SE2Tangent(twist_d).smallAdj()
            else:  # WEDGE
                A = -SE2Tangent(twist_d).hat()
            B = np.eye(self.nTwist)
            h = -twist_d
            x_next = x_var[:, i] + dt * (A @ x_var[:, i] + B @ self.vel_cmd_to_local_twist(u_var[:, i]) + h)
            opti.subject_to(x_var[:, i + 1] == x_next)

        # Cost function
        cost = 0
        for i in range(self.N):
            u_d = local_ref_control[:, i]
            cost += ca.mtimes([x_var[:, i].T, self.Q, x_var[:, i]]) + \
                    ca.mtimes([(u_var[:, i] - u_d).T, self.R, (u_var[:, i] - u_d)])
        cost += ca.mtimes([x_var[:, self.N].T, 100 * self.Q, x_var[:, self.N]])  # Terminal cost

        # Control bounds
        opti.subject_to(opti.bounded(self.v_min, u_var[0, :], self.v_max))
        opti.subject_to(opti.bounded(self.w_min, u_var[1, :], self.w_max))

        # Solver configuration
        opti.solver('qpoases', {'printLevel': 'none'})
        opti.minimize(cost)
        sol = opti.solve()

        u_opt = sol.value(u_var[:, 0])
        self.solve_time = time.time() - start_time
        return u_opt

def simulate_trajectory(mpc, init_state, traj_name, ref_trajectory_path):
    """Simulate the MPC tracking for a given trajectory."""
    ref_state, ref_control, dt = ref_trajectory_path
    n_steps = ref_state.shape[1]
    state_store = np.zeros((3, n_steps))
    vel_cmd_store = np.zeros((2, n_steps))
    state_store[:, 0] = init_state

    for i in range(n_steps - 1):
        state = state_store[:, i]
        vel_cmd = mpc.solve(state, ref_trajectory_path)
        vel_cmd_store[:, i] = vel_cmd
        xi = mpc.vel_cmd_to_local_twist(vel_cmd)
        X = SE2(state[0], state[1], state[2])
        X = X + SE2Tangent(xi * dt)
        state_store[:, i + 1] = [X.x(), X.y(), X.angle()]

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{traj_name} Trajectory Tracking')

    # Trajectory
    axs[0, 0].plot(ref_state[0, :], ref_state[1, :], 'r-', label='Reference')
    axs[0, 0].plot(state_store[0, :], state_store[1, :], 'b--', label='Actual')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    axs[0, 0].legend()
    axs[0, 0].axis('equal')

    # Distance error
    dist_error = np.linalg.norm(state_store[0:2, :] - ref_state[0:2, :], axis=0)
    axs[0, 1].plot(dist_error, 'k-')
    axs[0, 1].set_title('Distance Error')
    axs[0, 1].set_xlabel('Time Step')

    # Orientation error
    ori_error = np.zeros(n_steps)
    for i in range(n_steps):
        X_d = SE2(ref_state[0, i], ref_state[1, i], ref_state[2, i])
        X = SE2(state_store[0, i], state_store[1, i], state_store[2, i])
        ori_error[i] = scipy.linalg.norm(X_d.between(X).log().coeffs())
    axs[1, 0].plot(ori_error, 'k-')
    axs[1, 0].set_title('Orientation Error')
    axs[1, 0].set_xlabel('Time Step')

    # Velocity commands
    axs[1, 1].plot(vel_cmd_store[0, :-1], 'r-', label='Linear v')
    axs[1, 1].plot(vel_cmd_store[1, :-1], 'b--', label='Angular w')
    axs[1, 1].set_title('Velocity Commands')
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def test_multiple_trajectories():
    """Test GMPC with circle, eight, and line trajectories."""
    # Circle trajectory
    circle_config = {
        'type': TrajType.CIRCLE,
        'param': {
            'start_state': np.array([1, 1, np.pi / 2]),
            'linear_vel': 0.5,
            'angular_vel': 0.5,
            'nTraj': 270,
            'dt': 0.05
        }
    }
    traj_generator_circle = RefTrajGenerator(circle_config)
    ref_trajectory_circle = traj_generator_circle.get_traj()
    mpc_circle = GeometricMPC()
    init_state_circle = np.array([0.5, 1.5, np.pi/2])
    simulate_trajectory(mpc_circle, init_state_circle, "Circle", ref_trajectory_circle)

    # Eight trajectory
    eight_config = {
        'type': TrajType.EIGHT,
        'param': {
            'start_state': np.array([0, 0, 0]),
            'linear_vel': 0.5,
            'angular_vel': 1.0,
            'nTraj': 300,
            'dt': 0.05
        }
    }
    traj_generator_eight = RefTrajGenerator(eight_config)
    ref_trajectory_eight = traj_generator_eight.get_traj()
    mpc_eight = GeometricMPC()
    init_state_eight = np.array([0.1, 0.1, np.pi / 4])
    simulate_trajectory(mpc_eight, init_state_eight, "Eight", ref_trajectory_eight)

    # Line trajectory
    line_config = {
        'type': TrajType.LINE,
        'param': {
            'start_state': np.array([0, 0, 0]),
            'linear_vel': 1.0,
            'angular_vel': 0.0,
            'nTraj': 200,
            'dt': 0.05
        }
    }
    traj_generator_line = RefTrajGenerator(line_config)
    ref_trajectory_line = traj_generator_line.get_traj()
    mpc_line = GeometricMPC()
    init_state_line = np.array([-0.2, 0.2, np.pi / 6])
    simulate_trajectory(mpc_line, init_state_line, "Line", ref_trajectory_line)

if __name__ == "__main__":
    test_multiple_trajectories()