import numpy as np
import time
import casadi as ca
from manifpy import SE2, SE2Tangent

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