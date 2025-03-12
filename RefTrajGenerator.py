import numpy as np
from enum import Enum
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