from GeometricMPC import *
from PurePursuitController import *
from PyBulletCar import *
from RefTrajGenerator import *

# Main Simulation Loop
if __name__ == "__main__":
    # Trajectory configuration for a circular path
    radius = 3.0
    traj_config = {
        'type': TrajType.CIRCLE,
        'param': {
            'start_state': np.array([0, -3, 0]),
            'linear_vel': 5.0,
            'angular_vel': 5.0 / radius,
            'nTraj': 200,
            'dt': 0.05
        }
    }

    # Initialize components
    traj_generator = RefTrajGenerator(traj_config)  # Create a reference trajectory generator
    ref_trajectory_path = traj_generator.get_traj()  # Generate the reference trajectory
    ref_state = ref_trajectory_path[0]  # Extract the reference states (x, y, theta)

    start_point = [0, 0, 0.0]
    start_heading = np.pi/2
    start_orientation = p.getQuaternionFromEuler([0, 0, start_heading])
    car = PyBulletCar("racecar/racecar.urdf", start_point, start_orientation)  # Create the car in PyBullet

    mpc = GeometricMPC()  # Create the Geometric MPC controller
    mpc.setup_solver(Q=[20000, 20000, 2000], R=0.3, N=20)  # Set up the MPC solver parameters
    mpc.set_control_bound(v_min=-10, v_max=10, w_min=-np.pi, w_max=np.pi)  # Set control input bounds

    pure_pursuit = PurePursuitController(
        min_lookahead=0.3,
        max_lookahead=1.5,
        wheelbase=car.wheelbase,
        max_speed=15.0,  # Maximum speed on straight paths
        min_speed=2.0,  # Minimum speed during sharp turns
        max_steering_angle=np.pi/6  # Maximum steering angle in radians
    )


    # Visualize reference trajectory (blue)
    for i in range(ref_state.shape[1] - 1):
        p.addUserDebugLine([ref_state[0, i], ref_state[1, i], 0.1],
                           [ref_state[0, i + 1], ref_state[1, i + 1], 0.1],
                           [0, 0, 1], 2)  # Draw debug lines for the reference trajectory

    # Define the controller selector
    use_pure_pursuit = True  # Set to True to use Pure Pursuit, False to use Geometric MPC

    # Simulation loop
    trace_points = []  # List to store car's trajectory points
    while True:
        current_state = car.get_state()  # Get the current state of the car
        if use_pure_pursuit:
            # Use Pure Pursuit controller
            path = ref_state[:2, :].T  # Extract [x, y] points as (nTraj, 2) array
            v, w = pure_pursuit.compute_control(current_state, path)  # Compute control inputs using Pure Pursuit
            if pure_pursuit.lookahead_point is not None:
                # Update sphere position to lookahead point
                p.resetBasePositionAndOrientation(
                    car.lookahead_id,
                    [pure_pursuit.lookahead_point[0], pure_pursuit.lookahead_point[1], 0.1],
                    [0, 0, 0, 1]
                )  # Update the lookahead point visualization
                # Draw green line from car to lookahead point
                p.addUserDebugLine(
                    [current_state[0], current_state[1], 0.1],
                    [pure_pursuit.lookahead_point[0], pure_pursuit.lookahead_point[1], 0.1],
                    [0, 1, 0], 2, lifeTime=0.1
                )  # Draw a debug line to visualize the lookahead vector
        else:
            # Hide sphere when using MPC
            p.resetBasePositionAndOrientation(car.lookahead_id, [0, 0, -10], [0, 0, 0, 1]) # Hide lookahead sphere when MPC is being used
            v, w = mpc.solve(current_state, ref_trajectory_path)  # Compute control inputs using Geometric MPC

        car.apply_control(v, w)  # Apply the computed control inputs to the car
        car.step_simulation(dt=0.05)  # Step the simulation forward

        # Trace the car's path (yellow)
        trace_points.append((current_state[0], current_state[1]))  # Add current position to trace
        if len(trace_points) > 1:
            for i in range(len(trace_points) - 1):
                p.addUserDebugLine([trace_points[i][0], trace_points[i][1], 0.05],
                                   [trace_points[i + 1][0], trace_points[i + 1][1], 0.05],
                                   [1, 1, 0], 2, lifeTime=10)  # Draw debug lines for car's trajectory
        if len(trace_points) > 500:
            trace_points.pop(0)  # Keep only the last 500 points in the trace

        time.sleep(0.001)  # Add a small delay for visualization