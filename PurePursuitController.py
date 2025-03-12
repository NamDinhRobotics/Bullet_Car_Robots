import numpy as np
import math

class PurePursuitController:
    def __init__(self, min_lookahead, max_lookahead, wheelbase, max_speed, min_speed, max_steering_angle):
        """Initialize the Pure Pursuit controller with adaptive velocity and lookahead distance."""
        self.lookahead_distance = min_lookahead
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.wheelbase = wheelbase
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_steering_angle = max_steering_angle
        self.lookahead_point = None  # To store the lookahead point for visualization
        self.prev_delta = 0.0  # Previous steering angle, initialized to 0

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
        """Compute control inputs (linear and angular velocity) with adaptive speed and lookahead distance."""

        # Extract current state
        current_position = current_state[:2]
        current_heading = current_state[2]

        # Find closest and lookahead points
        closest_point, closest_segment_idx = self.find_closest_point(current_position, path)
        lookahead_point = self.find_lookahead_point(closest_point, closest_segment_idx, path)
        self.lookahead_point = lookahead_point

        # Compute steering angle
        delta = self.compute_steering_angle(current_position, current_heading, lookahead_point)

        # Scale velocity based on current steering angle (matching C++ logic)
        normalized_steering = abs(delta) / self.max_steering_angle  # 0.0 to 1.0
        speed = self.max_speed * (1.0 - normalized_steering * (1.0 - self.min_speed / self.max_speed))
        speed = max(speed, self.min_speed)  # Ensure speed is not below min_speed

        # Adapt lookahead_distance based on previous steering angle
        # normalized_steering = abs(delta) / self.max_steering_angle  # 0.0 to 1.0
        lookahead_distance_tmp = self.max_lookahead * (1.0 - normalized_steering * (1.0 -self.min_lookahead /self.max_lookahead ))
        self.lookahead_distance = max(lookahead_distance_tmp, self.min_lookahead)

        # Compute angular velocity
        w = (speed / self.wheelbase) * math.tan(delta)

        # Update previous steering angle for next iteration
        self.prev_delta = delta

        return speed, w