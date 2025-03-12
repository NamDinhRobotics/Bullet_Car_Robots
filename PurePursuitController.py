import numpy as np
import math


class PurePursuitController:
    def __init__(self, lookahead_distance, wheelbase, max_speed, min_speed, max_steering_angle):
        """Initialize the Pure Pursuit controller with adaptive velocity."""
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_steering_angle = max_steering_angle
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
        """Compute control inputs (linear and angular velocity) with adaptive speed."""
        current_position = current_state[:2]
        current_heading = current_state[2]
        closest_point, closest_segment_idx = self.find_closest_point(current_position, path)
        lookahead_point = self.find_lookahead_point(closest_point, closest_segment_idx, path)
        self.lookahead_point = lookahead_point  # Store for visualization
        delta = self.compute_steering_angle(current_position, current_heading, lookahead_point)

        # Scale velocity based on steering angle
        normalized_steering = abs(delta) / self.max_steering_angle  # 0.0 to 1.0
        speed = self.max_speed * (1.0 - normalized_steering * (1.0 - self.min_speed / self.max_speed))

        # Ensure speed is not lower than the minimum value
        speed = max(speed, self.min_speed)

        # Compute angular velocity with the scaled speed
        w = (speed / self.wheelbase) * math.tan(delta)
        return speed, w