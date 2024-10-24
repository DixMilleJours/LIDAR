import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.transforms import Affine2D
from typing import List, Optional, Tuple, Dict
import dataclasses

# Explanation:
#
# The class <-> component relationship is described here:
# LIDARSimulator -> ray casting
# CarController -> given lidar data, return steering angle
# CarKinematics -> kinematic model
# TrackVisualizer -> draw track, car, obstacles, etc.

# Data structures for the simulation
@dataclasses.dataclass
class CarState:
    x: float  # x position
    y: float  # y position
    theta: float  # current heading angle in radians --- do we need this?
    steering_angle: float = 0.0  # current steering angle

@dataclasses.dataclass
class TrackConfig:
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    obstacles: List[Tuple[float, float, float]]
    track_width: float  # Added track width parameter
    inner_boundary: np.ndarray  # Array of (x, y) points for inner boundary
    outer_boundary: np.ndarray  # Array of (x, y) points for outer boundary

def generate_track_boundaries(centerline_points: List[Tuple[float, float]], track_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate inner and outer track boundaries from centerline points with consistent width."""
    points = np.array(centerline_points)
    inner_boundary = []
    outer_boundary = []
    
    # Ensure the track is closed by appending the first point if needed
    if not np.array_equal(points[0], points[-1]):
        points = np.append(points, [points[0]], axis=0)

    for i in range(len(points) - 1):
        # Get current point and next point
        current = points[i]
        next_point = points[i + 1]
        
        # Get previous point (wrap around to last point if at start)
        prev_point = points[i - 1] if i > 0 else points[-2]
        
        # Calculate direction vectors
        dir_to_next = next_point - current
        dir_to_prev = current - prev_point
        
        # Normalize direction vectors
        dir_to_next = dir_to_next / np.linalg.norm(dir_to_next)
        dir_to_prev = dir_to_prev / np.linalg.norm(dir_to_prev)
        
        # Calculate average direction for smooth corners
        avg_direction = dir_to_next + dir_to_prev
        avg_direction = avg_direction / np.linalg.norm(avg_direction)
        
        # Calculate normal vector (rotate average direction 90 degrees)
        normal = np.array([-avg_direction[1], avg_direction[0]])
        
        # Generate inner and outer points
        inner_point = current - normal * (track_width / 2)
        outer_point = current + normal * (track_width / 2)
        
        inner_boundary.append(inner_point)
        outer_boundary.append(outer_point)
    
    # Add closing points
    inner_boundary.append(inner_boundary[0])
    outer_boundary.append(outer_boundary[0])
    
    return np.array(inner_boundary), np.array(outer_boundary)

@dataclasses.dataclass
class LIDARReading:
    distances: np.ndarray  # Array of distances for each beam
    angles: np.ndarray    # Array of angles for each beam
    hit_points: np.ndarray  # Array of (x,y) coordinates where each beam hits

class LIDARSimulator:
    """
    LIDAR simulation component that performs ray casting to detect obstacles and track boundaries.
    Simulates a LIDAR sensor mounted on the front of the car.
    """
    def __init__(self, 
                 num_beams: int = 180,
                 max_range: float = 10.0,
                 angle_span: float = np.pi,
                 car_length: float = 5.0,
                 car_width: float = 2.5,
                 mount_offset: float = 0.0):  # Offset from car front center
        """
        Initialize the LIDAR simulator.
        
        Args:
            num_beams: Number of LIDAR beams
            max_range: Maximum detection range
            angle_span: Angular span of the sensor in radians
            car_length: Length of the car for mounting position
            car_width: Width of the car for mounting position
            mount_offset: Offset from car center (positive is right, negative is left)
        """
        self.num_beams = num_beams
        self.max_range = max_range
        self.angle_span = angle_span
        self.car_length = car_length
        self.car_width = car_width
        self.mount_offset = mount_offset
        
        # Pre-calculate beam angles for efficiency
        self.beam_angles = np.linspace(-angle_span/2, angle_span/2, num_beams)
    
    def get_sensor_position(self, car_state: 'CarState') -> np.ndarray:
        """Calculate LIDAR sensor position based on car state."""
        # Calculate offset from car's rear axle to sensor mounting point
        x_offset = self.car_length * np.cos(car_state.theta)
        y_offset = self.car_length * np.sin(car_state.theta)
        
        # Add lateral offset if sensor is not mounted at center
        if self.mount_offset != 0:
            lateral_x = -self.mount_offset * np.sin(car_state.theta)
            lateral_y = self.mount_offset * np.cos(car_state.theta)
            x_offset += lateral_x
            y_offset += lateral_y
            
        return np.array([
            car_state.x + x_offset,
            car_state.y + y_offset
        ])

    def cast_ray(self, 
                 origin: np.ndarray,
                 angle: float,
                 track_config: 'TrackConfig') -> Tuple[float, np.ndarray]:
        """
        Cast a single ray and find the closest intersection point.
        
        Returns:
            Tuple of (distance, hit_point)
        """
        # Calculate ray endpoint at max range
        direction = np.array([np.cos(angle), np.sin(angle)])
        end_point = origin + direction * self.max_range
        
        closest_dist = self.max_range
        closest_point = end_point.copy()
        
        # Check track boundary intersections
        for boundary in [track_config.inner_boundary, track_config.outer_boundary]:
            for i in range(len(boundary) - 1):
                intersection = self._line_intersection(
                    origin, end_point,
                    boundary[i], boundary[i + 1]
                )
                if intersection is not None:
                    dist = np.linalg.norm(intersection - origin)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point = intersection

        # Check obstacle intersections
        for obstacle in track_config.obstacles:
            intersection = self._circle_intersection(
                origin, end_point, 
                np.array([obstacle[0], obstacle[1]]), obstacle[2]
            )
            if intersection is not None:
                dist = np.linalg.norm(intersection - origin)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_point = intersection
                    
        return closest_dist, closest_point

    def get_readings(self, car_state: 'CarState', track_config: 'TrackConfig') -> LIDARReading:
        """Get LIDAR readings for the current car state and track configuration."""
        sensor_pos = self.get_sensor_position(car_state)
        
        # Initialize arrays for storing results
        distances = np.zeros(self.num_beams)
        hit_points = np.zeros((self.num_beams, 2))
        
        # Calculate absolute angles by adding car's heading to relative angles
        absolute_angles = self.beam_angles + car_state.theta
        
        # Cast rays for each beam
        for i, angle in enumerate(absolute_angles):
            distances[i], hit_points[i] = self.cast_ray(sensor_pos, angle, track_config)
        
        return LIDARReading(
            distances=distances,
            angles=self.beam_angles,  # Return relative angles for easier processing
            hit_points=hit_points
        )

    def _line_intersection(self,
                          ray_start: np.ndarray,
                          ray_end: np.ndarray,
                          line_start: np.ndarray,
                          line_end: np.ndarray) -> Optional[np.ndarray]:
        """Calculate intersection between ray and line segment using vector cross product."""
        p0, p1 = ray_start, ray_end
        p2, p3 = line_start, line_end
        
        s1 = p1 - p0
        s2 = p3 - p2
        
        det = np.cross(s1, s2)
        
        if abs(det) < 1e-10:  # Lines are parallel
            return None
            
        s = np.cross(p2 - p0, s2) / det
        t = np.cross(p2 - p0, s1) / det
        
        if 0 <= s <= 1 and 0 <= t <= 1:
            return p0 + s * s1
        
        return None

    def _circle_intersection(self,
                           ray_start: np.ndarray,
                           ray_end: np.ndarray,
                           circle_center: np.ndarray,
                           circle_radius: float) -> Optional[np.ndarray]:
        """Calculate intersection between ray and circle using quadratic equation."""
        direction = ray_end - ray_start
        direction = direction / np.linalg.norm(direction)
        
        oc = ray_start - circle_center
        
        a = 1  # Normalized direction vector
        b = 2 * np.dot(direction, oc)
        c = np.dot(oc, oc) - circle_radius * circle_radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
            
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        
        # Get smallest positive intersection
        t = float('inf')
        if t1 >= 0:
            t = min(t, t1)
        if t2 >= 0:
            t = min(t, t2)
            
        if t == float('inf'):
            return None
            
        return ray_start + direction * t

class CarController:
    def __init__(self, gap_threshold: float = 3.0, num_beams: int = 180, angle_span: float = np.pi, ratio: float = 0.95):
        self.gap_threshold = gap_threshold
        self.num_beams = num_beams
        self.angle_span = angle_span  # This should match LIDAR's angle_span
        self.ratio = ratio

    def get_steering_angle(self, lidar_data: np.ndarray) -> float:
        # print(f"lidar_data: {lidar_data}")
        """Steer the car towards the maximum gap in the LIDAR data"""
        # Create angle array matching LIDAR's angle distribution
        angle_array = np.linspace(-self.angle_span/2, self.angle_span/2, self.num_beams)
        
        # Add shape check with more informative error message
        if lidar_data.shape[0] != angle_array.shape[0]:
            raise ValueError(f"LIDAR data shape {lidar_data.shape} must match angle array shape {angle_array.shape}")

        # Split LIDAR data into segments based on gap threshold
        gaps = lidar_data > self.gap_threshold
        gap_indices = np.where(gaps)[0]
        
        if len(gap_indices) == 0:
            # If no gaps found, return neutral steering
            return 0.0
        
        # Find the longest continuous gap
        gap_diff = np.diff(gap_indices)
        gap_starts = np.concatenate([[0], np.where(gap_diff > 1)[0] + 1])
        gap_ends = np.concatenate([np.where(gap_diff > 1)[0], [len(gap_indices) - 1]])
        
        gap_lengths = gap_ends - gap_starts + 1
        longest_gap_idx = np.argmax(gap_lengths)
        start_idx = gap_indices[gap_starts[longest_gap_idx]]
        end_idx = gap_indices[gap_ends[longest_gap_idx]]
        
        # Find the furthest point in the gap
        best_idx = np.argmax(lidar_data[start_idx:end_idx+1]) + start_idx

        # Calculate center of the longest gap
        center_idx = (start_idx + end_idx) // 2

        # Use a weighted average of the center and best points
        steering_angle = angle_array[int(self.ratio*center_idx + (1-self.ratio)*best_idx)]
        return steering_angle

class CarKinematics:
    def __init__(self, car_length: float = 0.4, velocity: float = 1.0, dt: float = 0.05, x_std: float = 0.01, y_std: float = 0.01, theta_std: float = 0.01):
        self.car_length = car_length
        self.velocity = velocity
        self.dt = dt
        self.x_std = x_std
        self.y_std = y_std
        self.theta_std = theta_std
        self.state_history = []

    def update_state(self, current_state: CarState, steering_angle: float) -> CarState:
        """Update the car state using the kinematic model"""
        # Store old state in state history
        self.state_history.append(current_state)

        # Avoid instability for very small steering angles
        if abs(steering_angle) < 1e-2:
            steering_angle = 0.0

        # Update state with Gaussian noise
        x, y, theta = current_state.x, current_state.y, current_state.theta
        new_x = x + self.velocity * np.cos(theta) * self.dt + np.random.normal(0, self.x_std)
        new_y = y + self.velocity * np.sin(theta) * self.dt + np.random.normal(0, self.y_std)
        new_theta = theta + self.velocity / self.car_length * np.tan(steering_angle) * self.dt + np.random.normal(0, self.theta_std)
        # new_theta = theta + self.velocity / self.car_length * np.tan(steering_angle) * self.dt + np.random.normal(0, self.theta_std)
        # if steering_angle == 0:
        #     new_x = x + self.velocity * np.cos(theta) * self.dt + np.random.normal(0, self.x_std)
        #     new_y = y + self.velocity * np.sin(theta) * self.dt + np.random.normal(0, self.y_std)
        # else:
        #     new_x = x + self.car_length / np.tan(steering_angle) * (np.sin(theta + new_theta) - np.sin(theta)) + np.random.normal(0, self.x_std)
        #     new_y = y + self.car_length / np.tan(steering_angle) * (np.cos(theta) - np.cos(theta + new_theta)) + np.random.normal(0, self.y_std)

        return CarState(new_x, new_y, new_theta, steering_angle)

class TrackVisualizer:
    def __init__(self, track_config: TrackConfig):
        self.track_config = track_config
        self.setup_visualization()

    def setup_visualization(self):
        """Initialize the visualization"""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        bounds = self.track_config.bounds
        self.ax.set_xlim(bounds[0])
        self.ax.set_ylim(bounds[1])
        self.ax.set_aspect('equal')

        # Create car rectangle
        self.car_width = 0.4
        self.car_height = 0.2
        self.car = Rectangle((-self.car_width/2, -self.car_height/2),
                           self.car_width, self.car_height,
                           facecolor='red', edgecolor='darkred')
        self.ax.add_patch(self.car)

        # Draw track and obstacles
        self.draw_track()
        self.draw_obstacles()

    def draw_track(self):
        """Draw the track with filled area between boundaries"""
        # Draw outer boundary
        outer_track = Polygon(self.track_config.outer_boundary,
                            facecolor='lightgray',
                            edgecolor='black',
                            fill=True)
        self.ax.add_patch(outer_track)
        
        # Draw inner boundary (the hole)
        inner_track = Polygon(self.track_config.inner_boundary,
                            facecolor='white',
                            edgecolor='black',
                            fill=True)
        self.ax.add_patch(inner_track)

    def draw_obstacles(self):
        """Draw the obstacles"""
        for x, y, radius in self.track_config.obstacles:
            obstacle = Circle((x, y), radius, facecolor='gray', edgecolor='black')
            self.ax.add_patch(obstacle)

    def update_visualization(self, car_state: CarState):
        """Update the visualization for the current state"""
        # Update car position and orientation
        transform = Affine2D() \
            .rotate(car_state.theta) \
            .translate(car_state.x, car_state.y) \
            .get_matrix()
        self.car.set_transform(self.ax.transData + plt.matplotlib.transforms.Affine2D(transform))

def main():
    """Main simulation function with visualization of LIDAR readings"""
    # Define centerline points for a more rounded track
    centerline_points = [
        (0, 0),
        (2, 2),
        (4, 2),
        (5, 0),
        (4, -2),
        (2, -2),
        (0, 0)
    ]
    track_width = 1.5
    
    # Generate track boundaries
    inner_boundary, outer_boundary = generate_track_boundaries(centerline_points, track_width)
    
    # Create track configuration with obstacles inside the track
    track_config = TrackConfig(
        bounds=((-8, 8), (-8, 8)),
        obstacles=[
            (2, 0, 0.3),    # Center obstacle
            (3, 1, 0.2),    # Upper right obstacle
            (3, -1, 0.2),   # Lower right obstacle
            (1, 1, 0.2),    # Upper left obstacle
        ],
        track_width=track_width,
        inner_boundary=inner_boundary,
        outer_boundary=outer_boundary
    )

    # Initialize components with tuned parameters
    car_length = 0.1
    visualizer = TrackVisualizer(track_config)
    lidar = LIDARSimulator(
        num_beams=180,
        angle_span=np.pi * 2/3,  # 120 degree field of view
        max_range=5.0,
        car_length=car_length,
        car_width=0.2
    )
    controller = CarController(
        gap_threshold=0.25,
        num_beams=180,
        angle_span=np.pi * 2/3  # Match LIDAR's angle span
    )
    kinematics = CarKinematics(
        car_length=car_length,
        velocity=0.8,  # Reduced velocity for better control
        dt=0.05,
        x_std=0.001,   # Reduced noise for smoother motion
        y_std=0.001,
        theta_std=0.001
    )

    # Initial car state - start at the beginning of the track
    initial_heading = np.arctan2(
        centerline_points[1][1] - centerline_points[0][1],
        centerline_points[1][0] - centerline_points[0][0]
    )
    car_state = CarState(centerline_points[0][0], centerline_points[0][1], initial_heading)

    # Create lists to store trajectory and LIDAR data for visualization
    trajectory = [(car_state.x, car_state.y)]
    current_lidar_points = None
    
    def init():
        """Initialize animation"""
        # Create empty line objects for trajectory and LIDAR visualization
        trajectory_line, = visualizer.ax.plot([], [], 'g-', alpha=0.5, label='Trajectory')
        lidar_lines = [visualizer.ax.plot([], [], 'r-', alpha=0.1)[0] 
                      for _ in range(lidar.num_beams)]
        
        visualizer.ax.legend()
        return (visualizer.car, trajectory_line, *lidar_lines)

    def update(frame):
        """Update animation frame"""
        nonlocal car_state, trajectory, current_lidar_points
        
        # Get LIDAR readings
        lidar_reading = lidar.get_readings(car_state, track_config)

        # Get steering decision
        steering_angle = controller.get_steering_angle(lidar_reading.distances)

        # Update car state
        car_state = kinematics.update_state(car_state, steering_angle)

        # Update trajectory
        trajectory.append((car_state.x, car_state.y))

        # Update visualization
        visualizer.update_visualization(car_state)

        # Update trajectory line
        trajectory_x, trajectory_y = zip(*trajectory)
        trajectory_line = visualizer.ax.get_lines()[0]
        trajectory_line.set_data(trajectory_x, trajectory_y)

        # Update LIDAR visualization
        sensor_pos = lidar.get_sensor_position(car_state)
        lidar_lines = visualizer.ax.get_lines()[1:]

        for i, (line, hit_point) in enumerate(zip(lidar_lines, lidar_reading.hit_points)):
            line.set_data([sensor_pos[0], hit_point[0]],
                         [sensor_pos[1], hit_point[1]])

        return (visualizer.car, trajectory_line, *lidar_lines)

    def save_animation(ani):
        """Save animation to file with appropriate settings"""
        Writer = FFMpegWriter(
            fps=30,
            metadata=dict(artist='LIDAR Simulation'),
            bitrate=2000
        )
        ani.save("./lidar_simulation.mp4", writer=Writer)

    # Create animation with adjusted parameters
    ani = FuncAnimation(
        visualizer.fig,
        update,
        init_func=init,
        frames=500,
        interval=20,  # Smaller interval for smoother animation
        blit=True,
        repeat=False  # Don't repeat the animation
    )

    # Set up the plot
    visualizer.ax.set_title('LIDAR-based Navigation Simulation')
    visualizer.ax.grid(True, alpha=0.3)

    # Show plot and save animation
    plt.show()
    save_animation(ani)

if __name__ == "__main__":
    main()