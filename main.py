import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.transforms import Affine2D
from typing import List, Tuple, Dict
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
    """Generate inner and outer track boundaries from centerline points."""
    points = np.array(centerline_points)
    inner_boundary = []
    outer_boundary = []

    # Convert points to numpy array for easier manipulation
    points = np.array(points)

    for i in range(len(points)):
        # Get current point and next point (wrap around to first point if at end)
        current = points[i]
        next_point = points[(i + 1) % len(points)]

        # Calculate direction vector
        direction = next_point - current
        direction = direction / np.linalg.norm(direction)

        # Calculate normal vector (rotate direction 90 degrees)
        normal = np.array([-direction[1], direction[0]])

        # Generate inner and outer points
        inner_point = current - normal * (track_width / 2)
        outer_point = current + normal * (track_width / 2)

        inner_boundary.append(inner_point)
        outer_boundary.append(outer_point)
    
    return np.array(inner_boundary), np.array(outer_boundary)

class LIDARSimulator:
    """Placeholder for the LIDAR simulation component"""
    def __init__(self, num_beams: int = 180, max_range: float = 10.0, angle_span: float = np.pi):
        self.num_beams = num_beams
        self.max_range = max_range
        self.angle_span = angle_span

    def get_distances(self, car_state: CarState, track_config: TrackConfig) -> np.ndarray:
        """Placeholder for ray tracing function"""
        # Returns array of distances for each beam
        return np.ones(self.num_beams) * self.max_range

class CarController:
    def __init__(self, gap_threshold: float = 3.0, num_beams: int = 180, angle_span: float = np.pi):
        self.gap_threshold = gap_threshold
        self.num_beams = num_beams
        self.angle_span = angle_span

    def get_steering_angle(self, lidar_data: np.ndarray) -> float:
        """Steer the car towards the maximum gap in the LIDAR data"""
        # Get angle array for LIDAR data
        angle_array = np.arange(start=-self.angle_span/2, stop=self.angle_span/2, step=self.angle_span/(self.num_beams-1))
        if lidar_data.shape != angle_array.shape:
            raise ValueError("LIDAR data shape must match angle array shape")
        
        # Split LIDAR data into segments based on gap threshold
        free_segments = np.split(lidar_data, np.where(lidar_data < self.gap_threshold)[0])

        # Find the longest free segment
        max_gap = max(free_segments, key=len, default=[])

        # Find the center index of the longest free segment
        start_idx = lidar_data.tolist().index(max_gap[0])
        best_idx = int(start_idx + len(max_gap) / 2)
        return angle_array[best_idx]

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
        new_theta = theta + self.velocity / self.car_length * np.tan(steering_angle) * self.dt + np.random.normal(0, self.theta_std)
        if steering_angle == 0:
            new_x = x + self.velocity * np.cos(theta) * self.dt + np.random.normal(0, self.x_std)
            new_y = y + self.velocity * np.sin(theta) * self.dt + np.random.normal(0, self.y_std)
        else:
            new_x = x + self.car_length / np.tan(steering_angle) * (np.sin(theta + new_theta) - np.sin(theta)) + np.random.normal(0, self.x_std)
            new_y = y + self.car_length / np.tan(steering_angle) * (np.cos(theta) - np.cos(theta + new_theta)) + np.random.normal(0, self.y_std)

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
        # Create a polygon that represents the track
        track_polygon = np.vstack([
            self.track_config.outer_boundary,
            np.flipud(self.track_config.inner_boundary)
        ])

        # Draw the track as a filled polygon
        track = Polygon(track_polygon, facecolor='lightgray', edgecolor='black')
        self.ax.add_patch(track)

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
    # Define centerline points for the track
    centerline_points = [(0, 0), (3, 3), (3, -3), (-3, -3), (-3, 3), (0, 0)]
    track_width = 1.0  # Define track width
    
    # Generate track boundaries
    inner_boundary, outer_boundary = generate_track_boundaries(centerline_points, track_width)
    
    # Create track configuration
    track_config = TrackConfig(
        bounds=((-5, 5), (-5, 5)),
        obstacles=[(2, 2, 0.5), (-2, -2, 0.5), (2, -2, 0.3)],
        track_width=track_width,
        inner_boundary=inner_boundary,
        outer_boundary=outer_boundary
    )

    # Initialize components
    visualizer = TrackVisualizer(track_config)
    lidar = LIDARSimulator()
    controller = CarController()
    kinematics = CarKinematics()

    # Initial car state
    car_state = CarState(0, 0, 0)

    def init():
        """Initialize animation"""
        return (visualizer.car,)

    def update(frame):
        """Update animation frame"""
        nonlocal car_state

        # Get LIDAR data
        lidar_data = lidar.get_distances(car_state, track_config)

        # Get steering decision
        steering_angle = controller.get_steering_angle(lidar_data)

        # Update car state
        car_state = kinematics.update_state(car_state, steering_angle, 0.05)

        # Update visualization
        visualizer.update_visualization(car_state)
        return (visualizer.car,)

    def savemp4(ani):
        Writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("./animation.mp4", writer=Writer)

    # Create animation
    ani = FuncAnimation(visualizer.fig, update, init_func=init,
                       frames=500, interval=50, blit=True)

    plt.show()
    savemp4(ani)

if __name__ == "__main__":
    main()