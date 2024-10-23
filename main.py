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

    def get_lidar_origin(self, car_state):
        """Get lidar origin at front edge, equidistant from left and right side"""

        # given that car position is measured from rear axle
        x_offset = (self.car_height) * np.cos(car_state.theta)
        y_offset = (self.car_height) * np.sin(car_state.theta)

        # add offset to given car position to locate sensor origin
        lidar_pos = np.array([car_state.x + x_offset, car_state.y + y_offset])

        return lidar_pos

    def generate_rays(self, car_state):
        """Generate endpoints for each ray"""

        ray_endpoint = np.zeros((self.num_beams, 2))

        # obtain angles at which rays occur at and offset by car angle
        angles = np.linspace(0, self.angle_span, self.num_beams) + car_state.theta

        # get coordinates of endpoints for each ray
        for j in range(len(angles)):
            ray_endpoint[j ,0] = self.max_range * np.cos(angles[j])
            ray_endpoint[j ,1] = self.max_range * np.sin(angles[j])

        return ray_endpoint

    def boundary_intersection(self, p0, p1, p2, p3):
        """Get intersection coordinates of the ray and the track edges (line segment)"""

        # p0 and p1 define start and end of LIDAR ray
        # p2 and p3 define start and end of boundary segment
        s1 = p1 - p0
        s2 = p3 - p2

        # calculate determinant of sys of eq
        det = -s1[0]*s2[1] + s2[0]*s1[1]

        # if det = 0, parallel, no intersection; return max_range value
        if det == 0: 
            return self.max_range

        # calculate t and u values at which the points intersect
        t = ( -s2[1]*(p2[0] - p0[0]) + s2[0]*(p2[1] - p0[1]) )/det
        u = ( s1[0]*(p2[1] - p2[1]) + s1[1]*(p2[0] - p0[0]) )/det

        # calculate intersection point
        if 0 <= u <= 1 and t >= 0:
            POI = p0 + t*s1
            return POI          # POI has x and y coordinates
        
        # no intersection, return max_range
        return self.max_range

    def obstacle_intersection(self, p0, p1, obstacle):
        """Get intersection coordinates of the ray and the obstacles"""

        s = p1 - p0

        # unpack obstacle center coordinates and radius
        h = obstacle[0]; k = obstacle[1]; r = obstacle[2]

        # calculate quadratic constants
        A = s[0]^2 + s[1]^2
        B = 2*( (p0[0] - h)*s[0] + (p0[1] - k)*s[1] )
        C = ( (p0[0] - h)**2 + (p0[1] - k)**2 - r**2 )

        discriminant = B**2 - 4*A*C

        if discriminant < 0:        # if negative sqrt, no intersection, return max_range value
            return self.max_range
        if discriminant == 0:
            t = -B / (2*A)
        if discriminant > 0:
            t1 = (-B + np.sqrt(discriminant)) / (2*A)
            t2 = (-B - np.sqrt(discriminant)) / (2*A)
            t = min(t1, t2)     # smaller t value intersects obstacle first
        
        POI = p0 + t*s
        return POI      # POI has x and y coordinates

    def calc_distance(lidar_pos, POI):
        """Calculate distance between two points"""

        distance = np.sqrt( (POI[0] - lidar_pos[0])**2 + (POI[1] - lidar_pos[1])**2 )
        return distance

    def get_distances(self, car_state: CarState, track_config: TrackConfig) -> np.ndarray:
        """Calculate the min distance between lidar origin and obstacles"""

        # get lidar coordinates
        lidar_pos = self.get_lidar_origin(self, car_state)

        # generate rays
        ray_endpoint = self.generate_rays(self, car_state)

        # initialize bins with max_range values
        dist = np.ones(self.num_beams) * self.max_range

        for i in range(len(dist)):      # iterate over each ray
            ray = ray_endpoint[i, :]

            # check for intersecting points with inner boundary
            for q in range(len(track_config.inner_boundary) - 1):   
                POI = self.boundary_intersection(self, lidar_pos, ray, track_config.inner_boundary[q, :], 
                                            track_config.inner_boundary[q + 1, :])
                distance = self.calc_distance(lidar_pos, POI)

                # take the minimum distance
                dist[i] = min(dist[i], distance)

            # check for intersecting points with outer boundary
            for m in track_config.outer_boundary:
                POI = self.boundary_intersection(self, lidar_pos, ray, track_config.outer_boundary[m, :], 
                                            track_config.outer_boundary[m + 1, :])                
                distance = self.calc_distance(lidar_pos, POI)

                # take the minimum distance
                dist[i] = min(dist[i], distance)
            
            # check for intersecting points with obstacles (list)
            for obst in track_config.obstacles:
                POI = self.obstacle_intersection(self, lidar_pos, ray, obst)               
                distance = self.calc_distance(lidar_pos, POI)

                # take the minimum distance
                dist[i] = min(dist[i], distance)

        return dist

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