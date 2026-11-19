"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: Stores configurable episode parameters and initial conditions
"""

from dataclasses import dataclass
import numpy as np

from constants import MU_EARTH, R_EARTH

# Orbit Conifg
@dataclass
class OrbitConfig:
    mean_motion: float  # rad/s

    @property
    def semi_major_axis(self) -> float:
        return (MU_EARTH / self.mean_motion**2)**(1 / 3)

    @property
    def altitude(self) -> float:
        return self.semi_major_axis - R_EARTH


# Target Config
@dataclass
class TargetConfig:
    initial_position: np.ndarray          # m, CW frame
    initial_velocity: np.ndarray          # m/s, CW frame

    initial_euler_angles: np.ndarray      # rad
    initial_angular_velocity: np.ndarray  # rad/s, target body frame

    length: float                         # m
    width: float                          # m
    height: float                         # m

    inertia_tensor: np.ndarray            # kg*m^2, 3x3


# Inspector Config
@dataclass
class InspectorConfig:
    initial_position: np.ndarray          # m, CW frame
    initial_velocity: np.ndarray          # m/s, CW frame

    mass: float                            # kg


# Camera Config
@dataclass
class CameraConfig:
    sensor_model: str  

    # Parameters for a future calibrated image projection.
    # A conventional pinhole model is not valid over 360 degrees.
    focal_length_x: float | None = None   # pixels
    focal_length_y: float | None = None   # pixels
    principal_point_x: float | None = None  # pixels
    principal_point_y: float | None = None  # pixels

    image_width: int | None = None        # pixels
    image_height: int | None = None       # pixels

    measurement_noise_std: float | None = None


# Sim Config
@dataclass
class SimulationConfig:
    initial_time: float                   # s
    time_step: float                      # s
    final_time: float                     # s

    integration_method: str = "RK45"
    relative_tolerance: float = 1e-9
    absolute_tolerance: float = 1e-9


# Episode Config
@dataclass
class EpisodeConfig:
    orbit: OrbitConfig
    target: TargetConfig
    inspector: InspectorConfig
    camera: CameraConfig
    simulation: SimulationConfig
