
import numpy as np

from config import (
    OrbitConfig,
    TargetConfig,
    InspectorConfig,
    CameraConfig,
    SimulationConfig,
    EpisodeConfig
)


def create_episode_01():

    # Reference circular orbit
    # Mean motion is prescribed; altitude is derived.
    orbit = OrbitConfig(
        mean_motion=0.0011  # rad/s
    )

    # Noncooperative target spacecraft
    # Position and velocity are in the common CW frame.
    # Angular velocity is expressed in the target body frame.
    target = TargetConfig(
        initial_position=np.array([0.0, 0.0, 0.0]),
        initial_velocity=np.array([0.02, 0.0, 0.0]),

        initial_euler_angles=np.array([0.1, 0.2, 0.3]),
        initial_angular_velocity=0.1*np.array([0.01, 0.015, 0.02]),

        length=2.0,
        width=1.5,
        height=1.0,

        inertia_tensor=np.diag([8.0, 10.0, 12.0])
    )

    # Inspector spacecraft: 3DoF translation only
    inspector = InspectorConfig(
        initial_position=np.array([0.0, -30.0, 20.0]),
        initial_velocity=np.array([0.02, 0.0, 0.0]),
        mass=20.0
    )

    # Ideal target-tracking pinhole camera
    # Camera boresight is aligned with target COM.
    # Inspector attitude dynamics are not modeled.
    camera = CameraConfig(
        sensor_model="pinhole_target_tracking",

        focal_length_x=800.0,
        focal_length_y=800.0,

        principal_point_x=640.0,
        principal_point_y=360.0,

        image_width=1280,
        image_height=720,

        measurement_noise_std=1.0
    )

    # Numerical integration settings
    simulation = SimulationConfig(
        initial_time=0.0,
        time_step=1.0,
        final_time=600.0,

        integration_method="RK45",
        relative_tolerance=1e-12,
        absolute_tolerance=1e-9
    )

    return EpisodeConfig(
        orbit=orbit,
        target=target,
        inspector=inspector,
        camera=camera,
        simulation=simulation
    )


#vary the range, alter the initial attitude rate, translational rate