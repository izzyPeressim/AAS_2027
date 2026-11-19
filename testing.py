import numpy as np
from episodes.episode_01 import create_episode_01
from spacecraft.geometry import create_box_vertices, vertices_in_cw_frame
from sensors.camera import project_points

config = create_episode_01()

target = config.target
inspector = config.inspector

vertices_body = create_box_vertices(
    target.length,
    target.width,
    target.height
)

vertices_cw = vertices_in_cw_frame(
    vertices_body,
    target.initial_position,
    target.initial_euler_angles,
    config.simulation.initial_time,
    config.orbit.mean_motion
)

pixels, visible = project_points(
    vertices_cw,
    inspector.initial_position,
    target.initial_position,
    config.camera
)

print("Pixel coordinates:")
print(pixels)

print("\nVisibility:")
print(visible)

print("\nNumber of visible vertices:", np.sum(visible))

from sensors.measurements import measurement_function

target_state = np.concatenate((
    target.initial_position,
    target.initial_velocity,
    target.initial_euler_angles,
    target.initial_angular_velocity
))

y, visible = measurement_function(
    target_state,
    inspector.initial_position,
    config.simulation.initial_time,
    config.orbit.mean_motion,
    vertices_body,
    config.camera
)

print("Measurement vector:", y)
print("Measurement shape:", y.shape)
print("Visibility:", visible)

from estimation.information import measurement_jacobian

H = measurement_jacobian(
    target_state,
    inspector.initial_position,
    config.simulation.initial_time,
    config.orbit.mean_motion,
    vertices_body,
    config.camera,
)

assert visible.all(), "Expected all eight corners in frame"
assert H.shape == (16, 12), H.shape
assert np.isfinite(H).all()
np.testing.assert_allclose(H[:, 3:6], 0.0, atol=1e-8)
np.testing.assert_allclose(H[:, 9:12], 0.0, atol=1e-8)
print("Jacobian column norms:", np.linalg.norm(H, axis=0))

from estimation.information import information_gain

prior_std = np.array([
    1.0, 1.0, 1.0,       # Position [m]
    0.1, 0.1, 0.1,       # Velocity [m/s]
    0.1, 0.1, 0.1,       # Euler angles [rad]
    0.01, 0.01, 0.01    # Angular velocity [rad/s]
])

IG_baseline = information_gain(H, prior_std, 1.0)
IG_noisy = information_gain(H, prior_std, 5.0)
IG_uncertain = information_gain(H, 2.0 * prior_std, 1.0)

print("\n--- INFORMATION GAIN TEST ---")
print(f"Baseline IG:          {IG_baseline:.6f}")
print(f"Higher noise IG:      {IG_noisy:.6f}")
print(f"Higher uncertainty IG:{IG_uncertain:.6f}")

assert np.isfinite(IG_baseline)
assert IG_baseline >= 0.0
assert IG_noisy < IG_baseline
assert IG_uncertain > IG_baseline

print("PASS: Single-measurement information gain tests")
