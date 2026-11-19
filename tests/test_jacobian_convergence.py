"""
Author: Isabella Peressim
Date Created: 11/18/2026
Description: Numerical convergence checks for Episode 1 camera Jacobians.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episodes.episode_01 import create_episode_01
from spacecraft.geometry import create_box_vertices
from dynamics.propagation import propagate_cw
from sensors.measurements import measurement_function
from estimation.information import (
    measurement_jacobian,
    propagated_measurement,
    propagated_measurement_jacobian,
)

BASE_STEPS = np.array([1e-4] * 3 + [1e-5] * 3 + [1e-5] * 3 + [1e-6] * 3)


def finite_difference_jacobian(func, state, step_factor):
    y_ref, visible_ref = func(state)
    visible_ref = np.asarray(visible_ref, dtype=bool)
    H = np.zeros((len(y_ref), len(state)))
    for j in range(len(state)):
        h = BASE_STEPS[j] * step_factor
        plus = state.copy()
        minus = state.copy()
        plus[j] += h
        minus[j] -= h
        y_plus, vis_plus = func(plus)
        y_minus, vis_minus = func(minus)
        if not (np.array_equal(vis_plus, visible_ref) and
                np.array_equal(vis_minus, visible_ref)):
            raise AssertionError(f"Visibility changed for component {j}")
        H[:, j] = (y_plus - y_minus) / (2 * h)
    if not np.all(np.isfinite(H)):
        raise AssertionError("Jacobian contains nonfinite values")
    return H


def compare_jacobians(label, func, implementation):
    H4 = finite_difference_jacobian(func, state, 4.0)
    H2 = finite_difference_jacobian(func, state, 2.0)
    H1 = finite_difference_jacobian(func, state, 1.0)
    Himpl = implementation(state)
    if Himpl.shape != (16, 12):
        raise AssertionError(f"Unexpected Jacobian shape: {Himpl.shape}")
    scale = max(np.linalg.norm(H1), 1.0)
    coarse_change = np.linalg.norm(H4 - H2) / scale
    fine_change = np.linalg.norm(H2 - H1) / scale
    implementation_error = np.linalg.norm(Himpl - H1) / scale
    print(f"{label}")
    print(f"  Relative change (4x -> 2x step): {coarse_change:.3e}")
    print(f"  Relative change (2x -> 1x step): {fine_change:.3e}")
    print(f"  Implementation vs independent FD: {implementation_error:.3e}")
    if fine_change > 1e-3:
        raise AssertionError("Fine-step Jacobians disagree by more than 0.1%")
    if implementation_error > 1e-6:
        raise AssertionError("Implementation differs from independent finite differences")

episode = create_episode_01()
n = episode.orbit.mean_motion
state = np.concatenate((
    episode.target.initial_position,
    episode.target.initial_velocity,
    episode.target.initial_euler_angles,
    episode.target.initial_angular_velocity,
)).astype(float)
inspector_initial = np.concatenate((
    episode.inspector.initial_position,
    episode.inspector.initial_velocity,
)).astype(float)
vertices = create_box_vertices(
    episode.target.length, episode.target.width, episode.target.height
)

for t in (0.0, 25.0, 100.0):
    if t == 0:
        inspector_pos = inspector_initial[:3]
    else:
        solution = propagate_cw(
            inspector_initial, n, np.zeros(3),
            (0.0, t), np.array([t])
        )
        inspector_pos = solution.y[:3, -1]

    if t == 0:
        def model(x):
            return measurement_function(x, inspector_pos, t, n, vertices, episode.camera)

        def implementation(x):
            return measurement_jacobian(x, inspector_pos, t, n, vertices, episode.camera)
    else:
        def model(x):
            return propagated_measurement(
                x, inspector_pos, t, n, vertices, episode.camera,
                episode.target.inertia_tensor
            )

        def implementation(x):
            return propagated_measurement_jacobian(
                x, inspector_pos, t, n, vertices, episode.camera,
                episode.target.inertia_tensor
            )

    compare_jacobians(f"t = {t:g} s", model, implementation)

print("PASS: Episode 1 Jacobian finite-difference convergence")
