"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: Numerical integration and propagation over time
"""

import numpy as np
from scipy.integrate import solve_ivp
from .cw import cw_dynamics
from .attitude import attitude_dynamics

def propagate_cw(initial_state, mean_motion, accel, t_span, t_eval, method = "RK45", rtol = 1e-12, atol = 1e-9):
    solution = solve_ivp(
        fun=cw_dynamics,
        t_span=t_span,
        y0=initial_state,
        method=method,
        t_eval=t_eval,
        args=(mean_motion, accel),
        rtol=rtol,
        atol=atol
    )

    if not solution.success:
        raise RuntimeError(f"CW propagation failed: {solution.message}")

    return solution


def propagate_attitude(initial_state, inertia_tensor, torque, t_span, t_eval, method="RK45", rtol=1e-12, atol=1e-9):
    solution = solve_ivp(
        fun=attitude_dynamics,
        t_span=t_span,
        y0=initial_state,
        method=method,
        t_eval=t_eval,
        args=(inertia_tensor, torque),
        rtol=rtol,
        atol=atol
    )

    if not solution.success:
        raise RuntimeError(
            f"Attitude propagation failed: {solution.message}"
        )

    return solution

def propagate_target_state(state, t_start, t_end, mean_motion, inertia_tensor):
    translation = state[0:6]
    rotation = state[6:12]

    if t_start == t_end:
        return state.copy()

    t_span = (t_start, t_end)
    t_eval = np.array([t_end])

    accel = np.zeros(3)
    torque = np.zeros(3)

    translation_solution = propagate_cw(
        translation,
        mean_motion,
        accel,
        t_span,
        t_eval
    )

    rotation_solution = propagate_attitude(
        rotation,
        inertia_tensor,
        torque,
        t_span,
        t_eval
    )

    final_translation = translation_solution.y[:, -1]
    final_rotation = rotation_solution.y[:, -1]

    final_state = np.concatenate((final_translation, final_rotation))

    return final_state


def state_transition_jacobian(state, t_start, t_end, mean_motion, inertia_tensor):
    num_states = len(state)

    if t_start == t_end:
        return np.eye(num_states)

    F = np.zeros((num_states, num_states))

    for j in range(num_states):
        if j < 3:
            epsilon = 1e-4
        elif j < 6:
            epsilon = 1e-5
        elif j < 9:
            epsilon = 1e-5
        else:
            epsilon = 1e-6

        x_plus = state.copy()
        x_minus = state.copy()

        x_plus[j] += epsilon
        x_minus[j] -= epsilon

        f_plus = propagate_target_state(
            x_plus, t_start, t_end, mean_motion, inertia_tensor
        )

        f_minus = propagate_target_state(
            x_minus, t_start, t_end, mean_motion, inertia_tensor
        )

        F[:, j] = (f_plus - f_minus) / (2 * epsilon)

    return F
