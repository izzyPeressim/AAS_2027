"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: Measurement Jacobians and information-gain calculations
"""

import numpy as np
from sensors.measurements import measurement_function
from dynamics.propagation import propagate_cw, propagate_attitude

def appropriate_step_size(j):
    if j < 3:
        return 1e-4
    elif j < 6:
        return 1e-5
    elif j < 9:
        return 1e-5
    else:
        return 1e-6

def measurement_jacobian(target_state, inspector_pos, t, n, target_geom, camera_config):
    y0, visible0 = measurement_function(target_state, inspector_pos, t, n, target_geom, camera_config)

    num_meas = len(y0)
    num_states = len(target_state)

    H = np.zeros((num_meas, num_states))

    for j in range(num_states):
        epsilon = appropriate_step_size(j)

        x_plus = target_state.copy()
        x_minus = target_state.copy()

        x_plus[j] += epsilon
        x_minus[j] -= epsilon

        y_plus, visible_plus = measurement_function(x_plus, inspector_pos, t, n, target_geom, camera_config)
        y_minus, visible_minus = measurement_function(x_minus, inspector_pos, t, n, target_geom, camera_config)

        if not np.array_equal(visible_plus, visible0) or not np.array_equal(visible_minus, visible0):
            raise ValueError("Feature visibility changed during Jacobian calculation")

        H[:, j] = (y_plus - y_minus) / (2 * epsilon)

    return H

def information_gain(H, prior_std, sigma_pixel):
    scaled_H = H * prior_std / sigma_pixel

    num_target_states = H.shape[1]
    I =np.eye(num_target_states)

    info_matrix = I + scaled_H.T @ scaled_H
    sign, log_det = np.linalg.slogdet(info_matrix)

    if sign <= 0:
        raise ValueError("Information matrix must have a positive determinant")

    return 0.5 * log_det

def propagated_measurement(init_target_state, inspector_pos, t, n, target_geom, camera_config, inertia_tensor):
    init_translation = init_target_state[0:6]
    init_rotation = init_target_state[6:12]

    if t == 0:
        propagated_state = init_target_state.copy()
    else:
        t_span = (0.0, t)
        t_eval = np.array([0.0, t])

        translational_acceleration = np.zeros(3)
        external_torque = np.zeros(3)

        translation_solution = propagate_cw(init_translation, n, 
                                translational_acceleration, t_span, t_eval)

        rotation_solution = propagate_attitude(
            init_rotation,
            inertia_tensor,
            external_torque,
            t_span,
            t_eval
        )

        final_translation = translation_solution.y[:, -1]
        final_rotation = rotation_solution.y[:, -1]

        propagated_state = np.concatenate((final_translation, final_rotation))

    predicted_measurement, visible = measurement_function(propagated_state, inspector_pos, t,
                                                            n, target_geom, camera_config)
    
    return predicted_measurement, visible

def propagated_measurement_jacobian(init_target_state, inspector_pos, t,
                            n, target_geom, camera_config, inertia_tensor):

    y0, visible0 = propagated_measurement(init_target_state, inspector_pos, t, n,
                                        target_geom, camera_config, inertia_tensor)

    num_meas = len(y0)
    num_states = len(init_target_state)

    H = np.zeros((num_meas, num_states))
    for j in range(num_states):
        epsilon = appropriate_step_size(j)

        x_plus = init_target_state.copy()
        x_minus = init_target_state.copy()

        x_plus[j] += epsilon
        x_minus[j] -= epsilon

        y_plus, visible_plus = propagated_measurement(x_plus, inspector_pos, t, n, target_geom, camera_config, inertia_tensor)
        y_minus, visible_minus = propagated_measurement(x_minus, inspector_pos, t, n, target_geom, camera_config, inertia_tensor)

        if not np.array_equal(visible_plus, visible0) or not np.array_equal(visible_minus, visible0):
            raise ValueError("Feature visibility changed during Jacobian calculation")

        H[:, j] = (y_plus - y_minus) / (2 * epsilon)

    return H

def accumulated_info_gain(H_history, prior_std, sigma_pixel):
    num_states = len(prior_std)

    info_sum = np.zeros((num_states, num_states))

    for H in H_history:
        scaled_H = H * prior_std / sigma_pixel
        info_sum += scaled_H.T @ scaled_H

    info_matrix = np.eye(num_states) + info_sum

    sign, log_det = np.linalg.slogdet(info_matrix)

    if sign <= 0:
        raise ValueError("Information matrix must have a positive determinant")

    return 0.5 * log_det