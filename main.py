"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: Runs an episode and coordinates propagation, measurements, and outputs
"""

import numpy as np
import matplotlib.pyplot as plt
from episodes.episode_01 import create_episode_01
from spacecraft.target import Target
from spacecraft.inspector import Inspector
from spacecraft.geometry import create_box_vertices
from sensors.measurements import measurement_function
from estimation.information import (measurement_jacobian, information_gain, 
                                    propagated_measurement, propagated_measurement_jacobian, accumulated_info_gain)
from visualization.orbits import plot_cw_traj
from visualization.target_animation import animate_target
from visualization.camera_view import animate_camera
from visualization.information_plot import plot_accumulated_information_gain

def main():

    episode = create_episode_01()

    n = episode.orbit.mean_motion
    sim = episode.simulation

    start = sim.initial_time
    end = sim.final_time
    step = sim.time_step

    time_grid = np.arange(start, end + step / 2, step)

    target = Target(episode.target)
    inspector = Inspector(episode.inspector)

    target.propagate(n, time_grid)
    inspector.propagate(n, time_grid)

    target_states = []
    inspector_states = []

    for i, t in enumerate(time_grid):
        target_state = target.get_state(i)
        inspector_state = inspector.get_state(i)

        target_states.append(target_state)
        inspector_states.append(inspector_state)

    vertices_body = create_box_vertices(episode.target.length, episode.target.width, episode.target.height)

    meas_hist = []
    vis_hist = []

    for i, t in enumerate(time_grid):
        target_state = target_states[i]

        inspector_pos = inspector_states[i][0:3]

        pred_meas, visible = measurement_function(target_state, inspector_pos, t, n, vertices_body, episode.camera)

        meas_hist.append(pred_meas)
        vis_hist.append(visible)

    visibility = np.asarray(vis_hist, dtype=bool)

    visible_counts = visibility.sum(axis=1)

    print("Minimum visible vertices:", visible_counts.min())
    print("Maximum visible vertices:", visible_counts.max())
    print("Frames with all 8 visible:", np.sum(visible_counts == 8))
    print("Total frames:", len(time_grid))
    print("All vertices visible throughout:", np.all(visibility))

    prior_std = np.array([
        1.0, 1.0, 1.0,       # Position [m]
        0.1, 0.1, 0.1,       # Velocity [m/s]
        0.1, 0.1, 0.1,       # Euler angles [rad]
        0.01, 0.01, 0.01     # Angular velocity [rad/s]
    ])

    observation_indices = [0, 25, 50, 75, 100]

    H_history = []
    ig_times = []
    ig_values = []

    for k in observation_indices:
        H_k = propagated_measurement_jacobian(
            target_states[0],
            inspector_states[k][:3],
            time_grid[k],
            episode.orbit.mean_motion,
            vertices_body,
            episode.camera,
            episode.target.inertia_tensor
        )

        H_history.append(H_k)

        ig = accumulated_info_gain(
            H_history,
            prior_std,
            episode.camera.measurement_noise_std
        )

        ig_times.append(time_grid[k])
        ig_values.append(ig)

        print(
            f"Time: {time_grid[k]:.1f} s | "
            f"Accumulated IG: {ig:.6f} nats"
        )

    ig_fig, ig_ax = plot_accumulated_information_gain(
        ig_times,
        ig_values
    )

    plt.show()


    k = 0 #choosing first time step for one frame proof of concept

    x_target = target_states[k]
    r_inspector = inspector_states[k][0:3]
    t_observation = time_grid[k]

    H = measurement_jacobian(x_target, r_inspector, t_observation,
                                n, vertices_body, episode.camera)

    IG = information_gain(H, prior_std, episode.camera.measurement_noise_std)

    print(f"Single-measurement information gain: {IG:.6f} nats")

    k_test = 100

    y_direct = meas_hist[k_test]

    y_propagated, visible_propagated = propagated_measurement(
        target_states[0],
        inspector_states[k_test][0:3],
        time_grid[k_test],
        n,
        vertices_body,
        episode.camera,
        episode.target.inertia_tensor
    )

    print(
        "Maximum measurement difference:",
        np.max(np.abs(y_direct - y_propagated))
    )

    print(
        "Visibility matches:",
        np.array_equal(vis_hist[k_test], visible_propagated)
    )

    H_future = propagated_measurement_jacobian(target_states[0], inspector_states[k_test][0:3], time_grid[k_test],
                                                n, vertices_body, episode.camera, episode.target.inertia_tensor)

    print("Future Jacobian shape:", H_future.shape)
    print("Future velocity column norms:",
        np.linalg.norm(H_future[:, 3:6], axis=0))
    print("Future angular velocity column norms:",
        np.linalg.norm(H_future[:, 9:12], axis=0))

    observation_indices = [0, 25, 50, 75, 100]

    H_history = []
    IG_history = []

    for k in observation_indices:

        H_k = propagated_measurement_jacobian(
            target_states[0],
            inspector_states[k][0:3],
            time_grid[k],
            n,
            vertices_body,
            episode.camera,
            episode.target.inertia_tensor
        )

        H_history.append(H_k)  

        IG_k = accumulated_info_gain(
            H_history,
            prior_std,
            episode.camera.measurement_noise_std
        )

        IG_history.append(IG_k)

        print(f"Time: {time_grid[k]:.0f} s | Accumulated IG: {IG_k:.6f} nats")  
    
    print("Target initial position:", target_states[0][:3])
    print("Target final position:", target_states[-1][:3])

    print("Inspector initial position:", inspector_states[0][:3])
    print("Inspector final position:", inspector_states[-1][:3])
    
    plot_cw_traj(time_grid, target_states, inspector_states)
    
    
    fig, animation = animate_target(
        time_grid,
        target_states,
        inspector_states,
        episode.target,
        episode.orbit.mean_motion,
        frame_step=5
    )

    plt.show()

    camera_fig, camera_animation = animate_camera(
        time_grid,
        target_states,
        inspector_states,
        vertices_body,
        episode.orbit.mean_motion,
        episode.camera,
        frame_step=5
    )

    plt.show()
        
    return time_grid, target_states, inspector_states, meas_hist, vis_hist

if __name__ == "__main__":
    main()

