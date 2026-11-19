"""
Author: Isabella Peressim
Date Created: 11/18/2026
Description: Displays the rectangular target as a 3D wireframe.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from spacecraft.geometry import (
    create_box_vertices,
    vertices_in_cw_frame
)


def animate_target(time_grid, target_states, inspector_states,
                   target_config, mean_motion, frame_step=5):

    time_grid = np.asarray(time_grid)
    target_states = np.asarray(target_states)
    inspector_states = np.asarray(inspector_states)

    if time_grid.ndim != 1 or len(time_grid) == 0:
        raise ValueError("time_grid must be a nonempty 1D array")

    if target_states.shape != (len(time_grid), 12):
        raise ValueError("target_states must have shape (N, 12)")

    if inspector_states.shape != (len(time_grid), 6):
        raise ValueError("inspector_states must have shape (N, 6)")

    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")

    vertices_body = create_box_vertices(
        target_config.length,
        target_config.width,
        target_config.height
    )

    # Vertex indices differ by one sign for each box edge
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    frame_indices = np.arange(0, len(time_grid), frame_step)

    if frame_indices[-1] != len(time_grid) - 1:
        frame_indices = np.append(frame_indices, len(time_grid) - 1)

    fig = plt.figure(figsize=(14, 6))
    ax_target = fig.add_subplot(121, projection="3d")
    ax_orbit = fig.add_subplot(122, projection="3d")

    # Left View: Rotating target geometry

    initial_vertices = vertices_in_cw_frame(
        vertices_body,
        target_states[0, 0:3],
        target_states[0, 6:9],
        time_grid[0],
        mean_motion
    )

    vertex_scatter = ax_target.scatter(
        initial_vertices[:, 0],
        initial_vertices[:, 1],
        initial_vertices[:, 2],
        color="royalblue",
        s=45,
        label="Target vertices"
    )

    edge_lines = []

    for a, b in edges:
        line, = ax_target.plot(
            initial_vertices[[a, b], 0],
            initial_vertices[[a, b], 1],
            initial_vertices[[a, b], 2],
            color="royalblue",
            linewidth=1.5
        )
        edge_lines.append(line)

    target_com = ax_target.scatter(
        [], [], [],
        color="black",
        s=45,
        marker="x",
        label="Target COM"
    )

    target_positions = target_states[:, 0:3]

    max_dimension = max(
        target_config.length,
        target_config.width,
        target_config.height
    )

    margin = 1.5 * max_dimension

    for axis_index, setter in enumerate([
        ax_target.set_xlim,
        ax_target.set_ylim,
        ax_target.set_zlim
    ]):
        setter(
            np.min(target_positions[:, axis_index]) - margin,
            np.max(target_positions[:, axis_index]) + margin
        )

    ax_target.set_xlabel("CW X (m)")
    ax_target.set_ylabel("CW Y (m)")
    ax_target.set_zlabel("CW Z (m)")
    ax_target.set_box_aspect((1, 1, 1))
    ax_target.legend(loc="upper right")

    # --------------------------------------------------
    # Right View: Inspector CW trajectory
    # --------------------------------------------------

    inspector_positions = inspector_states[:, 0:3]

    ax_orbit.plot(
        inspector_positions[:, 0],
        inspector_positions[:, 1],
        inspector_positions[:, 2],
        color="lightgray",
        linestyle="--",
        label="Full inspector trajectory"
    )

    trajectory_line, = ax_orbit.plot(
        [], [], [],
        color="darkorange",
        linewidth=2,
        label="Inspector trajectory traveled"
    )

    inspector_marker, = ax_orbit.plot(
        [], [], [],
        marker="o",
        color="darkorange",
        markersize=7,
        linestyle="None",
        label="Inspector"
    )

    target_marker, = ax_orbit.plot(
        [], [], [],
        marker="o",
        color="royalblue",
        markersize=7,
        linestyle="None",
        label="Target COM"
    )

    los_line, = ax_orbit.plot(
        [], [], [],
        color="gray",
        linestyle=":",
        linewidth=1,
        label="Line of sight"
    )

    all_positions = np.vstack((
        target_positions,
        inspector_positions
    ))

    lower = all_positions.min(axis=0)
    upper = all_positions.max(axis=0)

    center = (lower + upper) / 2
    half_range = max(np.max(upper - lower) / 2, 1.0) * 1.1

    ax_orbit.set_xlim(center[0] - half_range,
                      center[0] + half_range)
    ax_orbit.set_ylim(center[1] - half_range,
                      center[1] + half_range)
    ax_orbit.set_zlim(center[2] - half_range,
                      center[2] + half_range)

    ax_orbit.set_box_aspect((1, 1, 1))
    ax_orbit.set_xlabel("CW X (m)")
    ax_orbit.set_ylabel("CW Y (m)")
    ax_orbit.set_zlabel("CW Z (m)")
    ax_orbit.set_title("Inspector Relative Motion")
    ax_orbit.legend(loc="upper right", fontsize=8)

    # Animation update
    def update(frame_index):
        i = frame_indices[frame_index]
        t = time_grid[i]

        target_pos = target_states[i, 0:3]
        inspector_pos = inspector_states[i, 0:3]

        vertices = vertices_in_cw_frame(
            vertices_body,
            target_pos,
            target_states[i, 6:9],
            t,
            mean_motion
        )

        # Update target vertices
        vertex_scatter._offsets3d = (
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2]
        )

        # Update box edges
        for line, (a, b) in zip(edge_lines, edges):
            line.set_data(
                vertices[[a, b], 0],
                vertices[[a, b], 1]
            )
            line.set_3d_properties(
                vertices[[a, b], 2]
            )

        target_com._offsets3d = (
            [target_pos[0]],
            [target_pos[1]],
            [target_pos[2]]
        )

        # Update traveled inspector trajectory.
        traveled = inspector_positions[:i + 1]

        trajectory_line.set_data(
            traveled[:, 0],
            traveled[:, 1]
        )
        trajectory_line.set_3d_properties(
            traveled[:, 2]
        )

        inspector_marker.set_data(
            [inspector_pos[0]],
            [inspector_pos[1]]
        )
        inspector_marker.set_3d_properties(
            [inspector_pos[2]]
        )

        target_marker.set_data(
            [target_pos[0]],
            [target_pos[1]]
        )
        target_marker.set_3d_properties(
            [target_pos[2]]
        )

        los_line.set_data(
            [inspector_pos[0], target_pos[0]],
            [inspector_pos[1], target_pos[1]]
        )
        los_line.set_3d_properties(
            [inspector_pos[2], target_pos[2]]
        )

        ax_target.set_title(
            f"Rotating Target | t = {t:.1f} s"
        )

        return (
            vertex_scatter,
            target_com,
            *edge_lines,
            trajectory_line,
            inspector_marker,
            target_marker,
            los_line
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=50,
        blit=False,
        repeat=True
    )

    update(0)
    plt.tight_layout()

    return fig, animation
