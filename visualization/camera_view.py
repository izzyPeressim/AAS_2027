"""
Author: Isabella Peressim
Date Created: 11/18/2026
Description: Animated pinhole camera observations of target vertices
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sensors.measurements import measurement_function


def animate_camera(time_grid, target_states, inspector_states,
                   vertices_body, mean_motion, camera_config,
                   frame_step=5):

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

    frame_indices = np.arange(0, len(time_grid), frame_step)

    if frame_indices[-1] != len(time_grid) - 1:
        frame_indices = np.append(frame_indices, len(time_grid) - 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax_zoom = inset_axes(
        ax,
        width="35%",
        height="45%",
        loc="lower right"
    )

    ax_zoom.set_xlim(600, 680)
    ax_zoom.set_ylim(400, 320)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_title("Magnified Target", fontsize=10)

    zoom_points = ax_zoom.scatter(
        [], [],
        color="royalblue",
        s=40
    )

    ax.set_xlim(0, camera_config.image_width)
    ax.set_ylim(camera_config.image_height, 0)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    ax.set_title("Pinhole Camera Observations")

    ax.axvline(
        camera_config.principal_point_x,
        color="lightgray",
        linestyle="--"
    )

    ax.axhline(
        camera_config.principal_point_y,
        color="lightgray",
        linestyle="--"
    )

    points = ax.scatter(
        [], [],
        color="royalblue",
        s=60,
        label="Visible target vertices"
    )

    labels = [
        ax.text(0, 0, str(i), fontsize=9, visible=False)
        for i in range(len(vertices_body))
    ]

    ax.legend(loc="upper right")

    def update(frame_index):
        i = frame_indices[frame_index]

        measurement, visible = measurement_function(
            target_states[i],
            inspector_states[i, 0:3],
            time_grid[i],
            mean_motion,
            vertices_body,
            camera_config
        )

        visible = np.asarray(visible, dtype=bool)
        pixels = measurement.reshape(-1, 2)

        if pixels.shape[0] != np.count_nonzero(visible):
            raise ValueError("Measurement and visibility dimensions disagree")

        points.set_offsets(pixels if len(pixels) else np.empty((0, 2)))

        zoom_points.set_offsets(
            pixels if len(pixels) else np.empty((0, 2))
        )

        visible_indices = np.flatnonzero(visible)

        for label in labels:
            label.set_visible(False)

        for pixel, vertex_index in zip(pixels, visible_indices):
            labels[vertex_index].set_position(
                (pixel[0] + 8, pixel[1] - 8)
            )
            labels[vertex_index].set_visible(True)

        ax.set_title(
            f"Pinhole Camera Observations | t = {time_grid[i]:.1f} s"
        )

        return (points, zoom_points, *labels)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=50,
        blit=False,
        repeat=True
    )

    update(0)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.92)

    return fig, animation
