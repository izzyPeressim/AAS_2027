"""
Author: Isabella Peressim
Date Created: 11/18/2026
Description: Visualize the target and inspector CW trajectories, including their motion relative to one another.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_cw_traj(time_grid, target_states, inspector_states):
    target_states = np.asarray(target_states)
    inspector_states = np.asarray(inspector_states)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        target_states[:, 0],
        target_states[:, 1],
        target_states[:, 2],
        label="Target"
    )

    ax.plot(
        inspector_states[:, 0],
        inspector_states[:, 1],
        inspector_states[:, 2],
        label="Inspector"
    )

