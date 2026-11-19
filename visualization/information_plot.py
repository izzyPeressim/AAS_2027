
"""
Author: Isabella Peressim
Date Created: 11/18/2026
Description: Visualization of accumulated information gain
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_accumulated_information_gain(times, ig_values):
    times = np.asarray(times, dtype=float)
    ig_values = np.asarray(ig_values, dtype=float)

    if times.ndim != 1 or ig_values.ndim != 1:
        raise ValueError("Times and IG values must be 1D arrays")

    if len(times) == 0 or len(times) != len(ig_values):
        raise ValueError("Times and IG values must have matching nonzero lengths")

    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(ig_values)):
        raise ValueError("Times and IG values must be finite")

    if np.any(np.diff(times) <= 0):
        raise ValueError("Observation times must be strictly increasing")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        times,
        ig_values,
        marker="o",
        linewidth=2,
        color="royalblue",
        label="Accumulated information gain"
    )

    ax.set_xlabel("Observation time (s)")
    ax.set_ylabel("Accumulated information gain (nats)")
    ax.set_title("Accumulated Information Gain vs. Time")

    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.set_xlim(left=times[0], right=times[-1] + 1e-9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    return fig, ax
