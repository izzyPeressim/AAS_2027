"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: Inspector state and translational control; expandable to 6DoF
"""

import numpy as np
from dynamics.propagation import propagate_cw


class Inspector:

    def __init__(self, config):
        self.config = config

        # Physical properties
        self.mass = config.mass

        # Initial state in CW reference frame
        self.initial_state = np.concatenate((
            config.initial_position,
            config.initial_velocity
        ))

        # Propagation results
        self.translation_solution = None

    def propagate(self, mean_motion, time_grid, acceleration=None):

        time_grid = np.asarray(time_grid, dtype=float)

        if time_grid.ndim != 1 or len(time_grid) < 2:
            raise ValueError(
                "time_grid must contain at least two times."
            )

        if not np.all(np.diff(time_grid) > 0):
            raise ValueError(
                "time_grid must be strictly increasing."
            )

        if acceleration is None:
            acceleration = np.zeros(3)

        t_span = (time_grid[0], time_grid[-1])

        self.translation_solution = propagate_cw(
            self.initial_state,
            mean_motion,
            acceleration,
            t_span,
            time_grid
        )

    def get_state(self, time_index):

        if self.translation_solution is None:
            raise RuntimeError(
                "Inspector must be propagated before requesting state."
            )

        return self.translation_solution.y[:, time_index]
