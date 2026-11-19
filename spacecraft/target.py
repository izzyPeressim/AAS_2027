"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: Target state, mass properties, and target-specific dynamics
"""

import numpy as np

from dynamics.propagation import propagate_cw, propagate_attitude


class Target:

    def __init__(self, config):
        self.config = config

        # Target geometry
        self.length = config.length
        self.width = config.width
        self.height = config.height

        # Target inertia tensor
        self.inertia_tensor = np.asarray(
            config.inertia_tensor, dtype=float
        )

        # Initial translational state in CW frame
        self.initial_translation = np.concatenate((
            config.initial_position,
            config.initial_velocity
        ))

        # Initial rotational state
        self.initial_rotation = np.concatenate((
            config.initial_euler_angles,
            config.initial_angular_velocity
        ))

        # Full 12-element target state
        self.initial_state = np.concatenate((
            self.initial_translation,
            self.initial_rotation
        ))

        # Propagation results
        self.translation_solution = None
        self.rotation_solution = None

    def propagate(self, mean_motion, time_grid):

        time_grid = np.asarray(time_grid, dtype=float)

        if time_grid.ndim != 1 or len(time_grid) < 2:
            raise ValueError(
                "time_grid must contain at least two times."
            )

        if not np.all(np.diff(time_grid) > 0):
            raise ValueError(
                "time_grid must be strictly increasing."
            )

        t_span = (time_grid[0], time_grid[-1])

        acceleration = np.zeros(3)#unforced target (for now)

        self.translation_solution = propagate_cw(
            self.initial_translation,
            mean_motion,
            acceleration,
            t_span,
            time_grid
        )
        torque = np.zeros(3) #torque free target rotation

        self.rotation_solution = propagate_attitude(
            self.initial_rotation,
            self.inertia_tensor,
            torque,
            t_span,
            time_grid
        )

    def get_state(self, time_index):

        if (
            self.translation_solution is None
            or self.rotation_solution is None
        ):
            raise RuntimeError(
                "Target must be propagated before requesting state."
            )

        translation = self.translation_solution.y[:, time_index]
        rotation = self.rotation_solution.y[:, time_index]

        # [position, velocity, Euler angles, angular velocity]
        return np.concatenate((translation, rotation))
