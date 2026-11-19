"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: CW equations for both inspector and target, using a common mean motion
"""

import numpy as np

def cw_dynamics(t, state, mean_motion, acceleration):
    x, y, z, xdot, ydot, zdot = state
    ux, uy, uz = acceleration

    ax = 3 * mean_motion ** 2 * x + 2 * mean_motion * ydot + ux
    ay = -2 * mean_motion * xdot + uy
    az = -mean_motion ** 2 * z + uz

    state_derivative = np.array([xdot, ydot, zdot, ax, ay, az])
    return state_derivative

# CW dynamics for a deputy relative to a common fictitious chief.
# The chief defines the circular reference orbit and rotating CW frame.
#
# Both the target and inspector are propagated independently using
# this function, with the same reference mean motion.
#
# Target: uncontrolled deputy (zero acceleration).
# Inspector: controlled deputy (acceleration is the optimization input).
#
# Relative inspector-target states are obtained by subtracting their
# respective chief-relative states.