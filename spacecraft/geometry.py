"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: Rectangular-prism vertices and transformations into the common frame
"""
import numpy as np
from dynamics.attitude import euler_to_dcm

def create_box_vertices(l, w, h):
    p = []
    pos_neg = [1, -1]
    for i in pos_neg:
        for j in pos_neg:
            for k in pos_neg:
                p.append([i*l/2, j*w/2, k*h/2])
    return np.array(p)

def cw_to_inertial_dcm(time, mean_motion):
    angle = mean_motion * time

    C_IC = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])

    return C_IC


def vertices_in_cw_frame(vertices_body, target_position_cw, euler_angles, time, mean_motion):
    # Body to inertial
    C_IB = euler_to_dcm(euler_angles)

    # CW to inertial
    C_IC = cw_to_inertial_dcm(time, mean_motion)

    # Body to CW
    C_CB = C_IC.T @ C_IB

    # Rotate all eight vertices into the CW frame
    vertices_rotated = vertices_body @ C_CB.T

    # Translate vertices by target COM position
    vertices_cw = vertices_rotated + target_position_cw

    return vertices_cw
