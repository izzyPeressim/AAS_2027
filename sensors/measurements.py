"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: Constructs y=h(x)+w from the projected target features
"""

import numpy as np
from spacecraft.geometry import vertices_in_cw_frame
from sensors.camera import project_points

def measurement_function(target_state, inspector_pos, t, n, target_geom, camera_config):
    target_pos = target_state[0:3]
    euler_angles = target_state[6:9]

    verts_cw = vertices_in_cw_frame(target_geom, target_pos, euler_angles, t, n)

    pixels, visible = project_points(verts_cw, inspector_pos, target_pos, camera_config)

    valid_pixels = pixels[visible]

    predicted_measurement = valid_pixels.flatten()

    return predicted_measurement, visible

