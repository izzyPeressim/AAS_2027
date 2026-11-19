"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: Camera geometry, intrinsics, and projection conventions
"""


import numpy as np

def camera_orientation(inspector_position, target_position):

    rel_pos = target_position - inspector_position
    dist = np.linalg.norm(rel_pos)
    
    if dist < 1e-9:
        raise RuntimeError(
                    f"Camera and Target Collision")
    
    z_axis = rel_pos / dist

    ref_up = [0, 0, 1]

    if (np.abs(np.dot(ref_up, z_axis)) > 0.99):
        ref_up = [0, 1, 0]

    x_axis = np.cross(ref_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    C_camera_cw = np.vstack((x_axis, y_axis, z_axis))

    return C_camera_cw

def project_points(vertices_cw, inspector_pos, target_pos, camera_config):
    C_camera_cw = camera_orientation(inspector_pos, target_pos)
    rel_verts = vertices_cw - inspector_pos
    verts_cam = rel_verts @ C_camera_cw.T

    fx = camera_config.focal_length_x
    fy = camera_config.focal_length_y
    cx = camera_config.principal_point_x
    cy = camera_config.principal_point_y

    width = camera_config.image_width
    height = camera_config.image_height

    pixels = np.full((8, 2), np.nan)
    visible = np.zeros(8, dtype=bool)

    for i, vertex in enumerate(verts_cam):
        X, Y, Z = vertex

        if Z <= 0:
            continue

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        if 0 <= u < width and 0 <= v < height:
            pixels[i] = [u, v]
            visible[i] = True

    return pixels, visible