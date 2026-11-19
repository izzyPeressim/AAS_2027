"""
Author: Isabella Peressim
Date Created: 11/12/2026
Description: Euler rotational dynamics, attitude kinematics, and attitude-to-DCM conversion
"""

import numpy as np

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)


def euler_to_dcm(euler_angles): #3 2 1 body to inertia
    phi, theta, psi = euler_angles

    # Rotation about x-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, cos(phi), -sin(phi)],
        [0, sin(phi), cos(phi)]
    ])

    # Rotation about y-axis (pitch)
    Ry = np.array([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)]
    ])

    # Rotation about z-axis (yaw)
    Rz = np.array([
        [cos(psi), -sin(psi), 0],
        [sin(psi), cos(psi), 0],
        [0, 0, 1]
    ])

    dcm = Rz @ Ry @ Rx

    return dcm

#choose euler angle representation (321 or 313) depending on the scenario

def euler_angle_rates(euler_angles, omega):
    phi, theta, psi = euler_angles

    E = np.array([
        [1.0, np.sin(phi) * np.tan(theta),
              np.cos(phi) * np.tan(theta)],

        [0.0, np.cos(phi), -np.sin(phi)],

        [0.0, np.sin(phi) / np.cos(theta),
              np.cos(phi) / np.cos(theta)]
    ])

    return E @ np.asarray(omega)


def attitude_dynamics(t, state, inertia_tensor, torque):
    phi, theta, psi = state[0:3]
    omegax, omegay, omegaz = state[3:6]

    euler_angles = np.array([phi, theta, psi])
    omega = np.array([omegax, omegay, omegaz])

    euler_dot = euler_angle_rates(euler_angles, omega)

    H = inertia_tensor @ omega

    gyro = np.cross(omega, H)

    omega_dot = np.linalg.solve(inertia_tensor, torque - gyro)
    
    state_derivative = np.concatenate((euler_dot, omega_dot))
    return state_derivative

