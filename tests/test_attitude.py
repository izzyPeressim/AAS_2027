"""
Author: Isabella Peressim
Date Created: 11/18/2026
Description: Confirm attitude dynamics of the target.
"""


import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics.attitude import euler_to_dcm, euler_angle_rates
from dynamics.propagation import propagate_attitude


def main():
    inertia = np.diag([8.0, 10.0, 12.0])
    initial_state = np.array([0.1, 0.2, 0.3, 0.01, 0.015, 0.02])
    times = np.linspace(0.0, 600.0, 601)
    solution = propagate_attitude(initial_state, inertia, np.zeros(3),
                                  (times[0], times[-1]), times)
    states = solution.y.T
    eulers, omegas = states[:, :3], states[:, 3:]
    energy = 0.5 * np.einsum('ni,ij,nj->n', omegas, inertia, omegas)
    h_body = omegas @ inertia.T
    h_inertial = np.array([euler_to_dcm(e) @ h for e, h in zip(eulers, h_body)])
    h_magnitudes = np.linalg.norm(h_inertial, axis=1)
    energy_drift = np.max(np.abs(energy - energy[0])) / max(abs(energy[0]), 1e-15)
    h_magnitude_drift = np.max(np.abs(h_magnitudes - h_magnitudes[0])) / max(h_magnitudes[0], 1e-15)
    h_vector_drift = np.max(np.linalg.norm(h_inertial - h_inertial[0], axis=1)) / max(h_magnitudes[0], 1e-15)
    min_cos_pitch = np.min(np.abs(np.cos(eulers[:, 1])))
    orthogonality_error = max(np.linalg.norm(euler_to_dcm(e).T @ euler_to_dcm(e) - np.eye(3), ord='fro') for e in eulers)

    # Independent kinematic check: numerical DCM time derivative against C @ skew(omega).
    kinematic_errors = []
    for k in range(1, len(times) - 1):
        C = euler_to_dcm(eulers[k])
        C_dot_fd = (euler_to_dcm(eulers[k + 1]) - euler_to_dcm(eulers[k - 1])) / (times[k + 1] - times[k - 1])
        wx, wy, wz = omegas[k]
        skew = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
        kinematic_errors.append(np.linalg.norm(C_dot_fd - C @ skew, ord='fro'))
    max_kinematic_error = max(kinematic_errors)

    print(f'Max relative rotational-energy drift: {energy_drift:.3e}')
    print(f'Max relative inertial angular-momentum magnitude drift: {h_magnitude_drift:.3e}')
    print(f'Max relative inertial angular-momentum VECTOR drift: {h_vector_drift:.3e}')
    print(f'Minimum |cos(pitch)|: {min_cos_pitch:.6f}')
    print(f'Max DCM orthogonality error (Frobenius): {orthogonality_error:.3e}')
    print(f'Max DCM kinematics error (1/s): {max_kinematic_error:.3e}')

    assert energy_drift < 1e-6, 'Rotational energy is not conserved sufficiently'
    assert h_magnitude_drift < 1e-6, 'Angular momentum magnitude is not conserved sufficiently'
    assert h_vector_drift < 1e-6, 'Inertial angular momentum direction is not conserved sufficiently'
    assert min_cos_pitch > 0.05, 'Euler angles approach a 3-2-1 singularity'
    assert orthogonality_error < 1e-10, 'DCM is not orthogonal'
    assert max_kinematic_error < 1e-4, 'DCM derivative inconsistent with body angular velocity'
    print('PASS: Attitude dynamics and kinematics checks')


if __name__ == '__main__':
    main()
