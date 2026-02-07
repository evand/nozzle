"""Transonic kernel: initial data line for MOC near the sonic throat.

The sonic line (M=1) is a singular point for MOC. Hall's (1962) perturbation
expansion provides the initial data slightly downstream where M > 1.

Reference:
    Hall, "Transonic Flow in Two-Dimensional and Axially-Symmetric Nozzles",
    QJMAM, 15(4), 1962, pp. 487-508.

    Also presented in Zucrow & Hoffman, *Gas Dynamics* Vol. 2, Ch. 16.
"""

import numpy as np
from nozzle.gas import prandtl_meyer, mach_angle


def hall_initial_line(n_points, gamma=1.4, R_wall=1.5, x_start=0.05):
    """Generate initial data line downstream of the sonic throat.

    Uses a linear Mach number distribution from axis to wall at a small
    distance downstream of the throat, consistent with the transonic
    perturbation analysis.

    For an axisymmetric nozzle with wall radius of curvature R_wall
    (normalized by throat radius), the transonic acceleration is:

        dM/dx|_{throat} ≈ sqrt((γ+1)/(2·R_wall))    (Hall 1962, Eq. 2.5)

    At a small x downstream, the Mach number distribution is approximately:
        M(x, y) ≈ 1 + ε·f(y/r*)

    where ε = x/R_wall is the perturbation parameter and f captures the
    radial variation.

    Parameters
    ----------
    n_points : int
        Number of points on the initial data line (from axis to wall).
    gamma : float
        Ratio of specific heats.
    R_wall : float
        Wall radius of curvature at throat, normalized by throat radius.
    x_start : float
        Starting x position downstream of throat (normalized by r*).
        Must be small enough for perturbation to be valid.

    Returns
    -------
    x : ndarray, shape (n_points,)
        Axial positions (all equal to x_start).
    y : ndarray, shape (n_points,)
        Radial positions from 0 (axis) to y_wall.
    M : ndarray, shape (n_points,)
        Mach numbers along the initial line.
    theta : ndarray, shape (n_points,)
        Flow angles [radians] along the initial line.

    Notes
    -----
    Zucrow & Hoffman Ch. 16; Hall (1962) Eqs. 2.1-2.15.
    The wall y-coordinate at x_start accounts for wall curvature:
        y_wall = 1 + x²/(2·R_wall)  (circular arc approximation).
    """
    # Wall position at x_start (circular arc approximation)
    y_wall = 1.0 + x_start**2 / (2 * R_wall)

    # Radial positions from axis to wall
    y = np.linspace(0, y_wall, n_points)

    # Normalized radial coordinate
    eta = y / y_wall  # 0 at axis, 1 at wall

    # Transonic Mach gradient (Hall 1962, Eq. 2.5; Z&H Eq. 16.45)
    dMdx = np.sqrt((gamma + 1) / (2 * R_wall))

    # Centerline Mach number at x_start
    M_axis = 1.0 + dMdx * x_start

    # Radial Mach distribution: M decreases from axis to wall due to
    # the acceleration being strongest on the axis.
    # From Hall's perturbation: M(x,y) ≈ 1 + a1·x - b1·x·(y/r*)²
    # where a1 = dMdx, b1 captures the radial decrease.
    # For axisymmetric flow: b1 ≈ (γ+1)/(4·R_wall)  (Z&H Eq. 16.50)
    b1 = (gamma + 1) / (4 * R_wall)
    M = 1.0 + dMdx * x_start - b1 * x_start * eta**2

    # Ensure M > 1 everywhere (sonic at worst at wall)
    M = np.maximum(M, 1.0 + 1e-8)

    # Flow angle distribution: θ varies from 0 on axis to wall slope
    # From perturbation: θ(x,y) ≈ c1·x·(y/r*)  (Z&H Eq. 16.48)
    # Wall slope at x_start: dyw/dx = x_start/R_wall
    theta_wall = x_start / R_wall
    theta = theta_wall * eta

    # x positions (all on the same vertical line)
    x = np.full(n_points, x_start)

    return x, y, M, theta
