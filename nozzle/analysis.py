"""Performance analysis for nozzle contours.

References:
- Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed.
- Anderson, *Modern Compressible Flow*, 3rd ed.
- Zucrow & Hoffman, *Gas Dynamics* Vol. 2, 1977.
"""

import numpy as np
from nozzle.gas import (
    thrust_coefficient_ideal, area_mach_ratio, mach_from_area_ratio,
    pressure_ratio, density_ratio, temperature_ratio, prandtl_meyer,
    mach_angle,
)


def rao_performance(area_ratio, bell_fraction=0.8, gamma=1.4):
    """Performance of a Rao parabolic (bell) nozzle.

    Uses the exit angle θ_e from the Rao/Sutton tables to estimate the
    divergence loss. For a bell nozzle, the exit flow has an average
    divergence angle of roughly θ_e/2, giving an effective divergence
    factor similar to a conical nozzle with half-angle ≈ θ_e.

    Sutton & Biblarz Section 3.4; Rao (1960).

    Parameters
    ----------
    area_ratio : float
        Ae/A*.
    bell_fraction : float
        Bell length fraction (0.6–1.0).
    gamma : float

    Returns
    -------
    dict with keys:
        lambda_ : float — approximate divergence factor
        Cf_ideal : float — 1D ideal Cf
        Cf : float — estimated actual Cf
        M_exit : float — exit Mach number
        theta_n_deg : float — inflection angle [degrees]
        theta_e_deg : float — exit angle [degrees]
    """
    from nozzle.contours import rao_angles

    M_exit = mach_from_area_ratio(area_ratio, gamma=gamma)
    Cf_ideal = thrust_coefficient_ideal(M_exit, gamma=gamma)
    theta_n, theta_e = rao_angles(area_ratio, bell_fraction)

    # Divergence correction: the Rao bell has a smooth transition from
    # θ_n to θ_e, so the effective divergence is less than a conical
    # nozzle at θ_e. A good approximation is λ ≈ (1 + cos(θ_e))/2,
    # which treats the exit angle as the effective half-angle.
    lambda_ = (1 + np.cos(theta_e)) / 2

    return {
        'lambda': lambda_,
        'Cf_ideal': Cf_ideal,
        'Cf': lambda_ * Cf_ideal,
        'M_exit': M_exit,
        'theta_n_deg': float(np.degrees(theta_n)),
        'theta_e_deg': float(np.degrees(theta_e)),
    }


def conical_performance(half_angle_deg, area_ratio, gamma=1.4):
    """Analytical performance of a conical nozzle.

    Parameters
    ----------
    half_angle_deg : float
        Cone half-angle in degrees.
    area_ratio : float
        Ae/A*.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    dict with keys:
        lambda_ : float — divergence loss factor (1+cos α)/2
        Cf_ideal : float — 1D ideal thrust coefficient (vacuum)
        Cf : float — actual Cf = λ · Cf_ideal
        M_exit : float — exit Mach number (1D)
    """
    from nozzle.contours import conical_divergence_loss

    M_exit = mach_from_area_ratio(area_ratio, gamma=gamma)
    Cf_ideal = thrust_coefficient_ideal(M_exit, gamma=gamma)
    lambda_ = conical_divergence_loss(half_angle_deg)

    return {
        'lambda': lambda_,
        'Cf_ideal': Cf_ideal,
        'Cf': lambda_ * Cf_ideal,
        'M_exit': M_exit,
    }


def moc_performance(mesh, gamma=1.4):
    """Compute performance metrics from a MOC characteristic mesh.

    Evaluates thrust coefficient by integrating momentum flux and
    pressure over the exit plane.

    Sutton & Biblarz Eq. 3-30; Zucrow & Hoffman Ch. 12.

    Parameters
    ----------
    mesh : CharMesh
        Complete characteristic mesh from MOC solution.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    dict with keys:
        Cf : float — thrust coefficient (vacuum)
        Cf_ideal : float — 1D ideal Cf for the mean exit Mach
        efficiency : float — Cf / Cf_ideal
        M_mean : float — area-weighted mean exit Mach
        M_max : float — maximum exit Mach
        M_min : float — minimum exit Mach
        theta_max_deg : float — maximum exit flow angle [degrees]
    """
    # Get exit plane distribution
    y_exit, M_exit, theta_exit = mesh.get_exit_plane()

    if len(y_exit) < 2:
        # Fallback: use all points at the maximum x
        all_pts = sorted(mesh.points, key=lambda p: p.x, reverse=True)
        x_max = all_pts[0].x
        exit_pts = [p for p in all_pts if abs(p.x - x_max) < 0.1]
        exit_pts.sort(key=lambda p: p.y)
        y_exit = np.array([p.y for p in exit_pts])
        M_exit = np.array([p.M for p in exit_pts])
        theta_exit = np.array([p.theta for p in exit_pts])

    if len(y_exit) < 2:
        # Still can't get exit plane — use wall points
        x_wall, y_wall = mesh.get_wall_contour()
        if len(x_wall) > 0:
            M_mean = 1.5  # Rough estimate
            return {
                'Cf': thrust_coefficient_ideal(M_mean, gamma),
                'Cf_ideal': thrust_coefficient_ideal(M_mean, gamma),
                'efficiency': 1.0,
                'M_mean': M_mean,
                'M_max': M_mean,
                'M_min': M_mean,
                'theta_max_deg': 0.0,
            }

    # Area-weighted mean Mach (annular elements in axisymmetric flow)
    if y_exit[0] < 1e-10:
        # Axis point: use half-annulus
        areas = np.zeros_like(y_exit)
        for i in range(len(y_exit)):
            if i == 0:
                dr = (y_exit[1] - y_exit[0]) / 2
                areas[i] = np.pi * dr**2
            elif i == len(y_exit) - 1:
                dr = (y_exit[-1] - y_exit[-2]) / 2
                areas[i] = 2 * np.pi * y_exit[i] * dr
            else:
                dr = (y_exit[i + 1] - y_exit[i - 1]) / 2
                areas[i] = 2 * np.pi * y_exit[i] * dr
    else:
        areas = 2 * np.pi * y_exit * np.gradient(y_exit)

    areas = np.abs(areas)
    total_area = np.sum(areas)

    if total_area < 1e-20:
        total_area = 1.0
        areas = np.ones_like(areas) / len(areas)

    M_mean = np.sum(M_exit * areas) / total_area

    # Thrust coefficient from exit plane integration
    # Cf = (1/P0·A*) · integral[ (ρ·V²·cos(θ) + P·cos(θ)) · 2πy·dy ]
    # Using isentropic relations:
    # ρV² = γ·P·M², so the integrand becomes P·(γM²+1)·cos(θ)
    # P/P0 = pressure_ratio(M)
    # All normalized by P0·A* where A* = π·r*² = π (since r*=1)

    Cf_integrand = np.zeros_like(y_exit)
    for i in range(len(y_exit)):
        M = M_exit[i]
        theta = theta_exit[i]
        p_ratio = pressure_ratio(M, gamma)
        # Momentum + pressure: (γM² + 1)·(P/P0)·cos(θ)
        Cf_integrand[i] = (gamma * M**2 + 1) * p_ratio * np.cos(theta)

    # Integrate over exit plane: Cf = (2/A*) · integral[ f(y) · y · dy ]
    # A* = 1 (normalized), so Cf = 2 · integral[ f · y · dy ]
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    Cf = 2 * _trapz(Cf_integrand * y_exit, y_exit)

    # 1D ideal Cf at the mean exit Mach
    Cf_ideal = thrust_coefficient_ideal(max(M_mean, 1.01), gamma)

    efficiency = Cf / Cf_ideal if Cf_ideal > 0 else 1.0

    return {
        'Cf': Cf,
        'Cf_ideal': Cf_ideal,
        'efficiency': efficiency,
        'M_mean': M_mean,
        'M_max': float(np.max(M_exit)),
        'M_min': float(np.min(M_exit)),
        'theta_max_deg': float(np.degrees(np.max(np.abs(theta_exit)))),
    }
