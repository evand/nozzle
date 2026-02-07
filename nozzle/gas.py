"""Isentropic and Prandtl-Meyer relations for calorically perfect gas.

Every function cites its published source:
- Anderson, *Modern Compressible Flow* (MCF), 3rd ed., 2003
- NACA 1135, "Equations, Tables, and Charts for Compressible Flow", 1953
- Zucrow & Hoffman, *Gas Dynamics* Vol. 2, 1977
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Isentropic relations (Anderson MCF Ch. 3, NACA 1135 Eqs. 44-48)
# ---------------------------------------------------------------------------

def pressure_ratio(M, gamma=1.4):
    """P/P0 = (1 + (γ-1)/2 · M²)^(-γ/(γ-1)).

    Anderson MCF Eq. 3.30; NACA 1135 Eq. 44.
    """
    return (1 + (gamma - 1) / 2 * M**2) ** (-gamma / (gamma - 1))


def temperature_ratio(M, gamma=1.4):
    """T/T0 = (1 + (γ-1)/2 · M²)^(-1).

    Anderson MCF Eq. 3.28; NACA 1135 Eq. 43.
    """
    return (1 + (gamma - 1) / 2 * M**2) ** (-1)


def density_ratio(M, gamma=1.4):
    """ρ/ρ0 = (1 + (γ-1)/2 · M²)^(-1/(γ-1)).

    Anderson MCF Eq. 3.31; NACA 1135 Eq. 45.
    """
    return (1 + (gamma - 1) / 2 * M**2) ** (-1 / (gamma - 1))


def area_mach_ratio(M, gamma=1.4):
    """A/A* from the isentropic area-Mach relation.

    Anderson MCF Eq. 3.36; NACA 1135 Eq. 80:
        A/A* = (1/M) · [(2/(γ+1)) · (1 + (γ-1)/2 · M²)]^((γ+1)/(2(γ-1)))
    """
    gp1 = gamma + 1
    gm1 = gamma - 1
    exponent = gp1 / (2 * gm1)
    return (1.0 / M) * ((2.0 / gp1) * (1 + gm1 / 2 * M**2)) ** exponent


def mach_from_area_ratio(area_ratio, gamma=1.4, supersonic=True):
    """Invert A/A*(M) to find M given A/A*.

    Uses Brent's method on [1, 50] for supersonic or [1e-6, 1] for subsonic.
    Anderson MCF Section 3.4.
    """
    if np.isclose(area_ratio, 1.0, atol=1e-12):
        return 1.0

    def f(M):
        return area_mach_ratio(M, gamma) - area_ratio

    if supersonic:
        return brentq(f, 1.0 + 1e-12, 50.0)
    else:
        return brentq(f, 1e-6, 1.0 - 1e-12)


def mach_angle(M):
    """Mach angle μ = arcsin(1/M) [radians].

    Anderson MCF Eq. 9.1; NACA 1135.
    """
    return np.arcsin(1.0 / M)


# ---------------------------------------------------------------------------
# Prandtl-Meyer function (Anderson MCF Eq. 9.42; NACA 1135 Eq. 153)
# ---------------------------------------------------------------------------

def prandtl_meyer(M, gamma=1.4):
    """Prandtl-Meyer angle ν(M) [radians].

    Anderson MCF Eq. 9.42:
        ν = √((γ+1)/(γ-1)) · arctan(√((γ-1)/(γ+1)·(M²-1))) - arctan(√(M²-1))

    Returns 0 for M=1, monotonically increasing for M>1.
    """
    if np.isscalar(M):
        if M <= 1.0:
            return 0.0
    gp1 = gamma + 1
    gm1 = gamma - 1
    Msq_m1 = M**2 - 1
    return (np.sqrt(gp1 / gm1) * np.arctan(np.sqrt(gm1 / gp1 * Msq_m1))
            - np.arctan(np.sqrt(Msq_m1)))


def mach_from_prandtl_meyer(nu, gamma=1.4):
    """Invert ν(M) to find M given Prandtl-Meyer angle [radians].

    Uses Brent's method. Anderson MCF Section 9.6.
    """
    if nu <= 0:
        return 1.0

    def f(M):
        return prandtl_meyer(M, gamma) - nu

    # Upper bound: ν_max occurs as M → ∞
    nu_max = (np.sqrt((gamma + 1) / (gamma - 1)) - 1) * np.pi / 2
    if nu >= nu_max:
        raise ValueError(f"nu = {np.degrees(nu):.2f}° exceeds nu_max = {np.degrees(nu_max):.2f}°")

    return brentq(f, 1.0 + 1e-12, 200.0)


# ---------------------------------------------------------------------------
# Thrust coefficient (Anderson MCF Eq. 12.30; Sutton & Biblarz Eq. 3-30)
# ---------------------------------------------------------------------------

def thrust_coefficient_ideal(M_exit, gamma=1.4, p_ratio_exit=None):
    """Ideal (1D) thrust coefficient Cf = F / (P0 · A*).

    Anderson MCF Eq. 12.30; Sutton & Biblarz Eq. 3-30:
        Cf = √(2γ²/(γ-1) · (2/(γ+1))^((γ+1)/(γ-1)) · (1 - (Pe/P0)^((γ-1)/γ)))
             + (Pe/P0 - Pa/P0) · Ae/A*

    For vacuum (Pa=0) and matched (Pe=Pa) combined in the momentum form:
        Cf = γ·M_e²·(Ae/A*)^(-1)·(2/(γ+1)·(1+(γ-1)/2·M_e²))^((γ+1)/(2(γ-1)))·(1/M_e) ...

    Simplified momentum form for vacuum ambient (Pa=0):
        Cf = Ae/A* · (2γ/(γ-1) · (2/(γ+1))^((γ+1)/(γ-1)) · ... )

    We use the direct momentum + pressure form:
        F = m_dot · V_e + (Pe - Pa) · Ae
        Cf = F / (P0 · A*)

    For adapted nozzle or vacuum:
        Cf = sqrt(2γ²/(γ-1) · (2/(γ+1))^((γ+1)/(γ-1)) · [1 - (Pe/P0)^((γ-1)/γ)])
             + (Pe/P0) · (Ae/A*)      [for vacuum, Pa=0]

    Parameters
    ----------
    M_exit : float
        Exit Mach number.
    gamma : float
        Ratio of specific heats.
    p_ratio_exit : float or None
        Pe/P0. If None, computed from M_exit isentropically.

    Returns Cf for vacuum expansion (Pa=0).
    """
    gm1 = gamma - 1
    gp1 = gamma + 1

    if p_ratio_exit is None:
        p_ratio_exit = pressure_ratio(M_exit, gamma)

    ar = area_mach_ratio(M_exit, gamma)

    # Momentum term: from conservation of energy + continuity
    # Cf_mom = gamma * M_e * sqrt(2/(gamma-1) * ... ) simplified to:
    # Using Sutton Eq. 3-30 form:
    momentum_term = np.sqrt(
        2 * gamma**2 / gm1
        * (2 / gp1) ** (gp1 / gm1)
        * (1 - p_ratio_exit ** (gm1 / gamma))
    )

    # Pressure term (vacuum: Pa=0)
    pressure_term = p_ratio_exit * ar

    return momentum_term + pressure_term


def thrust_coefficient_vacuum(M_exit, gamma=1.4):
    """Cf for vacuum expansion (Pa=0). Convenience wrapper."""
    return thrust_coefficient_ideal(M_exit, gamma)


# ---------------------------------------------------------------------------
# Stagnation property recovery
# ---------------------------------------------------------------------------

def mach_from_pressure_ratio(p_over_p0, gamma=1.4):
    """M from P/P0 via isentropic relation. Anderson MCF Eq. 3.30 inverted."""
    return np.sqrt(2 / (gamma - 1) * (p_over_p0 ** (-(gamma - 1) / gamma) - 1))


def stagnation_pressure(p_static, M, gamma=1.4):
    """P0 = P_static / (P/P0). Anderson MCF Eq. 3.30."""
    return p_static / pressure_ratio(M, gamma)


# ---------------------------------------------------------------------------
# Speed of sound and velocity
# ---------------------------------------------------------------------------

def speed_of_sound(T, gamma=1.4, R=287.058):
    """a = sqrt(γ·R·T) [m/s]. Anderson MCF Eq. 1.5."""
    return np.sqrt(gamma * R * T)


def velocity_from_mach(M, T, gamma=1.4, R=287.058):
    """V = M·a [m/s]."""
    return M * speed_of_sound(T, gamma, R)
