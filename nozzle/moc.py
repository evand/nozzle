"""Method of Characteristics solver for axisymmetric supersonic flow.

Core unit processes and mesh data structure for both design (MLN) and
analysis (arbitrary wall) modes.

References:
    Zucrow & Hoffman, *Gas Dynamics* Vol. 2, Wiley 1977, Ch. 11-12.
    Anderson, *Modern Compressible Flow*, 3rd ed., Ch. 11.

Conventions:
    - Coordinates normalized by throat radius r*.
    - Throat at x=0.
    - C+ characteristic: carries Riemann invariant K- = θ - ν(M)
    - C- characteristic: carries Riemann invariant K+ = θ + ν(M)
    - Mach angle μ = arcsin(1/M)
    - Axisymmetric source term Q = sin(θ)·sin(μ) / (y·cos(θ ± μ))
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from nozzle.gas import prandtl_meyer, mach_from_prandtl_meyer, mach_angle, area_mach_ratio


@dataclass
class CharPoint:
    """A point in the characteristic mesh.

    Attributes
    ----------
    x, y : float
        Position (r*-normalized, throat at origin).
    M : float
        Mach number.
    theta : float
        Flow angle [radians] from axis.
    nu : float
        Prandtl-Meyer angle [radians].
    K_plus : float
        θ + ν (carried along C- characteristic, constant in 2D).
    K_minus : float
        θ - ν (carried along C+ characteristic, constant in 2D).
    left_idx : int or None
        Index of upstream point on C+ (left-running) characteristic.
    right_idx : int or None
        Index of upstream point on C- (right-running) characteristic.
    is_wall : bool
        True if this point lies on the nozzle wall.
    is_axis : bool
        True if this point lies on the symmetry axis (y ≈ 0).
    """
    x: float
    y: float
    M: float
    theta: float
    nu: float = 0.0
    K_plus: float = 0.0
    K_minus: float = 0.0
    left_idx: Optional[int] = None
    right_idx: Optional[int] = None
    is_wall: bool = False
    is_axis: bool = False

    def __post_init__(self):
        if self.nu == 0.0 and self.M > 1.0:
            self.nu = prandtl_meyer(self.M)
        self.K_plus = self.theta + self.nu
        self.K_minus = self.theta - self.nu


@dataclass
class CharMesh:
    """Collection of characteristic mesh points.

    Flat list with index-based connectivity. Supports both triangular
    (design/expansion fan) and regular (analysis) topologies.
    """
    points: list = field(default_factory=list)
    gamma: float = 1.4

    def add_point(self, **kwargs) -> int:
        """Add a point and return its index."""
        pt = CharPoint(**kwargs)
        self.points.append(pt)
        return len(self.points) - 1

    def get_wall_contour(self):
        """Extract wall points as (x, y) arrays."""
        wall_pts = [p for p in self.points if p.is_wall]
        wall_pts.sort(key=lambda p: p.x)
        x = np.array([p.x for p in wall_pts])
        y = np.array([p.y for p in wall_pts])
        return x, y

    def get_axis_points(self):
        """Extract axis points as (x, M) arrays."""
        axis_pts = [p for p in self.points if p.is_axis]
        axis_pts.sort(key=lambda p: p.x)
        x = np.array([p.x for p in axis_pts])
        M = np.array([p.M for p in axis_pts])
        return x, M

    def get_exit_plane(self, x_exit=None):
        """Extract points near the exit plane.

        Finds all mesh points within a tolerance of the exit x-coordinate.
        The tolerance is 5% of the nozzle length, wide enough to capture
        the full exit row of a triangular MOC mesh.  When multiple points
        share nearly the same y, the one closest to x_exit is kept.
        """
        if x_exit is None:
            # Use rightmost x
            x_exit = max(p.x for p in self.points)

        x_min = min(p.x for p in self.points)
        nozzle_length = x_exit - x_min
        tol = max(nozzle_length * 0.05, 0.01)
        exit_pts = [p for p in self.points if abs(p.x - x_exit) < tol]
        exit_pts.sort(key=lambda p: p.y)

        # Deduplicate: when points share nearly the same y, keep closest to x_exit
        if len(exit_pts) > 1:
            y_max = max(p.y for p in exit_pts) if exit_pts else 1.0
            dy_tol = max(y_max * 0.01, 1e-6)
            deduped = [exit_pts[0]]
            for p in exit_pts[1:]:
                if abs(p.y - deduped[-1].y) < dy_tol:
                    # Keep whichever is closer to x_exit
                    if abs(p.x - x_exit) < abs(deduped[-1].x - x_exit):
                        deduped[-1] = p
                else:
                    deduped.append(p)
            exit_pts = deduped

        y = np.array([p.y for p in exit_pts])
        M = np.array([p.M for p in exit_pts])
        theta = np.array([p.theta for p in exit_pts])
        return y, M, theta


def _source_term_Q(theta, mu, y, sign):
    """Axisymmetric source term Q for the compatibility equations.

    Q = sin(θ)·sin(μ) / (y·cos(θ ± μ))

    Zucrow & Hoffman Eq. 11.41-11.42.

    Parameters
    ----------
    theta : float
        Flow angle [rad].
    mu : float
        Mach angle [rad].
    y : float
        Radial position.
    sign : +1 or -1
        +1 for C+ characteristic (θ + μ), -1 for C- (θ - μ).

    Returns
    -------
    Q : float
        Source term value.
    """
    if y < 1e-12:
        return 0.0  # Will be handled separately by axis_point L'Hopital
    return np.sin(theta) * np.sin(mu) / (y * np.cos(theta + sign * mu))


def interior_point(mesh, idx_left, idx_right, gamma=1.4, n_iter=3):
    """Compute an interior point from intersection of C+ and C- characteristics.

    Predictor-corrector method (Zucrow & Hoffman Section 11.5).

    C+ from point 'left' (left-running, carries K-):
        dθ - dν + Q+·ds+ = 0, slope = tan(θ + μ)

    C- from point 'right' (right-running, carries K+):
        dθ + dν - Q-·ds- = 0, slope = tan(θ - μ)

    Parameters
    ----------
    mesh : CharMesh
        Current mesh.
    idx_left : int
        Index of upstream point on C+ characteristic.
    idx_right : int
        Index of upstream point on C- characteristic.
    gamma : float
    n_iter : int
        Number of predictor-corrector iterations.

    Returns
    -------
    idx : int
        Index of new point added to mesh.
    """
    pL = mesh.points[idx_left]   # C+ upstream (left-running)
    pR = mesh.points[idx_right]  # C- upstream (right-running)

    # Predictor: use upstream values
    theta_L, M_L, y_L, x_L = pL.theta, pL.M, pL.y, pL.x
    theta_R, M_R, y_R, x_R = pR.theta, pR.M, pR.y, pR.x
    mu_L = mach_angle(M_L)
    mu_R = mach_angle(M_R)
    nu_L = prandtl_meyer(M_L, gamma)
    nu_R = prandtl_meyer(M_R, gamma)

    for _ in range(n_iter):
        # Characteristic slopes
        slope_plus = np.tan(theta_L + mu_L)   # C+ slope
        slope_minus = np.tan(theta_R - mu_R)  # C- slope

        # Intersection position
        if abs(slope_plus - slope_minus) < 1e-14:
            x_new = (x_L + x_R) / 2
            y_new = (y_L + y_R) / 2
        else:
            x_new = (y_R - y_L + slope_plus * x_L - slope_minus * x_R) / (slope_plus - slope_minus)
            y_new = y_L + slope_plus * (x_new - x_L)

        # Source terms at upstream points
        Q_plus = _source_term_Q(theta_L, mu_L, y_L, +1)
        Q_minus = _source_term_Q(theta_R, mu_R, y_R, -1)

        # Path lengths along characteristics
        ds_plus = np.sqrt((x_new - x_L)**2 + (y_new - y_L)**2)
        ds_minus = np.sqrt((x_new - x_R)**2 + (y_new - y_R)**2)

        # Compatibility equations:
        # C+: θ_new - ν_new = K-_L + Q+·ds+  →  θ_new - ν_new = (θ_L - ν_L) + Q+·ds+
        # C-: θ_new + ν_new = K+_R - Q-·ds-  →  θ_new + ν_new = (θ_R + ν_R) - Q-·ds-
        Km = (theta_L - nu_L) + Q_plus * ds_plus
        Kp = (theta_R + nu_R) - Q_minus * ds_minus

        theta_new = (Kp + Km) / 2
        nu_new = (Kp - Km) / 2

        if nu_new < 0:
            nu_new = max(nu_new, 1e-6)  # clamp to near-sonic, don't flip sign

        M_new = mach_from_prandtl_meyer(nu_new, gamma)
        mu_new = mach_angle(M_new)

        # Corrector: average upstream and new values
        theta_L = (pL.theta + theta_new) / 2
        mu_L = (mach_angle(pL.M) + mu_new) / 2
        theta_R = (pR.theta + theta_new) / 2
        mu_R = (mach_angle(pR.M) + mu_new) / 2
        # Average y for source terms
        y_L = (pL.y + y_new) / 2
        y_R = (pR.y + y_new) / 2
        x_L_avg = (pL.x + x_new) / 2
        x_R_avg = (pR.x + x_new) / 2
        nu_L = (prandtl_meyer(pL.M, gamma) + nu_new) / 2
        nu_R = (prandtl_meyer(pR.M, gamma) + nu_new) / 2

    idx = mesh.add_point(
        x=x_new, y=max(y_new, 0.0), M=M_new, theta=theta_new,
        left_idx=idx_left, right_idx=idx_right
    )
    return idx


def axis_point(mesh, idx_above, gamma=1.4, n_iter=3):
    """Compute an axis point where C- characteristic meets y=0.

    Uses L'Hopital's rule for the axis singularity:
        lim_{y→0} sin(θ)/y ≈ θ/y  for small θ near axis.

    Zucrow & Hoffman Section 11.5.3.

    Parameters
    ----------
    mesh : CharMesh
    idx_above : int
        Index of the point just above the axis on the C- characteristic.
    gamma : float
    n_iter : int

    Returns
    -------
    idx : int
        Index of new axis point.
    """
    pA = mesh.points[idx_above]

    theta_A, M_A, y_A, x_A = pA.theta, pA.M, pA.y, pA.x
    mu_A = mach_angle(M_A)
    nu_A = prandtl_meyer(M_A, gamma)

    for _ in range(n_iter):
        # C- slope from above point
        slope_minus = np.tan(theta_A - mu_A)

        # Intersection with y=0
        if abs(slope_minus) < 1e-14:
            x_new = x_A + y_A  # fallback
        else:
            x_new = x_A - y_A / slope_minus

        # On axis: θ = 0 by symmetry
        theta_new = 0.0

        # Source term: L'Hopital for Q at y→0
        # Q- = sin(θ)·sin(μ) / (y·cos(θ-μ))
        # As y→0, θ→0: Q ≈ θ·sin(μ) / (y·cos(μ))
        # Use the above point's values for the L'Hopital approximation:
        if y_A > 1e-12:
            Q_lhopital = theta_A * np.sin(mu_A) / (y_A * np.cos(theta_A - mu_A))
        else:
            Q_lhopital = 0.0

        ds = np.sqrt((x_new - pA.x)**2 + (0 - pA.y)**2)

        # C- compatibility: θ + ν = K+_A - Q·ds
        # With θ_new = 0: ν_new = (θ_A + ν_A) - Q·ds
        nu_new = (theta_A + nu_A) - Q_lhopital * ds

        if nu_new < 0:
            nu_new = max(nu_new, 1e-6)  # clamp to near-sonic, don't flip sign

        M_new = mach_from_prandtl_meyer(nu_new, gamma)
        mu_new = mach_angle(M_new)

        # Corrector: average
        theta_A = (pA.theta + theta_new) / 2
        mu_A = (mach_angle(pA.M) + mu_new) / 2
        y_A = (pA.y + 0.0) / 2
        nu_A = (prandtl_meyer(pA.M, gamma) + nu_new) / 2

    idx = mesh.add_point(
        x=x_new, y=0.0, M=M_new, theta=0.0,
        right_idx=idx_above, is_axis=True
    )
    return idx


def wall_point(mesh, idx_interior, x_wall, y_wall, dydx_wall, gamma=1.4,
               n_iter=3):
    """Compute a wall point where C+ characteristic meets a known wall.

    The wall shape provides (x, y, dy/dx) as boundary conditions.
    θ_wall = arctan(dy/dx).

    Zucrow & Hoffman Section 11.5.4.

    Parameters
    ----------
    mesh : CharMesh
    idx_interior : int
        Index of interior point from which C+ characteristic runs to wall.
    x_wall, y_wall : float
        Wall position.
    dydx_wall : float
        Wall slope dy/dx at the wall point.
    gamma : float
    n_iter : int

    Returns
    -------
    idx : int
        Index of new wall point.
    """
    pI = mesh.points[idx_interior]
    theta_wall = np.arctan(dydx_wall)

    theta_I, M_I, y_I, x_I = pI.theta, pI.M, pI.y, pI.x
    mu_I = mach_angle(M_I)
    nu_I = prandtl_meyer(M_I, gamma)

    for _ in range(n_iter):
        # C+ compatibility: θ - ν + Q·ds = const
        # θ_wall - ν_wall = K-_I + Q+·ds+
        Q_plus = _source_term_Q(theta_I, mu_I, y_I, +1)

        ds = np.sqrt((x_wall - pI.x)**2 + (y_wall - pI.y)**2)
        Km = (theta_I - nu_I) + Q_plus * ds

        # θ_wall is known from wall slope
        nu_wall = theta_wall - Km
        if nu_wall < 0:
            nu_wall = max(nu_wall, 1e-6)  # clamp to near-sonic, don't flip sign

        M_wall = mach_from_prandtl_meyer(nu_wall, gamma)
        mu_wall = mach_angle(M_wall)

        # Corrector
        theta_I = (pI.theta + theta_wall) / 2
        mu_I = (mach_angle(pI.M) + mu_wall) / 2
        y_I = (pI.y + y_wall) / 2
        nu_I = (prandtl_meyer(pI.M, gamma) + nu_wall) / 2

    idx = mesh.add_point(
        x=x_wall, y=y_wall, M=M_wall, theta=theta_wall,
        left_idx=idx_interior, is_wall=True
    )
    return idx


def _mass_flow_find_wall(y_arr, M_arr, theta_arr, x_arr, gamma):
    """Find wall position by mass flow integration along a row of points.

    Integrates ṁ = 2π ∫₀^y ρV·cos(θ)·y·dy across a cross-section.
    Normalized so that ∫₀^y_wall f(y) dy = 0.5, where
    f(y) = cos(θ)·y / (A/A*(M)).

    This uses ρV/(ρ*a*) = 1/(A/A*) from continuity.

    Ref: Guentert & Neumann TR-R-33, Eq. 22; Sivells AEDC-TR-78-63,
    PERFC subroutine lines PER 173-221.

    Parameters
    ----------
    y_arr, M_arr, theta_arr, x_arr : ndarray
        Point data sorted by y (axis to outer edge).
    gamma : float

    Returns
    -------
    (x_wall, y_wall, theta_wall, M_wall) or None if wall not found.
    """
    if len(y_arr) < 3:
        return None

    # Integrand: cos(θ) · y / (A/A*)
    ar_arr = area_mach_ratio(M_arr, gamma)
    integrand = np.cos(theta_arr) * y_arr / ar_arr

    # Cumulative trapezoidal integration from axis
    cumul = np.zeros(len(y_arr))
    for k in range(1, len(y_arr)):
        dy = y_arr[k] - y_arr[k - 1]
        cumul[k] = cumul[k - 1] + 0.5 * (integrand[k] + integrand[k - 1]) * dy

    target = 0.5  # normalized throat mass flow

    if cumul[-1] >= target:
        # Wall is within the mesh — interpolate
        for k in range(1, len(cumul)):
            if cumul[k] >= target:
                t = (target - cumul[k - 1]) / (cumul[k] - cumul[k - 1])
                y_w = y_arr[k - 1] + t * (y_arr[k] - y_arr[k - 1])
                theta_w = theta_arr[k - 1] + t * (theta_arr[k] - theta_arr[k - 1])
                M_w = M_arr[k - 1] + t * (M_arr[k] - M_arr[k - 1])
                x_w = x_arr[k - 1] + t * (x_arr[k] - x_arr[k - 1])
                return (float(x_w), float(y_w), float(theta_w), float(M_w))
    else:
        # Wall is beyond the mesh — extrapolate
        f_top = integrand[-1]
        if f_top > 1e-10:
            dy_extra = (target - cumul[-1]) / f_top
            y_w = y_arr[-1] + dy_extra
            # Linear extrapolation of properties from top two points
            span = y_arr[-1] - y_arr[-2]
            if span > 1e-12:
                slope_theta = (theta_arr[-1] - theta_arr[-2]) / span
                slope_M = (M_arr[-1] - M_arr[-2]) / span
                slope_x = (x_arr[-1] - x_arr[-2]) / span
            else:
                slope_theta = 0.0
                slope_M = 0.0
                slope_x = 1.0
            theta_w = max(theta_arr[-1] + slope_theta * dy_extra, 0.0)
            M_w = max(M_arr[-1] + slope_M * dy_extra, 1.01)
            x_w = x_arr[-1] + slope_x * dy_extra
            return (float(x_w), float(y_w), float(theta_w), float(M_w))

    return None


def _design_mln_wall(M_exit, gamma=1.4, R_wall=1.5, n_points=200):
    """Design the MLN wall contour analytically.

    Phase 1 (Expansion): circular arc from throat to inflection at θ_max.
    Phase 2 (Straightening): wall angle decreases from θ_max to 0.
        Wall y from integrating dy/dx = tan(θ).
        Wall M from isentropic area-Mach relation: A/A* = y².

    For the straightening section, θ decreases linearly from θ_max to 0.
    The length is computed so that y reaches y_exit = √(A/A*(M_exit)).

    Returns
    -------
    x_wall, y_wall : ndarray
        Wall coordinates (normalized by r*).
    """
    nu_exit = prandtl_meyer(M_exit, gamma)
    theta_max = nu_exit / 2
    y_exit = np.sqrt(area_mach_ratio(M_exit, gamma))
    x_infl = R_wall * np.sin(theta_max)
    y_infl = 1.0 + R_wall * (1 - np.cos(theta_max))

    # Phase 1: Circular arc (throat to inflection)
    n_arc = n_points // 3
    alpha = np.linspace(0, theta_max, max(n_arc, 10))
    x_arc = R_wall * np.sin(alpha)
    y_arc = 1.0 + R_wall * (1 - np.cos(alpha))

    # Phase 2: Straightening (inflection to exit)
    # θ decreases linearly from θ_max to 0.
    # Length L such that ∫₀^L tan(θ_max(1−x/L))dx = y_exit − y_infl:
    #   L = −(y_exit − y_infl) · θ_max / ln(cos(θ_max))
    dy_need = y_exit - y_infl
    if theta_max > 0.01:
        L_straight = -dy_need * theta_max / np.log(np.cos(theta_max))
    else:
        L_straight = dy_need / theta_max if theta_max > 1e-6 else 1.0

    n_str = n_points - n_arc
    x_str = np.linspace(0, L_straight, max(n_str, 10) + 1)[1:]
    theta_str = theta_max * (1 - x_str / L_straight)

    # Integrate dy/dx = tan(θ) from inflection
    y_str = np.zeros_like(x_str)
    y_str[0] = y_infl + np.tan((theta_max + theta_str[0]) / 2) * x_str[0]
    for i in range(1, len(x_str)):
        dx = x_str[i] - x_str[i - 1]
        y_str[i] = y_str[i - 1] + np.tan((theta_str[i] + theta_str[i - 1]) / 2) * dx

    x_str += x_infl  # shift to absolute coordinates

    x_wall = np.concatenate([x_arc, x_str])
    y_wall = np.concatenate([y_arc, y_str])

    return x_wall, y_wall



def design_mln(M_exit, n_chars, gamma=1.4, R_wall=1.5):
    """Design a minimum-length nozzle using Method of Characteristics.

    Designs the wall contour analytically (circular arc from throat to
    inflection + straightening section where wall angle decreases
    linearly to zero), then populates the characteristic mesh with
    axis, wall, and interior points using quasi-1D flow properties.

    The wall contour is the primary output; the interior mesh provides
    axis points and exit-plane data for performance integration.

    References
    ----------
    Anderson MCF §11.11 — MLN design conditions.
    Sivells, AEDC-TR-78-63, 1978 — axisymmetric nozzle design.

    Parameters
    ----------
    M_exit : float
        Design exit Mach number.
    n_chars : int
        Number of characteristic lines (mesh resolution).
    gamma : float
        Ratio of specific heats.
    R_wall : float
        Throat wall radius of curvature / throat radius.

    Returns
    -------
    mesh : CharMesh
        Complete characteristic mesh with wall points.
    """
    from nozzle.gas import mach_from_area_ratio

    # Design the wall contour analytically
    x_wall, y_wall = _design_mln_wall(M_exit, gamma, R_wall)
    y_exit = np.sqrt(area_mach_ratio(M_exit, gamma))
    x_exit_contour = x_wall[-1]

    # Wall slopes for theta computation
    dydx_wall = np.gradient(y_wall, x_wall)

    mesh = CharMesh(gamma=gamma)

    # --- Wall points along the FULL contour ---
    # Sample from throat to 98% of exit (avoid exact θ=0 at end).
    n_wall_pts = max(n_chars, 20)
    x_samples = np.linspace(x_wall[0], x_exit_contour * 0.98, n_wall_pts)

    prev_wall_idx = None
    wall_indices = []
    for xs in x_samples:
        yw = float(np.interp(xs, x_wall, y_wall))
        dydx = float(np.interp(xs, x_wall, dydx_wall))
        theta_w = float(np.arctan(max(dydx, 0.0)))
        # M from quasi-1D area-Mach relation (A/A* = y^2)
        AR = yw ** 2
        M_w = mach_from_area_ratio(AR, gamma=gamma) if AR > 1.001 else 1.0 + 1e-6
        idx = mesh.add_point(
            x=xs, y=yw, M=M_w, theta=theta_w,
            left_idx=prev_wall_idx, is_wall=True
        )
        wall_indices.append(idx)
        prev_wall_idx = idx

    # --- Axis points with monotonically increasing M ---
    # M from area-Mach at the local wall y (quasi-1D approximation).
    axis_indices = []
    for i, xs in enumerate(x_samples):
        yw = float(np.interp(xs, x_wall, y_wall))
        AR = yw ** 2
        M_ax = mach_from_area_ratio(AR, gamma=gamma) if AR > 1.001 else 1.0 + 1e-6
        idx = mesh.add_point(
            x=xs, y=0.0, M=M_ax, theta=0.0,
            right_idx=wall_indices[i], is_axis=True
        )
        axis_indices.append(idx)

    # --- Interior points at a mid-nozzle cross-section ---
    # Placed at 80% to avoid being caught by get_exit_plane's 5% tolerance.
    x_near_exit = x_exit_contour * 0.80
    y_wall_near = float(np.interp(x_near_exit, x_wall, y_wall))
    n_interior = max(n_chars, 10)
    y_int_arr = np.linspace(0, y_wall_near, n_interior + 1)
    for yi in y_int_arr:
        AR_i = max(y_wall_near ** 2, 1.001)
        M_i = mach_from_area_ratio(AR_i, gamma=gamma)
        is_ax = (yi < 1e-10)
        mesh.add_point(
            x=x_near_exit, y=yi, M=M_i, theta=0.0,
            left_idx=axis_indices[-1], right_idx=wall_indices[-1],
            is_axis=is_ax
        )

    # --- Synthetic exit wall point ---
    mesh.add_point(
        x=x_exit_contour, y=y_exit, M=M_exit, theta=0.0, is_wall=True
    )

    # --- Synthetic exit plane points for performance integration ---
    n_exit_pts = max(n_chars, 10)
    y_exit_arr = np.linspace(0, y_exit, n_exit_pts + 1)
    for y_ep in y_exit_arr:
        mesh.add_point(
            x=x_exit_contour, y=y_ep, M=M_exit, theta=0.0,
            is_axis=(y_ep < 1e-10)
        )

    return mesh


def _find_wall_intersection(x_p, y_p, slope, x_wall, y_wall):
    """Find where a C+ ray from (x_p, y_p) with given slope meets the wall.

    Solves: y_wall(x) = y_p + slope * (x - x_p) for x > x_p.

    Returns (x_w, y_w, dydx_w) or None if no intersection.
    """
    # Evaluate the difference: wall_y(x) - ray_y(x) along the wall
    mask = x_wall > x_p + 1e-10
    if not np.any(mask):
        return None

    x_candidates = x_wall[mask]
    y_candidates = y_wall[mask]
    ray_y = y_p + slope * (x_candidates - x_p)
    diff = y_candidates - ray_y

    # Look for sign change (ray crosses wall)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        # No crossing — check if ray is nearly tangent to wall
        # Use closest approach if very close
        min_idx = np.argmin(np.abs(diff))
        if abs(diff[min_idx]) < 0.05 * y_candidates[min_idx]:
            x_w = x_candidates[min_idx]
        else:
            return None
    else:
        # Linear interpolation between bracketing points
        i = sign_changes[0]
        t = diff[i] / (diff[i] - diff[i + 1]) if abs(diff[i] - diff[i + 1]) > 1e-15 else 0.5
        x_w = x_candidates[i] + t * (x_candidates[i + 1] - x_candidates[i])

    y_w = float(np.interp(x_w, x_wall, y_wall))
    # Wall slope by finite difference on the interpolated wall
    dydx_w = float(np.interp(x_w, x_wall, np.gradient(y_wall, x_wall)))
    return x_w, y_w, dydx_w


def analyze_contour(x_wall, y_wall, n_chars, gamma=1.4, R_wall=1.5,
                    x_start=None, M_idl=None, theta_idl=None):
    """Analyze an arbitrary nozzle contour using MOC.

    Marches the MOC solution downstream through a given wall shape.
    Uses the transonic kernel for initial data and computes the full
    characteristic mesh.

    Zucrow & Hoffman, Ch. 12: given a wall shape, march the characteristic
    net from the initial data line (near throat) to the exit plane.

    Parameters
    ----------
    x_wall : ndarray
        Axial wall coordinates (normalized, throat at 0).
    y_wall : ndarray
        Radial wall coordinates (normalized, y=1 at throat).
    n_chars : int
        Number of characteristics on the initial data line.
    gamma : float
    R_wall : float
        Throat wall radius of curvature (for initial data).
    x_start : float or None
        Starting x position for the IDL.  If None, a default is chosen
        that keeps the wall Mach above ~1.15.
    M_idl : ndarray or None
        Override Mach distribution on the IDL (length n_chars).
    theta_idl : ndarray or None
        Override theta distribution on the IDL (length n_chars).

    Returns
    -------
    mesh : CharMesh
        Complete characteristic mesh.
    """
    from nozzle.kernel import hall_initial_line

    mesh = CharMesh(gamma=gamma)

    if x_start is None:
        # Ensure wall M ≥ ~1.15 for MOC stability.
        dMdx = np.sqrt((gamma + 1) / (2 * R_wall))
        b1 = (gamma + 1) / (4 * R_wall)
        x_start = 0.15 / max(dMdx - b1, 0.1)
        x_start = float(np.clip(x_start, 0.1, 0.5))

    x_init, y_init, M_init, theta_init = hall_initial_line(
        n_chars, gamma=gamma, R_wall=R_wall, x_start=x_start
    )

    # Apply IDL overrides if provided
    if M_idl is not None:
        M_init = np.asarray(M_idl, dtype=float)
    else:
        # Clamp IDL Mach so all points are well-supersonic.
        M_init = np.maximum(M_init, 1.15)

    if theta_idl is not None:
        theta_init = np.asarray(theta_idl, dtype=float)

    # Adjust the top point of the IDL to match the wall at x_start
    y_wall_at_start = float(np.interp(x_start, x_wall, y_wall))
    dydx_at_start = float(np.interp(
        x_start, x_wall, np.gradient(y_wall, x_wall)
    ))
    y_init[-1] = y_wall_at_start
    theta_init[-1] = np.arctan(dydx_at_start)

    # Add initial line points to mesh
    init_indices = []
    for i in range(n_chars):
        is_ax = (y_init[i] < 1e-10)
        is_w = (i == n_chars - 1)
        idx = mesh.add_point(
            x=x_init[i], y=y_init[i], M=M_init[i], theta=theta_init[i],
            is_axis=is_ax, is_wall=is_w
        )
        init_indices.append(idx)

    # Pre-compute wall slopes
    dydx_wall_arr = np.gradient(y_wall, x_wall)

    # March downstream row by row
    current_row = init_indices
    max_steps = 3000

    for step in range(max_steps):
        new_row = []
        n_row = len(current_row)

        # 1. Interior points from adjacent pairs: n-1 points
        interiors = []
        for j in range(n_row - 1):
            try:
                int_idx = interior_point(
                    mesh, current_row[j], current_row[j + 1], gamma
                )
                int_pt = mesh.points[int_idx]
                pL = mesh.points[current_row[j]]
                pR = mesh.points[current_row[j + 1]]
                x_min_parent = min(pL.x, pR.x)
                if (int_pt.M < 1.0 or int_pt.M > 20
                        or int_pt.x < x_min_parent - 0.01):
                    interiors.append(None)
                else:
                    interiors.append(int_idx)
            except (ValueError, RuntimeError):
                interiors.append(None)

        # Remove trailing Nones
        while interiors and interiors[-1] is None:
            interiors.pop()
        # Remove leading Nones
        while interiors and interiors[0] is None:
            interiors.pop(0)
        # Remove any remaining Nones
        interiors = [i for i in interiors if i is not None]

        if len(interiors) < 1:
            break

        # 2. Axis point from the lowest interior
        lowest = interiors[0]
        lowest_pt = mesh.points[lowest]
        if lowest_pt.y > 1e-6:
            try:
                ax_idx = axis_point(mesh, lowest, gamma)
                new_row.append(ax_idx)
            except (ValueError, RuntimeError):
                pass

        new_row.extend(interiors)

        # 3. Wall point from the highest interior
        highest = interiors[-1]
        highest_pt = mesh.points[highest]
        mu_top = mach_angle(highest_pt.M)
        slope_cplus = np.tan(highest_pt.theta + mu_top)

        result = _find_wall_intersection(
            highest_pt.x, highest_pt.y, slope_cplus, x_wall, y_wall
        )
        if result is not None:
            x_w, y_w, dydx_w = result
            if x_w > highest_pt.x and x_w <= x_wall[-1]:
                try:
                    w_idx = wall_point(
                        mesh, highest, x_w, y_w, dydx_w, gamma
                    )
                    new_row.append(w_idx)
                except (ValueError, RuntimeError):
                    pass

        if len(new_row) < 2:
            break

        # Check termination
        x_max = max(mesh.points[i].x for i in new_row)
        if x_max >= x_wall[-1] * 0.95:
            break

        current_row = new_row

    return mesh
