"""Nozzle contour generators.

All contours return (x, y) arrays in throat-radius-normalized coordinates
with throat at x=0, y=1 (wall) or y=0 (axis).

References:
- Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed.
- Rao, *Jet Propulsion* 1958; *ARS Journal* 1960
- Anderson, *Modern Compressible Flow*, 3rd ed.
"""

import numpy as np
from pathlib import Path
from nozzle.gas import area_mach_ratio, prandtl_meyer, mach_from_prandtl_meyer, mach_angle


def conical_nozzle(half_angle_deg, area_ratio, n_points=200):
    """Generate a conical nozzle contour.

    Parameters
    ----------
    half_angle_deg : float
        Cone half-angle in degrees.
    area_ratio : float
        Exit-to-throat area ratio Ae/A*.
    n_points : int
        Number of points along the contour.

    Returns
    -------
    x : ndarray
        Axial coordinates (throat at x=0), normalized by throat radius.
    y : ndarray
        Radial coordinates, normalized by throat radius (y=1 at throat).

    Notes
    -----
    Pure geometry: y_exit = sqrt(Ae/A*), length = (y_exit - 1) / tan(α).
    Sutton & Biblarz Section 3.4.
    """
    alpha = np.radians(half_angle_deg)
    y_throat = 1.0
    y_exit = np.sqrt(area_ratio)  # Ae/A* = (re/r*)^2
    length = (y_exit - y_throat) / np.tan(alpha)

    x = np.linspace(0, length, n_points)
    y = y_throat + x * np.tan(alpha)

    return x, y


def conical_divergence_loss(half_angle_deg):
    """Divergence loss factor λ = (1 + cos α) / 2 for a conical nozzle.

    Sutton & Biblarz Eq. 3-34; Malina (1940).
    λ=1 for parallel flow, λ=0.9830 for 15° half-angle.
    """
    alpha = np.radians(half_angle_deg)
    return (1 + np.cos(alpha)) / 2


def rao_parabolic_nozzle(area_ratio, bell_fraction=0.8, gamma=1.4,
                         n_points=200, upstream_radius_ratio=1.5):
    """Generate a Rao parabolic (bell) nozzle contour.

    Approximation from Rao (1960) "Approximation of Optimum Thrust Nozzle
    Contour", ARS Journal. Geometry: upstream circular arc (throat to
    inflection) followed by a parabolic section (inflection to exit).

    Parameters
    ----------
    area_ratio : float
        Exit-to-throat area ratio Ae/A*.
    bell_fraction : float
        Fraction of equivalent 15° conical nozzle length (0.6 to 1.0).
    gamma : float
        Ratio of specific heats.
    n_points : int
        Number of points along the contour.
    upstream_radius_ratio : float
        Radius of upstream circular arc / throat radius. Typically 1.5.

    Returns
    -------
    x : ndarray
        Axial coordinates, normalized by throat radius.
    y : ndarray
        Radial coordinates, normalized by throat radius.
    theta_n : float
        Inflection angle [radians].
    theta_e : float
        Exit angle [radians].

    Notes
    -----
    θ_n and θ_e from Rao (1960) lookup / Sutton Table 3-4 interpolation.
    """
    if area_ratio < 1.5:
        raise ValueError(
            f"Area ratio {area_ratio} too small for Rao parabolic nozzle "
            f"(minimum 1.5)"
        )

    theta_n, theta_e = rao_angles(area_ratio, bell_fraction)

    # Equivalent 15° conical length
    y_exit = np.sqrt(area_ratio)
    L_conical = (y_exit - 1.0) / np.tan(np.radians(15))
    L_nozzle = bell_fraction * L_conical

    # Upstream circular arc: from throat (0, 1) to inflection point
    R_u = upstream_radius_ratio  # normalized by r_throat

    # Scale R_u down if the arc would consume too much of the nozzle length
    x_n_limit = 0.55 * L_nozzle
    if R_u * np.sin(theta_n) > x_n_limit:
        R_u = x_n_limit / np.sin(theta_n)
    # Arc center is at (0, 1 + R_u) — above throat on wall
    # Actually, the arc center is downstream and above:
    # The arc goes from the throat tangent to the inflection angle θ_n
    x_center = 0.0
    y_center = 1.0 + R_u

    # Arc from angle -π/2 (pointing down = throat tangent) to angle (-π/2 + θ_n)
    arc_angles = np.linspace(-np.pi / 2, -np.pi / 2 + theta_n, n_points // 3)
    x_arc = x_center + R_u * np.cos(arc_angles)  # wrong: needs to be sin-based
    y_arc = y_center + R_u * np.sin(arc_angles)

    # Inflection point N: end of arc
    x_n = x_arc[-1]
    y_n = y_arc[-1]

    # Exit point E
    x_e = L_nozzle
    y_e = y_exit

    # Parabolic section from N(x_n, y_n, slope=tan(θ_n)) to E(x_e, y_e, slope=tan(θ_e))
    # Parametric parabola: x(t) = (1-t)²·x_n + 2t(1-t)·x_q + t²·x_e
    #                      y(t) = (1-t)²·y_n + 2t(1-t)·y_q + t²·y_e
    # where Q is the intersection of tangent lines at N and E.
    # Tangent at N: slope = tan(θ_n), tangent at E: slope = tan(θ_e)
    # Line from N: y - y_n = tan(θ_n)·(x - x_n)
    # Line from E: y - y_e = tan(θ_e)·(x - x_e)
    tan_n = np.tan(theta_n)
    tan_e = np.tan(theta_e)

    if abs(tan_n - tan_e) < 1e-12:
        # Degenerate: straight line
        x_q = (x_n + x_e) / 2
        y_q = (y_n + y_e) / 2
    else:
        x_q = (y_e - y_n + tan_n * x_n - tan_e * x_e) / (tan_n - tan_e)
        y_q = y_n + tan_n * (x_q - x_n)

    t = np.linspace(0, 1, 2 * n_points // 3 + 1)
    x_par = (1 - t)**2 * x_n + 2 * t * (1 - t) * x_q + t**2 * x_e
    y_par = (1 - t)**2 * y_n + 2 * t * (1 - t) * y_q + t**2 * y_e

    # Combine arc and parabola (skip duplicate point)
    x = np.concatenate([x_arc, x_par[1:]])
    y = np.concatenate([y_arc, y_par[1:]])

    # Defensive monotonicity filter: remove any non-monotonic points
    mask = np.ones(len(x), dtype=bool)
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            mask[i] = False
    x = x[mask]
    y = y[mask]

    return x, y, theta_n, theta_e


def rao_angles(area_ratio, bell_fraction=0.8):
    """Lookup θ_n and θ_e for Rao parabolic approximation.

    Interpolated from Sutton & Biblarz Table 3-4 / Rao (1960).
    Returns angles in radians.

    Parameters
    ----------
    area_ratio : float
        Ae/A*.
    bell_fraction : float
        Bell length as fraction of 15° conical (0.6–1.0).

    Returns
    -------
    theta_n : float
        Inflection angle [radians].
    theta_e : float
        Exit angle [radians].
    """
    # Sutton & Biblarz Table 3-4 data (degrees)
    # Columns: AR, θ_n(60%), θ_e(60%), θ_n(80%), θ_e(80%), θ_n(90%), θ_e(90%), θ_n(100%), θ_e(100%)
    _table = np.array([
        # AR   θn60  θe60  θn80  θe80  θn90  θe90  θn100  θe100
        [2,    21.0, 11.0, 17.0,  8.5, 15.5,  7.5, 14.0,   6.5],
        [3,    25.0, 14.5, 21.0, 10.5, 19.5,  9.5, 18.0,   8.0],
        [4,    28.0, 17.0, 24.0, 12.5, 23.0, 11.0, 21.0,   9.5],
        [5,    28.0, 16.0, 24.5, 12.0, 23.0, 10.0, 21.5,   8.5],
        [6,    29.0, 15.5, 25.0, 11.5, 23.5,  9.5, 22.0,   8.0],
        [8,    30.0, 14.5, 26.0, 10.5, 24.0,  8.5, 22.5,   7.0],
        [10,   31.0, 13.5, 26.5, 10.0, 24.5,  8.0, 23.0,   6.5],
        [15,   32.0, 12.5, 27.5,  9.0, 25.5,  7.0, 24.0,   5.5],
        [20,   33.0, 11.5, 28.5,  8.5, 26.0,  6.5, 24.5,   5.0],
        [25,   34.0, 11.0, 29.0,  8.0, 26.5,  6.0, 25.0,   4.5],
        [30,   34.5, 10.5, 29.5,  7.5, 27.0,  5.5, 25.5,   4.0],
        [40,   35.5, 10.0, 30.5,  7.0, 28.0,  5.0, 26.0,   3.5],
        [50,   36.0,  9.5, 31.0,  6.5, 28.5,  4.5, 26.5,   3.0],
        [100,  38.0,  8.0, 33.0,  5.0, 30.0,  3.5, 28.0,   2.0],
    ])

    ars = _table[:, 0]
    bell_fractions = np.array([0.6, 0.8, 0.9, 1.0])

    # θ_n columns: 1, 3, 5, 7; θ_e columns: 2, 4, 6, 8
    theta_n_cols = _table[:, [1, 3, 5, 7]]
    theta_e_cols = _table[:, [2, 4, 6, 8]]

    # Clamp area_ratio to table bounds
    ar_clamped = np.clip(area_ratio, ars[0], ars[-1])
    bf_clamped = np.clip(bell_fraction, bell_fractions[0], bell_fractions[-1])

    # 2D interpolation: first along AR, then along bell fraction
    from scipy.interpolate import RegularGridInterpolator

    interp_n = RegularGridInterpolator(
        (ars, bell_fractions), theta_n_cols,
        method='linear', bounds_error=False, fill_value=None
    )
    interp_e = RegularGridInterpolator(
        (ars, bell_fractions), theta_e_cols,
        method='linear', bounds_error=False, fill_value=None
    )

    point = np.array([[float(ar_clamped), float(bf_clamped)]])
    theta_n_deg = float(interp_n(point)[0])
    theta_e_deg = float(interp_e(point)[0])

    return np.radians(theta_n_deg), np.radians(theta_e_deg)


def truncated_ideal_contour(M_exit, truncation_fraction=0.8, n_chars=50,
                            gamma=1.4):
    """Generate a truncated ideal contour (TIC) nozzle.

    Truncates an MLN at a specified fraction of its full length.
    The truncated contour is shorter but has non-uniform, non-parallel
    exit flow.

    Parameters
    ----------
    M_exit : float
        Design exit Mach number of the full (untruncated) MLN.
    truncation_fraction : float
        Fraction of full MLN length to keep (0 < f <= 1).
    n_chars : int
        Number of characteristic lines.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    x_wall : ndarray
        Axial wall coordinates (normalized by throat radius).
    y_wall : ndarray
        Radial wall coordinates (normalized by throat radius).
    mesh : CharMesh
        Full MLN characteristic mesh (used for exit plane visualization;
        TIC performance is computed via quasi_1d_performance on the
        truncated wall contour).
    """
    # Generate full MLN
    x_full, y_full, mesh = minimum_length_nozzle(M_exit, n_chars, gamma)

    if truncation_fraction >= 1.0:
        return x_full, y_full, mesh

    # Truncate at the specified fraction of full length
    L_full = x_full[-1] - x_full[0]
    x_trunc = x_full[0] + truncation_fraction * L_full

    # Keep all points up to truncation, interpolate last point
    mask = x_full <= x_trunc
    x_wall = x_full[mask]
    y_wall = y_full[mask]

    # Interpolate exit point at exact truncation location
    if x_trunc < x_full[-1]:
        y_trunc = np.interp(x_trunc, x_full, y_full)
        x_wall = np.append(x_wall, x_trunc)
        y_wall = np.append(y_wall, y_trunc)

    return x_wall, y_wall, mesh


def minimum_length_nozzle(M_exit, n_chars=50, gamma=1.4):
    """Generate a minimum-length nozzle contour via MOC.

    Uses the method of characteristics to design an MLN with uniform,
    parallel exit flow at the specified Mach number.

    Anderson MCF Section 11.11; Example 11.2 (γ=1.4, M_exit=2.4).

    Parameters
    ----------
    M_exit : float
        Design exit Mach number.
    n_chars : int
        Number of characteristic lines (resolution).
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    x_wall : ndarray
        Axial wall coordinates (normalized by throat radius).
    y_wall : ndarray
        Radial wall coordinates (normalized by throat radius).
    mesh : CharMesh
        Complete characteristic mesh.
    """
    from nozzle.moc import design_mln

    mesh = design_mln(M_exit, n_chars, gamma)
    x_wall, y_wall = mesh.get_wall_contour()

    # Sort by x and ensure throat is included
    sort_idx = np.argsort(x_wall)
    x_wall = x_wall[sort_idx]
    y_wall = y_wall[sort_idx]

    return x_wall, y_wall, mesh


def length_constrained_nozzle(M_exit, target_length, n_chars=30, gamma=1.4,
                              tol=1e-3):
    """Find optimal nozzle for a given exit Mach and length budget.

    Bisects over M_design >= M_exit: designs MLN(M_design), truncates at
    target_length, checks if exit area ratio matches area_mach_ratio(M_exit).

    Parameters
    ----------
    M_exit : float
        Required exit Mach number (defines target area ratio).
    target_length : float
        Maximum nozzle length in r*-normalized units.
    n_chars : int
        MOC characteristic count for MLN generation.
    gamma : float
        Ratio of specific heats.
    tol : float
        Bisection tolerance on area ratio relative error.

    Returns
    -------
    x_wall, y_wall : ndarray
        Truncated wall contour.
    mesh : CharMesh
        MLN mesh at the converged M_design.
    info : dict
        M_design, truncation_fraction, area_ratio, iterations.
    """
    target_ar = area_mach_ratio(M_exit, gamma)

    # Try the full MLN at M_exit first — if it fits, it's already optimal
    x_full, y_full, mesh_full = minimum_length_nozzle(M_exit, n_chars, gamma)
    L_full = x_full[-1] - x_full[0]

    if L_full <= target_length:
        info = {
            'M_design': M_exit,
            'truncation_fraction': 1.0,
            'area_ratio': target_ar,
            'iterations': 0,
        }
        return x_full, y_full, mesh_full, info

    # Bisect: find M_design such that y(target_length)^2 == target AR
    M_low = M_exit
    # Find M_high: scan upward until MLN is long enough that y at target_length
    # gives area ratio >= target_ar
    M_high = M_exit * 1.5
    for _ in range(20):
        x_h, y_h, _ = minimum_length_nozzle(M_high, n_chars, gamma)
        if x_h[-1] - x_h[0] >= target_length:
            y_at_L = np.interp(x_h[0] + target_length, x_h, y_h)
            if y_at_L ** 2 >= target_ar:
                break
        M_high *= 1.5
    else:
        import warnings
        warnings.warn(
            f"Could not bracket M_design for target_length={target_length}. "
            f"Returning best attempt."
        )

    # Bisection loop
    iterations = 0
    max_iter = 50
    while iterations < max_iter:
        M_mid = (M_low + M_high) / 2
        x_mid, y_mid, mesh_mid = minimum_length_nozzle(M_mid, n_chars, gamma)
        L_mid = x_mid[-1] - x_mid[0]

        if L_mid < target_length:
            # MLN fits entirely — M_mid is too low (not enough over-design)
            M_low = M_mid
            iterations += 1
            continue

        y_at_L = np.interp(x_mid[0] + target_length, x_mid, y_mid)
        ar_at_L = y_at_L ** 2

        rel_err = (ar_at_L - target_ar) / target_ar
        iterations += 1

        if abs(rel_err) < tol:
            break
        elif ar_at_L < target_ar:
            # Not enough area — need higher M_design
            M_low = M_mid
        else:
            # Too much area — need lower M_design
            M_high = M_mid

    # Truncate the converged contour at target_length
    x_trunc_limit = x_mid[0] + target_length
    mask = x_mid <= x_trunc_limit
    x_wall = x_mid[mask]
    y_wall = y_mid[mask]

    # Interpolate exact endpoint
    if x_trunc_limit < x_mid[-1]:
        y_trunc = np.interp(x_trunc_limit, x_mid, y_mid)
        x_wall = np.append(x_wall, x_trunc_limit)
        y_wall = np.append(y_wall, y_trunc)

    trunc_frac = target_length / L_mid if L_mid > 0 else 1.0

    info = {
        'M_design': M_mid,
        'truncation_fraction': trunc_frac,
        'area_ratio': y_wall[-1] ** 2,
        'iterations': iterations,
    }

    return x_wall, y_wall, mesh_mid, info


def sivells_nozzle(M_exit, gamma=1.4, rc=1.5, inflection_angle_deg=None,
                   n_char=41, n_axis=21, nx=13, ix=0, ie=0,
                   downstream=False, ip=10, md=None, nd=None, nf=None):
    """Generate a nozzle contour using Sivells' CONTUR method.

    Uses the Method of Characteristics with mass flow integration to compute
    the wall contour. With downstream=False (default), only the upstream
    portion (throat to inflection) is computed. With downstream=True, the
    full contour from throat to exit is produced.

    Parameters
    ----------
    M_exit : float
        Design exit Mach number.
    gamma : float
        Ratio of specific heats.
    rc : float
        Throat radius of curvature / throat radius (default 1.5).
    inflection_angle_deg : float or None
        Wall angle at inflection point in degrees. If None, auto-derived
        as 30/M_exit (gives ~15° at M=2, ~7.5° at M=4).
    n_char : int
        Number of characteristic points (default 41).
    n_axis : int
        Number of axis points (default 21).
    nx : int
        Axis spacing exponent parameter (default 13, i.e. exponent = 1.3).
    ix : int
        Distribution type: 0 for 3rd-degree (default), nonzero for 4th-degree.
    ie : int
        0 for planar (default), 1 for axisymmetric.
    downstream : bool
        If True, compute the full contour (throat to exit). Default False.
    ip : int
        Downstream polynomial interpolation parameter (default 10).
    md : int or None
        Number of downstream characteristics. If None, defaults to n_char.
    nd : int or None
        Number of downstream axis points. If None, auto-computed.
    nf : int or None
        Number of exit characteristic points (negative). If None, auto-computed.

    Returns
    -------
    x : ndarray
        Axial coordinates (throat at x=0), normalized by throat radius.
    y : ndarray
        Radial coordinates, normalized by throat radius (y=1 at throat).
    """
    from nozzle.sivells import sivells_axial, sivells_perfc

    if inflection_angle_deg is None:
        inflection_angle_deg = 30.0 / M_exit

    # bmach must be > 1 and < M_exit; use validated heuristic
    bmach = min(max(1.5, 0.8 * M_exit), M_exit - 0.1)

    axial = sivells_axial(
        gamma=gamma,
        eta_deg=inflection_angle_deg,
        rc=rc,
        bmach=bmach,
        cmach=M_exit,
        ie=ie,
        n_char=n_char,
        n_axis=n_axis,
        ix=ix,
        nx=nx,
    )

    perfc = sivells_perfc(axial, gamma=gamma, ie=ie)

    # Convert CONTUR coordinates to r*-normalized (throat at x=0, y=1)
    yo = axial['yo']   # throat half-height in CONTUR coords
    xo = axial['xo']   # throat x-position in CONTUR coords
    wax = perfc['wax']  # wall x in CONTUR coords
    way = perfc['way']  # wall y in CONTUR coords

    x_up = (wax - xo) / yo
    y_up = way / yo

    if not downstream:
        return x_up, y_up

    # --- Downstream contour ---
    from nozzle.sivells import sivells_axial_downstream, sivells_perfc_downstream

    ds_axial = sivells_axial_downstream(
        axial, gamma=gamma, ie=ie, ip=ip, md=md, nd=nd, nf=nf,
    )

    ds_perfc = sivells_perfc_downstream(
        axial, ds_axial, gamma=gamma, ie=ie,
    )

    # Convert downstream wall to r*-normalized
    ds_wax = ds_perfc['wax']
    ds_way = ds_perfc['way']
    x_dn = (ds_wax - xo) / yo
    y_dn = ds_way / yo

    # Bridge the gap between upstream and downstream at inflection.
    # Both endpoints have the same wall angle (inflection angle), so
    # a straight-line interpolation is appropriate for the transition.
    x_gap_start = x_up[-1]
    y_gap_start = y_up[-1]
    x_gap_end = x_dn[0]
    y_gap_end = y_dn[0]
    gap = x_gap_end - x_gap_start

    if gap > 0.01:
        # Interpolate with ~same spacing as upstream wall
        dx_up = np.mean(np.diff(x_up[-5:])) if len(x_up) > 5 else 0.1
        n_bridge = max(5, int(gap / dx_up))
        # Exclude endpoints (already in upstream/downstream)
        t = np.linspace(0, 1, n_bridge + 2)[1:-1]
        x_bridge = x_gap_start + t * gap
        y_bridge = y_gap_start + t * (y_gap_end - y_gap_start)
        x = np.concatenate([x_up, x_bridge, x_dn])
        y = np.concatenate([y_up, y_bridge, y_dn])
    else:
        # No gap or overlap — just skip duplicate
        if gap > -0.01:
            x_dn = x_dn[1:]
            y_dn = y_dn[1:]
        x = np.concatenate([x_up, x_dn])
        y = np.concatenate([y_up, y_dn])

    return x, y


def load_contour_csv(path, r_throat=None, x_throat=None):
    """Load a nozzle wall contour from a CSV file.

    Expects two columns: x (axial) and y (radial). Can optionally
    normalize by throat radius and shift origin to throat.

    Parameters
    ----------
    path : str or Path
        Path to CSV file. First two columns are x, y.
        Lines starting with '#' are treated as comments.
    r_throat : float or None
        Throat radius for normalization. If None, the minimum y
        value is used as the throat radius.
    x_throat : float or None
        Axial position of the throat in the raw data. If None,
        the throat is located at the minimum y position.

    Returns
    -------
    x : ndarray
        Axial coordinates (throat at x=0), normalized by r*.
    y : ndarray
        Radial coordinates, normalized by r*.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file has fewer than 2 data points or invalid format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Contour file not found: {path}")

    data = np.loadtxt(path, delimiter=',', comments='#')
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected CSV with at least 2 columns (x, y), got shape {data.shape}"
        )
    if data.shape[0] < 2:
        raise ValueError(f"Need at least 2 data points, got {data.shape[0]}")

    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)

    # Sort by x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Locate throat (minimum y if not specified)
    if x_throat is not None:
        x = x - x_throat
    else:
        i_throat = np.argmin(y)
        x = x - x[i_throat]

    # Normalize by throat radius
    if r_throat is not None:
        x = x / r_throat
        y = y / r_throat
    else:
        y_min = np.min(y)
        if y_min > 0:
            x = x / y_min
            y = y / y_min

    return x, y


def convergent_section(contraction_ratio=3.0, convergent_half_angle_deg=30.0,
                       rc_upstream=1.5, rc_downstream=0.382, n_points=100):
    """Generate a convergent (subsonic) nozzle section.

    Constructs a wall contour from the chamber to the throat, using a
    standard four-segment geometry: chamber cylinder, upstream arc,
    straight converging line, and downstream arc.

    All coordinates are normalized by throat radius (r*). The throat is
    at (0, 1) and the contour extends to negative x values.

    Parameters
    ----------
    contraction_ratio : float
        Chamber-to-throat area ratio Ac/A* (default 3.0).
    convergent_half_angle_deg : float
        Half-angle of the straight converging section in degrees (default 30).
    rc_upstream : float
        Upstream circular arc radius / throat radius (default 1.5).
    rc_downstream : float
        Downstream circular arc radius / throat radius (default 0.382).
    n_points : int
        Total number of points in the output contour (default 100).

    Returns
    -------
    x : ndarray
        Axial coordinates (x <= 0), normalized by throat radius.
    y : ndarray
        Radial coordinates, normalized by throat radius. y[-1] = 1.0.

    Raises
    ------
    ValueError
        If the geometry is invalid (arcs overlap, chamber too small, etc.).
    """
    alpha = np.radians(convergent_half_angle_deg)
    y_chamber = np.sqrt(contraction_ratio)  # Ac/A* = (rc/r*)^2
    R_d = rc_downstream
    R_u = rc_upstream

    if abs(np.tan(alpha)) < 1e-12:
        raise ValueError("Convergent half-angle too small")

    # Auto-scale arc radii to fit the geometry.
    # We iteratively reduce radii until the straight line has positive length.
    for _ in range(20):
        y_d_center = 1.0 + R_d
        x_d_junction = R_d * np.cos(-np.pi / 2 - alpha)
        y_d_junction = y_d_center + R_d * np.sin(-np.pi / 2 - alpha)

        y_u_center = y_chamber - R_u
        y_u_tangent = y_u_center + R_u * np.cos(alpha)
        x_u_tangent = x_d_junction - (y_u_tangent - y_d_junction) / np.tan(alpha)
        x_u_center = x_u_tangent + R_u * np.sin(alpha)

        if x_u_center < x_d_junction - 0.01:
            break
        R_u *= 0.7
        R_d *= 0.7
    else:
        raise ValueError(
            f"Convergent geometry invalid: cannot fit arcs. "
            f"Try a larger contraction ratio or smaller half-angle."
        )

    # --- Downstream arc (closest to throat) ---
    x_d_center = 0.0
    y_d_center = 1.0 + R_d
    theta_d_start = -np.pi / 2 - alpha
    theta_d_end = -np.pi / 2

    x_d_junction = x_d_center + R_d * np.cos(theta_d_start)
    y_d_junction = y_d_center + R_d * np.sin(theta_d_start)

    # --- Upstream arc ---
    y_u_center = y_chamber - R_u
    y_u_tangent = y_u_center + R_u * np.cos(alpha)
    x_u_tangent = x_d_junction - (y_u_tangent - y_d_junction) / np.tan(alpha)
    x_u_center = x_u_tangent + R_u * np.sin(alpha)

    # --- Build segments (upstream → downstream) ---
    n_seg = max(n_points // 4, 5)

    # 1. Chamber cylinder
    chamber_length = min(0.5 * abs(x_u_center), 1.0)
    if chamber_length < 0.01:
        chamber_length = 0.5
    x_cyl = np.linspace(x_u_center - chamber_length, x_u_center, n_seg)
    y_cyl = np.full_like(x_cyl, y_chamber)

    # 2. Upstream arc: from pi/2 (chamber) to pi/2 + alpha (line tangent)
    theta_u = np.linspace(np.pi / 2, np.pi / 2 + alpha, n_seg)
    x_uarc = x_u_center + R_u * np.cos(theta_u)
    y_uarc = y_u_center + R_u * np.sin(theta_u)

    # 3. Straight line
    x_line = np.linspace(x_u_tangent, x_d_junction, n_seg)
    y_line = np.linspace(y_u_tangent, y_d_junction, n_seg)

    # 4. Downstream arc: from -pi/2 - alpha to -pi/2 (throat)
    theta_d = np.linspace(theta_d_start, theta_d_end, n_seg)
    x_darc = x_d_center + R_d * np.cos(theta_d)
    y_darc = y_d_center + R_d * np.sin(theta_d)

    # Concatenate (skip duplicate junction points)
    x = np.concatenate([x_cyl, x_uarc[1:], x_line[1:], x_darc[1:]])
    y = np.concatenate([y_cyl, y_uarc[1:], y_line[1:], y_darc[1:]])

    return x, y
