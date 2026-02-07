"""Port of Sivells' CONTUR program utility functions.

Line-by-line port of the validated F90 code at
docs/references/external_codes/contur/src/

Reference:
    Sivells, J.C., "A computer program for the aerodynamic design of
    axisymmetric and planar nozzles for supersonic and hypersonic wind
    tunnels," AEDC-TR-78-63, December 1978.

    F90 port by Rona & Zavalan (U. Leicester), validated against Mach 4
    test case. See docs/references/external_codes/contur/

Conventions:
    - All angles in radians unless noted
    - Velocity ratio w = V/a* (velocity normalized by critical speed of sound)
    - Coordinates normalized by throat radius r*
    - ie=1 for axisymmetric, ie=0 for planar
    - qt = 1/(1+ie): 0.5 for axisymmetric, 1.0 for planar
"""

import numpy as np
from nozzle.gas import prandtl_meyer, mach_from_prandtl_meyer, mach_angle


# ==========================================================================
# Gas constant helpers
# ==========================================================================

def contur_gas_constants(gamma):
    """Compute CONTUR's derived gas constants from γ.

    From main.f lines 78-89.

    Returns dict with keys: gm, g1, g2, g3, g4, g5, g6, g7, g8, g9, ga, rga, qt
    """
    gm = gamma - 1.0
    g1 = 1.0 / gm
    g9 = 2.0 * g1
    g8 = 1.0 / g9
    g7 = 1.0 + g8
    g6 = 1.0 / g7
    g5 = g8 * g6
    rga = 2.0 * g5
    ga = 1.0 / rga
    g4 = np.sqrt(g5)
    g3 = ga / 2.0
    g2 = 1.0 / g4
    return {
        'gm': gm, 'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4,
        'g5': g5, 'g6': g6, 'g7': g7, 'g8': g8, 'g9': g9,
        'ga': ga, 'rga': rga,
    }


def contur_fg_constants(gamma, ie=1):
    """Compute CONTUR's transonic coefficients (fg module) from γ.

    From axial.f lines 126-157.

    Parameters
    ----------
    gamma : float
    ie : int
        1 for axisymmetric, 0 for planar.

    Returns dict with keys: gc, gd, ge, gf, gh, gi, gj, gk, gr, gs, hb, hc, he
    """
    gm = gamma - 1.0
    qt = 1.0 / (1 + ie)

    gc = (2.0 * gamma / qt - 3.0) / 6.0 / (3 + ie)
    ge = (3.0 * (8 + ie) - 4.0 * gamma / qt) / 3.0 / (7 + ie)
    gh = (15.0 + (2 - 6 * ie) * gamma) / 12.0 / (5 + ie)
    gj = (gamma * (gamma + 9.25 * ie - 26.5) + 0.75 * (6 - ie)) / 12.0 / (3 - ie)
    gk = (gamma * (gamma + 2.25 * ie - 16.5) + 2.25 * (2 + ie)) / 6.0
    gr = (15.0 - (1 + 9 * ie) * gamma) / (15 + ie) / 18.0
    hb = (14.0 * gamma - 75.0 + 18 * ie) / (270.0 + 18 * ie)

    if ie == 1:
        gd = (gm * (652.0 * gm + 1319.0) + 1000.0) / 6912.0
        gf = (3612.0 + gm * (751.0 + gm * 754.0)) / 2880.0
        gi = (909.0 + gamma * (270.0 + gamma * 412.0)) / 10368.0
        gs = (gamma * (gamma * 2708.0 + 2079.0) + 2115.0) / 82944.0
        hc = (gamma * (2364.0 * gamma - 3915.0) + 14337.0) / 82944.0
        he = (gamma * (64.0 * gamma + 117.0) - 1026.0) / 1152.0
    else:
        gd = (gm * (32.0 * gm - 14.0) + 221.0) / 1080.0
        gf = (4230.0 + gm * (211.0 + gm * 334.0)) / 3780.0
        gi = (738.0 + gamma * (273.0 - gamma * 82.0)) / 7560.0
        gs = (gamma * (gamma * 782.0 + 3507.0) + 7767.0) / 272160.0
        hc = (gamma * (274.0 * gamma - 861.0) + 4464.0) / 17010.0
        he = (gamma * (32.0 * gamma + 87.0) - 561.0) / 540.0

    return {
        'gc': gc, 'gd': gd, 'ge': ge, 'gf': gf,
        'gh': gh, 'gi': gi, 'gj': gj, 'gk': gk,
        'gr': gr, 'gs': gs, 'hb': hb, 'hc': hc, 'he': he,
    }


# ==========================================================================
# cubic.f — Cardano's formula for cubic equations
# ==========================================================================

def cubic_solve(ea, eb, ec, ed):
    """Find the smallest positive real root of ea·x³ + eb·x² + ec·x + ed = 0.

    Port of cubic.f (61 lines). Uses Cardano's formula.

    Parameters
    ----------
    ea, eb, ec, ed : float
        Coefficients of the cubic: ea*x^3 + eb*x^2 + ec*x + ed = 0.

    Returns
    -------
    float
        Smallest positive real root.

    Raises
    ------
    ValueError
        If no positive real root exists.
    """
    e3 = eb / 3.0
    q1 = ea * ec / 3.0 - e3**2
    r1 = ea * (e3 * ec - ea * ed) / 2.0 - e3**3
    qr = q1**3 + r1**2
    rq = np.sqrt(abs(qr))
    q = np.sqrt(abs(q1))
    b = np.sign(r1) if r1 != 0 else 1.0

    cbb = -1.0
    cbc = -1.0

    if qr <= 0:
        # Three real roots
        if qr != 0:
            a = np.arcsin(-rq / q1 / q) / 3.0
        else:
            a = 0.0
        csa = np.cos(a)
        csna = np.sqrt(3.0) * np.sin(a)
        cba = (2.0 * b * q * csa - e3) / ea
        cbb = -(b * q * (csa + csna) + e3) / ea
        cbc = -(b * q * (csa - csna) + e3) / ea
    else:
        # One real root, two complex
        cbt1 = 0.0
        cbt2 = 0.0
        t1 = r1 + rq
        t2 = r1 - rq
        if t1 != 0:
            cbt1 = np.sign(t1) * np.exp(np.log(abs(t1)) / 3.0)
        if t2 != 0:
            cbt2 = np.sign(t2) * np.exp(np.log(abs(t2)) / 3.0)
        cba = (cbt1 + cbt2 - e3) / ea

    # Select smallest positive root
    roots = [cba, cbb, cbc]
    positive = [r for r in roots if r > 0]
    if not positive:
        raise ValueError(
            f"No positive real root: ea={ea}, eb={eb}, ec={ec}, ed={ed}, "
            f"roots={roots}"
        )
    return min(positive)


# ==========================================================================
# toric.f — Throat radius of curvature from velocity gradient
# ==========================================================================

def throat_curvature(wip, se, gamma=1.4, ie=1):
    """Compute throat radius of curvature from velocity gradient.

    Port of toric.f (20 lines).

    Parameters
    ----------
    wip : float
        Velocity gradient dW/dx at a point.
    se : float
        Streamline distance (2*sin(eta/2) for axisymmetric).
    gamma : float
    ie : int
        1 for axisymmetric, 0 for planar.

    Returns
    -------
    float
        Radius of curvature (normalized).
    """
    fg = contur_fg_constants(gamma, ie)
    gc, gd, ge, gf = fg['gc'], fg['gd'], fg['ge'], fg['gf']
    g = contur_gas_constants(gamma)
    g7 = g['g7']
    qt = 1.0 / (1 + ie)

    fw = wip * se * np.sqrt(qt * (gamma + 1.0))
    trr = fw * (1.0 + (gc + (3.0 * gc**2 - gd) * fw**2) * fw**2)

    for _ in range(100):
        tr2 = trr**2
        ie_val = 1.0 / qt - 1.0
        tk = (1.0 - g7 * (1.0 + (ge + gf * tr2) * tr2) * tr2**2
              / (45.0 + 3 * ie_val))**qt
        ff = fw / tk - trr * (1.0 - tr2 * (gc - gd * tr2))
        fp = 1.0 - tr2 * (3.0 * gc - 5.0 * gd * tr2)
        trr = trr + ff / fp
        if abs(ff) <= 0.1:
            break

    return 1.0 / trr**2


# ==========================================================================
# conic.f — Mach number derivatives in radial flow
# ==========================================================================

def conic_derivatives(xm, gamma=1.4, ie=1):
    """Compute Mach number derivatives in radial flow.

    Port of conic.f (24 lines).

    Parameters
    ----------
    xm : float
        Mach number.
    gamma : float
    ie : int

    Returns
    -------
    b : ndarray of shape (4,)
        b[0] = r/r* (radius ratio, r* normalized)
        b[1] = dM/dr * r*
        b[2] = d²M/dr² * r*²
        b[3] = d³M/dr³ * r*³
    """
    g = contur_gas_constants(gamma)
    g5, g6, g8, ga = g['g5'], g['g6'], g['g8'], g['ga']
    qt = 1.0 / (1 + ie)

    xmm = xm * xm
    xmm1 = xmm - 1.0
    xmm2 = xmm1**2
    bmm = 1.0 + g8 * xmm
    area = (g6 + g5 * xmm)**ga / xm

    b = np.zeros(4)
    b[0] = area**qt
    b[1] = xm * bmm / qt / xmm1 / b[0]
    c2 = 2.0 - (1.0 + 3.0 * g8) / qt
    c4 = g8 / qt - 1.0
    cmm = xmm * (c2 + xmm * c4) - 1.0 - 1.0 / qt
    b[2] = b[1] * cmm / xmm2 / b[0]
    dmm = (4.0 * c4 * xmm + 2.0 * c2) / cmm - 4.0 / xmm1
    b[3] = b[2] * (b[2] / b[1] + xm * b[1] * dmm - 1.0 / b[0])
    return b


# ==========================================================================
# sorce.f — Velocity derivatives in radial flow
# ==========================================================================

def source_derivatives(w, gamma=1.4, ie=1):
    """Compute velocity derivatives in radial flow.

    Port of sorce.f (25 lines).

    Parameters
    ----------
    w : float
        Velocity ratio V/a*.
    gamma : float
    ie : int

    Returns
    -------
    b : ndarray of shape (4,)
        b[0] = r/r* (radius ratio)
        b[1] = dW/dr * r*
        b[2] = d²W/dr² * r*²
        b[3] = d³W/dr³ * r*³
    """
    g = contur_gas_constants(gamma)
    g1, g7, g9 = g['g1'], g['g7'], g['g9']
    qt = 1.0 / (1 + ie)

    ww = w * w
    al = g7 * g9
    aww = al - ww
    ww1 = ww - 1.0
    area = (((al - 1.0) / aww)**g1) / w

    b = np.zeros(4)
    b[0] = area**qt
    axw = al * ww1 * b[0]
    b[1] = w * aww / axw / qt
    c2 = 3.0 / qt + al * (2.0 - 1.0 / qt)
    c4 = al + 1.0 / qt
    cww = ww * (c2 - ww * c4) - al * (1.0 + 1.0 / qt)
    b[2] = b[1] * cww / axw / ww1
    dww = (2.0 * c2 - 4.0 * c4 * ww) / cww - 4.0 / ww1
    b[3] = b[2] * (b[2] / b[1] + w * b[1] * dww - 1.0 / b[0])
    return b


# ==========================================================================
# scond.f — Parabolic derivative (finite differences, unequal spacing)
# ==========================================================================

def parabolic_derivative(a, b_in):
    """Compute parabolic derivatives of curve with unequally spaced points.

    Port of scond.f (25 lines).

    Parameters
    ----------
    a : ndarray
        Independent variable (x-coordinates).
    b_in : ndarray
        Dependent variable (y-values).

    Returns
    -------
    c : ndarray
        Derivatives db/da at each point.
    """
    n = len(a)
    c = np.zeros(n)

    # Interior points: weighted finite difference
    for k in range(1, n - 1):
        s = a[k] - a[k - 1]
        t = a[k + 1] - a[k]
        c[k] = ((b_in[k + 1] - b_in[k]) * s**2
                + (b_in[k] - b_in[k - 1]) * t**2) / (s**2 * t + s * t**2)

    # Left endpoint: one-sided 3-point formula
    s0 = a[1] - a[0]
    t0 = a[2] - a[1]
    q0 = s0 + t0
    c[0] = (-t0 * (q0 + s0) * b_in[0] + q0**2 * b_in[1]
            - s0**2 * b_in[2]) / q0 / s0 / t0

    # Right endpoint: one-sided 3-point formula
    sf = a[-2] - a[-3]
    tf = a[-1] - a[-2]
    qf = sf + tf
    qst = qf * sf * tf
    c[-1] = (sf * (qf + tf) * b_in[-1] - qf**2 * b_in[-2]
             + tf**2 * b_in[-3]) / qst

    return c


# ==========================================================================
# twixt.f — Lagrange interpolation coefficients
# ==========================================================================

def lagrange_interp_coeffs(s, xbl):
    """Compute 4-point Lagrange interpolation coefficients.

    Port of twixt.f (28 lines). Finds the 4 points in array s
    bracketing xbl and returns interpolation weights.

    Parameters
    ----------
    s : ndarray
        Monotonically increasing abscissa values.
    xbl : float
        Point at which to interpolate.

    Returns
    -------
    gma, gmb, gmc, gmd : float
        Interpolation coefficients for points [j-2, j-1, j, j+1]
        where j is the bracketing index.
    kbl : int
        Index j+1 (base-0 adjusted from Fortran's base-1).
    """
    kat = len(s)

    # Find bracketing index (port of the Fortran loop)
    j = kat - 1
    for l in range(1, kat):
        if s[kat - 1 - l] < xbl:
            j = kat - l
            break

    xbb = s[j] - xbl
    kbl = j + 1

    du = s[j + 1] - s[j] if j + 1 < len(s) else s[j] - s[j - 1]
    dt = s[j] - s[j - 1]
    ds = s[j - 1] - s[j - 2]
    dst = ds + dt
    dstu = dst + du
    dtu = dt + du

    gma = -xbb * (dt - xbb) * (du + xbb) / ds / dst / dstu
    gmb = xbb * (dst - xbb) * (du + xbb) / ds / dt / dtu
    gmc = (dst - xbb) * (dt - xbb) * (du + xbb) / dst / dt / du
    gmd = -xbb * (dst - xbb) * (dt - xbb) / dstu / dtu / du

    return gma, gmb, gmc, gmd, kbl


# ==========================================================================
# ofeld.f — Interior point computation (characteristic network)
# ==========================================================================

def ofeld(a, b, gamma=1.4, ie=1, max_iter=40):
    """Compute interior point from intersection of C- and C+ characteristics.

    Port of ofeld.f (77 lines).

    CONTUR convention:
        a = upstream point on C- (right-running, carries K+ = θ + ν)
        b = upstream point on C+ (left-running, carries K- = θ - ν)

    Each point is [x, y, M, ψ, θ] where ψ = Prandtl-Meyer angle ν.

    Parameters
    ----------
    a : array-like, shape (5,)
        Point on C- characteristic: [x, y, M, psi, theta].
    b : array-like, shape (5,)
        Point on C+ characteristic: [x, y, M, psi, theta].
    gamma : float
    ie : int
        1 for axisymmetric, 0 for planar.
    max_iter : int

    Returns
    -------
    c : ndarray, shape (5,)
        Computed interior point [x, y, M, psi, theta].
    converged : bool
        True if iteration converged.
    """
    g = contur_gas_constants(gamma)
    g2 = g['g2']

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a1 = np.arcsin(1.0 / a[2])  # Mach angle at A
    a2 = np.arcsin(1.0 / b[2])  # Mach angle at B

    t1 = a[4]  # theta at A
    t2 = b[4]  # theta at B

    fsy1 = 0.0
    fsy2 = 0.0

    if ie != 0:
        if a[1] != 0.0:
            fsy1 = np.sin(a[4]) / a[1] / a[2]
        else:
            t1 = 0.0
            fsy1 = a[4]  # L'Hopital: sin(θ)/y → θ/y, and for axis: use θ as derivative

        if b[1] != 0.0:
            fsy2 = np.sin(b[4]) / b[1] / b[2]
        else:
            t2 = 0.0
            fsy2 = b[4]

    tni = np.tan(t1 - a1)  # C- slope at A
    if b[2] != 1.0:
        tn2 = np.tan(t2 + a2)  # C+ slope at B
    else:
        tn2 = np.tan(t2 + a2)

    # Initial guess (2D compatibility, no source terms)
    hdpsi = 0.5 * (a[3] - b[3])
    ht3 = 0.5 * (t1 + t2) + hdpsi
    t3 = ht3 - 0.5 * ie * hdpsi
    hpsi3 = 0.5 * (a[3] + b[3] + t1 - t2)
    psi3 = hpsi3 + 0.5 * ie * (t1 - t2)

    c3 = mach_from_prandtl_meyer(psi3, gamma)
    told = t3
    converged = True

    for i in range(max_iter):
        fm3 = c3
        a3 = np.arcsin(1.0 / c3)

        tna = 0.5 * (tni + np.tan(t3 - a3))
        if b[2] != 1.0:
            tnb = 0.5 * (np.tan(t3 + a3) + tn2)
        else:
            tnb = 2.0 * np.tan(t3 + a3)

        dtn = tnb - tna
        if abs(dtn) < 1e-15:
            break

        x3 = (b[0] * tnb - a[0] * tna + a[1] - b[1]) / dtn
        y3 = (a[1] * tnb - b[1] * tna + (b[0] - a[0]) * tna * tnb) / dtn

        if ie == 0 or abs(y3) < 1e-9:
            # Planar or on axis — no source term correction
            break

        fsy3 = np.sin(t3) / y3 / fm3
        p1 = 0.5 * (fsy1 + fsy3) * (x3 - a[0]) * np.sqrt(1.0 + tna**2)
        p2 = 0.5 * (fsy2 + fsy3) * (x3 - b[0]) * np.sqrt(1.0 + tnb**2)

        t3 = ht3 + 0.5 * (p1 - p2)
        psi3 = hpsi3 + 0.5 * (p1 + p2)
        c3 = mach_from_prandtl_meyer(psi3, gamma)

        if abs(t3 - told) <= 1e-9:
            if abs(c3 - fm3) < 1e-9:
                break

        if i == max_iter - 1:
            converged = False
            break

        # Damped update
        temp = t3
        t3 = (t3 + told) * 0.5
        told = temp

    c = np.array([x3, y3, c3, psi3, t3])
    return c, converged


# ==========================================================================
# trans.f — Transonic throat characteristic
# ==========================================================================

def sivells_throat_characteristic(rto, gamma=1.4, ie=1, n_points=21):
    """Compute the throat characteristic using Sivells' series expansion.

    Port of trans.f (222 lines). Computes points on the characteristic
    line emanating from the throat region, from wall (y=1) down to axis (y=0).

    Parameters
    ----------
    rto : float
        RC + 1, where RC is the throat radius of curvature / throat radius.
        For the Mach 4 case: rto = 7.0 (RC = 6.0).
    gamma : float
    ie : int
        1 for axisymmetric, 0 for planar.
    n_points : int
        Number of points on the throat characteristic (nn in Fortran).
        Must be odd. Default 21.

    Returns
    -------
    fc : ndarray, shape (6, n_points)
        fc[0,:] = x/y*  (normalized by throat height y*)
        fc[1,:] = y/y*
        fc[2,:] = Mach number
        fc[3,:] = Prandtl-Meyer angle ψ (radians)
        fc[4,:] = flow angle θ (radians)
        fc[5,:] = normalized mass flow fraction (1 at axis, 0 at wall)
    wo : float
        Axial velocity ratio at throat (w0 = V0/a*).
    amn : float
        Mach number at the start of the characteristic (axis end).
    awp, awpp, cwoppp : float
        Velocity derivatives for the axial distribution.
    axn : float
        x-position of the characteristic at the axis (x/y*).

    Notes
    -----
    The characteristic runs from y=1 (wall) to y=0 (axis) in the
    normalized coordinates where y0=1 is the reference height.
    Points are stored with index [n_points-1] at y=0 (axis)
    and index [0] near y=0.

    Reference: Sivells AEDC-TR-78-63, Appendix A.
    """
    nn = n_points
    g = contur_gas_constants(gamma)
    g2, g5, g6, g7, g8 = g['g2'], g['g5'], g['g6'], g['g7'], g['g8']
    ga = g['ga']
    qt = 1.0 / (1 + ie)

    # Integration step sizing
    jj = 240 // (nn - 1)
    if jj % 2 != 0:
        jj += 1
    if jj < 10:
        jj = 10
    kk = jj * nn - jj

    # Transonic coefficients (from trans.f lines 42-76)
    gb = ie / 8.0
    gk = (gamma * (gamma + 2.25 * ie - 16.5) + 2.25 * (2 + ie)) / 12.0
    gu = 1.0 - gamma / 1.5
    gv = (0.5 * (5 - 3 * ie) * gamma + ie) / (9 - ie)
    gz = np.sqrt(qt * (gamma + 1.0))

    u22 = gb + gamma / 3.0 / (3 - ie)
    u42 = (gamma + (4 - ie) * 1.5) / 6.0 / (3 - ie)

    if ie == 1:
        # Axisymmetric coefficients
        gt = (gamma * (gamma * 92.0 + 180.0) - 9.0) / 1152.0
        u23 = (gamma * (304.0 * gamma + 255.0) - 54.0) / 1728.0
        u43 = (gamma * (388.0 * gamma + 777.0) + 153.0) / 2304.0
        u63 = (gamma * (556.0 * gamma + 1737.0) + 3069.0) / 10368.0
        up0 = (gamma * (52.0 * gamma + 75.0) - 9.0) / 192.0
        up2 = (gamma * (52.0 * gamma + 51.0) + 327.0) / 384.0
        v02 = (28.0 * gamma - 15.0) / 288.0
        v22 = (20.0 * gamma + 27.0) / 96.0
        v42 = (gamma / 3.0 + 1.0) / 3.0
        v03 = (gamma * (7100.0 * gamma + 2151.0) + 2169.0) / 82944.0
        v23 = (gamma * (3424.0 * gamma + 4071.0) - 972.0) / 13824.0
        v43 = (gamma * (3380.0 * gamma + 7551.0) + 3771.0) / 13824.0
        v63 = (gamma * (6836.0 * gamma + 23031.0) + 30627.0) / 82944.0
    else:
        # Planar coefficients
        gt = (gamma * (gamma * 134.0 + 429.0) + 123.0) / 4320.0
        u23 = (gamma * (854.0 * gamma + 807.0) + 279.0) / 12960.0
        u43 = (gamma * (194.0 * gamma + 549.0) - 63.0) / 2592.0
        u63 = (gamma * (362.0 * gamma + 1449.0) + 3177.0) / 12960.0
        up0 = (gamma * (26.0 * gamma + 51.0) - 27.0) / 144.0
        up2 = (gamma * (26.0 * gamma + 27.0) + 237.0) / 288.0
        v02 = (34.0 * gamma - 75.0) / 1080.0
        v22 = (10.0 * gamma + 15.0) / 108.0
        v42 = (22.0 * gamma + 75.0) / 360.0
        v03 = (gamma * (7570.0 * gamma + 3087.0) + 23157.0) / 544320.0
        v23 = (gamma * (5026.0 * gamma + 7551.0) - 4923.0) / 77760.0
        v43 = (gamma * (2254.0 * gamma + 6153.0) + 2979.0) / 25920.0
        v63 = (gamma * (6574.0 * gamma + 26481.0) + 40059.0) / 181440.0

    # Throat velocity
    wo = 0.0  # Will be set by caller or computed from throat conditions
    # Actually, wo is the axial velocity at the throat. From the series expansion:
    # wo = V0/a* at x=0, y=0 (throat center)
    # It's computed from the throat curvature and gamma
    # wwo = wo + correction terms...

    wwo_corr = (0.5 + (u42 - u22 + (u63 - u43 + u23) / rto) / rto) / rto

    # wo is needed. In CONTUR, wo is passed in from axial.f.
    # For the throat, wo comes from the transonic expansion.
    # We compute it iteratively: wwo = wo + correction, then
    # amn = wwo/sqrt(g7 - g8*wwo²)

    # From CONTUR Mach 4 output: wo = 0.97501029
    # But we need to compute it from the series.
    # In trans.f, wo is an input parameter computed by axial.f.
    # For now, compute from the throat conditions.

    # The throat velocity ratio w0 at x=0, y=0 is determined by the
    # transonic series. For a given rto, it satisfies:
    # wwo = wo + half/(rto) + (u42-u22)/rto² + (u63-u43+u23)/rto³
    # amn = wwo/sqrt(g7 - g8*wwo²) must be > 1

    # From the CONTUR code, wo is actually computed in axial.f as the
    # subsonic throat velocity. For the throat characteristic, it's the
    # velocity at y=0, x=0 (throat center), which is subsonic.

    # For standalone use, we'll compute wo from the CONTUR formula.
    # The Mach 4 output gives wo = 0.97501029 for rto=7.

    # Use a simpler approach: wo is determined by the expansion
    # w0 = 1 - 1/(2*S) where S = rto, approximately
    # More precisely: from the CONTUR output format
    wo = 1.0 - 1.0 / (2.0 * rto)  # First approximation

    # Actually, from the CONTUR series (Sivells Appendix A):
    # At y=0, the velocity w0 satisfies a series in 1/rto.
    # We can invert: given rto, compute wo.
    # But this requires the full axial.f logic.
    # For testing, accept wo as a parameter.

    # For now, use the known value for validation
    # wo = 0.97501029  # Mach 4 case

    # Actually, I should just make wo a parameter. Let me restructure.
    raise NotImplementedError(
        "sivells_throat_characteristic requires wo from axial distribution. "
        "Port axial.f first, or pass wo explicitly."
    )


def sivells_throat_characteristic_with_wo(
    rto, wo, gamma=1.4, ie=1, n_points=21, tk=1.0
):
    """Compute throat characteristic given the throat velocity wo.

    Port of trans.f. See sivells_throat_characteristic for full docs.

    Parameters
    ----------
    rto : float
        RC + 1 (throat radius of curvature parameter).
    wo : float
        Throat center velocity ratio V/a* at (x=0, y=0).
    tk : float
        Mass ratio y*/y0 (passed from axial.f). Default 1.0 for
        backward compatibility.
    gamma, ie, n_points : see sivells_throat_characteristic.

    Returns
    -------
    fc : ndarray, shape (6, n_points)
        Throat characteristic point data.
    result : dict
        Contains amn, amp, ampp, w, awp, awpp, cwoppp, axn.
    """
    nn = n_points
    g = contur_gas_constants(gamma)
    g2, g5, g6, g7, g8 = g['g2'], g['g5'], g['g6'], g['g7'], g['g8']
    ga = g['ga']
    qt = 1.0 / (1 + ie)

    # Integration parameters
    jj = 240 // (nn - 1)
    if jj % 2 != 0:
        jj += 1
    if jj < 10:
        jj = 10
    kk = jj * nn - jj

    # Transonic coefficients
    gb = ie / 8.0
    gk_coef = (gamma * (gamma + 2.25 * ie - 16.5) + 2.25 * (2 + ie)) / 12.0
    gu = 1.0 - gamma / 1.5
    gv = (0.5 * (5 - 3 * ie) * gamma + ie) / (9 - ie)
    gz = np.sqrt(qt * (gamma + 1.0))

    u22 = gb + gamma / 3.0 / (3 - ie)
    u42 = (gamma + (4 - ie) * 1.5) / 6.0 / (3 - ie)

    if ie == 1:
        gt = (gamma * (gamma * 92.0 + 180.0) - 9.0) / 1152.0
        u23 = (gamma * (304.0 * gamma + 255.0) - 54.0) / 1728.0
        u43 = (gamma * (388.0 * gamma + 777.0) + 153.0) / 2304.0
        u63 = (gamma * (556.0 * gamma + 1737.0) + 3069.0) / 10368.0
        up0 = (gamma * (52.0 * gamma + 75.0) - 9.0) / 192.0
        up2 = (gamma * (52.0 * gamma + 51.0) + 327.0) / 384.0
        v02 = (28.0 * gamma - 15.0) / 288.0
        v22 = (20.0 * gamma + 27.0) / 96.0
        v42 = (gamma / 3.0 + 1.0) / 3.0
        v03 = (gamma * (7100.0 * gamma + 2151.0) + 2169.0) / 82944.0
        v23 = (gamma * (3424.0 * gamma + 4071.0) - 972.0) / 13824.0
        v43 = (gamma * (3380.0 * gamma + 7551.0) + 3771.0) / 13824.0
        v63 = (gamma * (6836.0 * gamma + 23031.0) + 30627.0) / 82944.0
    else:
        gt = (gamma * (gamma * 134.0 + 429.0) + 123.0) / 4320.0
        u23 = (gamma * (854.0 * gamma + 807.0) + 279.0) / 12960.0
        u43 = (gamma * (194.0 * gamma + 549.0) - 63.0) / 2592.0
        u63 = (gamma * (362.0 * gamma + 1449.0) + 3177.0) / 12960.0
        up0 = (gamma * (26.0 * gamma + 51.0) - 27.0) / 144.0
        up2 = (gamma * (26.0 * gamma + 27.0) + 237.0) / 288.0
        v02 = (34.0 * gamma - 75.0) / 1080.0
        v22 = (10.0 * gamma + 15.0) / 108.0
        v42 = (22.0 * gamma + 75.0) / 360.0
        v03 = (gamma * (7570.0 * gamma + 3087.0) + 23157.0) / 544320.0
        v23 = (gamma * (5026.0 * gamma + 7551.0) - 4923.0) / 77760.0
        v43 = (gamma * (2254.0 * gamma + 6153.0) + 2979.0) / 25920.0
        v63 = (gamma * (6574.0 * gamma + 26481.0) + 40059.0) / 181440.0

    hvppp = (3 * ie - (10 - 3 * ie) * gamma) / 4.0 / rto / np.sqrt(rto)

    # Corrected throat velocity
    wwo = wo + (0.5 + (u42 - u22 + (u63 - u43 + u23) / rto) / rto) / rto
    wop = (1.0 - (gb - gt / rto) / rto) / np.sqrt(rto)
    wopp = (gu - gv / rto) / rto
    hoppp = gk_coef / rto / np.sqrt(rto)

    amn = wwo / np.sqrt(g7 - g8 * wwo**2)
    bet = np.sqrt(amn**2 - 1.0)
    psi1 = g2 * np.arctan(bet / g2) - np.arctan(bet)

    # Initialize fc array: [x, y, M, psi, theta, mass_flow]
    fc = np.zeros((6, nn))

    # Start at wall (y=1) marching down to axis (y=0)
    p1 = 0.0
    t1 = 0.0
    x1 = 0.0
    y1 = 1.0
    fsy1 = 0.0
    tn2 = -1.0 / bet

    # Point nn (last index) = axis in Fortran convention
    # In our 0-indexed array, fc[:, nn-1] corresponds to Fortran fc(:,nn)
    fc[0, nn - 1] = x1
    fc[1, nn - 1] = y1
    fc[2, nn - 1] = amn
    fc[3, nn - 1] = psi1
    fc[4, nn - 1] = 0.0
    fc[5, nn - 1] = 0.0

    bx = 1.0
    total_sum = 0.0
    fsa = (ie + 1) * amn / (g6 + g5 * amn**2)**ga

    for j_step in range(1, kk + 1):
        y = float(kk - j_step) / kk
        if ie == 1:
            bx = y + y
        yy = y * y
        tn1 = tn2

        # Velocity coefficients (series expansion)
        vo = (((yy * (yy * (yy * v63 - v43) + v23) - v03) / rto
               + yy * (yy * v42 - v22) + v02) / rto
              + 0.5 * (yy - 1.0) / (3 - ie)) / rto

        vp = (1.0 + ((yy * (2.0 * gamma + 3 * (4 - ie)) - 2.0 * gamma
               - 1.5 * ie) / (3 - ie) / 3.0
               + (yy * (6.0 * u63 * yy - 4.0 * u43) + 2.0 * u23) / rto)
              / rto) / np.sqrt(rto)

        vpp = 2.0 * (1.0 + (2.0 * up2 * yy - up0) / rto) / rto

        # Iterate for x and Mach from characteristic equations
        for _it in range(10):
            tna = 0.5 * (tn1 + tn2)
            x = x1 + (y - y1) / tna
            dxi = np.sqrt((y - y1)**2 + (x - x1)**2)
            xot = x / gz

            vy = gz * (vo + xot * (vp + xot * (0.5 * vpp + xot * hvppp / 3.0))) / np.sqrt(rto)
            w = amn / np.sqrt(g6 + g5 * amn**2)
            t = np.arcsin(np.clip(vy * y / w, -1.0, 1.0))
            fsy = ie * vy / w / amn if amn > 0 else 0.0
            p1 = 0.5 * (fsy1 + fsy) * dxi
            psi = p1 + psi1 + t1 - t
            fma = mach_from_prandtl_meyer(max(psi, 0.0), gamma)

            if abs(amn - fma) < 1e-10:
                break
            fmu = np.arcsin(1.0 / fma)
            tn2 = np.tan(t - fmu)
            amn = fma

        # Simpson's rule mass flow integration
        if j_step % 2 != 0:
            # Odd step: save for Simpson's
            a_s = y1 - y
            fsb = bx / np.sin(fmu - t) / (g6 + g5 * fma**2)**ga
        else:
            # Even step: complete Simpson's interval
            bs = y1 - y
            cs = a_s + bs
            s1_coef = (2.0 - bs / a_s) * cs / 6.0
            s3_coef = (2.0 - a_s / bs) * cs / 6.0
            s2_coef = cs - s1_coef - s3_coef
            fsc = bx / np.sin(fmu - t) / (g6 + g5 * fma**2)**ga
            add = s1_coef * fsa + s2_coef * fsb + s3_coef * fsc
            total_sum = add + total_sum
            fsa = fsc

        x1 = x
        y1 = y
        t1 = t
        fsy1 = fsy
        psi1 = psi

        # Store at output point intervals
        if j_step % jj == 0:
            k = nn - 1 - j_step // jj
            fc[0, k] = x
            fc[1, k] = y
            fc[2, k] = fma
            fc[3, k] = psi
            fc[4, k] = t
            fc[5, k] = total_sum

    # Normalize coordinates and mass flow (matches trans.f lines 156-160)
    for j in range(nn):
        fc[0, j] /= tk
        fc[1, j] /= tk
    if total_sum > 0:
        for j in range(nn):
            fc[5, j] = 1.0 - fc[5, j] / total_sum

    axn = fc[0, 0]  # x at axis (normalized by tk)

    # Compute velocity derivatives (trans.f lines 162-164)
    w_axis = amn / np.sqrt(g6 + g5 * amn**2)
    awop = wop * tk / gz
    awopp = wopp * (tk / gz)**2
    awoppp = 2.0 * hoppp * (tk / gz)**3

    # Corrected third derivative (from cubic fit)
    if axn != 0:
        cwoppp_val = 6.0 * (w_axis - wo - axn * (awop + axn * awopp / 2.0)) / axn**3
        if cwoppp_val < awoppp:
            cwoppp_val = awoppp
    else:
        cwoppp_val = awoppp

    awp = awop + axn * (awopp + axn * cwoppp_val / 2.0)
    awpp = awopp + axn * cwoppp_val

    amp = awp * g7 * (amn / w_axis)**3
    ampp = amp * (awpp / awp + 3.0 * g5 * amp * w_axis**2 / amn)

    result = {
        'amn': amn,     # Mach at axis end of throat characteristic
        'amp': amp,      # dM/dx at axis
        'ampp': ampp,    # d²M/dx² at axis
        'w': w_axis,     # velocity ratio at axis
        'awp': awp,      # dW/dx at axis
        'awpp': awpp,    # d²W/dx² at axis
        'cwoppp': cwoppp_val,  # corrected d³W/dx³
        'axn': axn,      # x-position of axis end
        'wo': wo,        # throat center velocity
        'wwo': wwo,      # corrected throat velocity
        'wop': wop,      # dW/dx at throat (unnormalized)
    }

    return fc, result


# ==========================================================================
# axial.f — Centerline Mach/velocity distribution
# ==========================================================================

def sivells_axial(gamma, eta_deg, rc, bmach, cmach, ie=0,
                  n_char=41, n_axis=21, ix=0, lr_sign=-1, nx=0):
    """Compute the axial velocity/Mach distribution and throat characteristic.

    Port of axial.f (813 lines). This is the core setup routine that:
    1. Computes throat parameters (wo, tk, yo)
    2. Calls trans for the throat characteristic
    3. Iterates to find the inflection point Mach (emach)
    4. Computes the upstream polynomial distribution coefficients
    5. Fills the axis array with M, psi, dtheta/dy at each point

    Parameters
    ----------
    gamma : float
        Ratio of specific heats.
    eta_deg : float
        Inflection angle in degrees.
    rc : float
        Throat radius of curvature / throat radius (e.g. 6.0).
    bmach : float
        Mach number at the start of the radial flow region.
    cmach : float
        Design exit Mach number.
    ie : int
        0 for planar, 1 for axisymmetric.
    n_char : int
        Number of points on the first characteristic (M parameter in CONTUR).
    n_axis : int
        Number of axis points (N parameter in CONTUR).
    ix : int
        Distribution type: 0 for 3rd-degree, nonzero for 4th-degree.
    lr_sign : int
        Sign of lr. Negative means compute throat characteristic.

    Returns
    -------
    result : dict
        Contains all computed parameters and arrays:
        - 'fc': throat characteristic array (6, n_points)
        - 'axis': axis distribution array (5, n_axis)
          axis[0,:] = x, axis[1,:] = 0, axis[2,:] = M,
          axis[3,:] = psi, axis[4,:] = dtheta/dy
        - Scalar parameters: wo, tk, yo, wwo, wwop, emach, fmach,
          wi, wip, wipp, etc.
        - 'c': polynomial coefficients (6,)
        - 'xi', 'xo', 'xoi', 'xie', 'xe': key x-positions

    Reference
    ---------
    Sivells AEDC-TR-78-63, Section 3.0, Eqs. 29-39.
    """
    g = contur_gas_constants(gamma)
    gm, g1, g2, g4, g5, g6, g7, g8, g9, ga = (
        g['gm'], g['g1'], g['g2'], g['g4'], g['g5'],
        g['g6'], g['g7'], g['g8'], g['g9'], g['ga']
    )
    qt = 1.0 / (1 + ie)
    conv = 180.0 / np.pi

    # --- Transonic coefficients (fg module, lines 126-157) ---
    fg = contur_fg_constants(gamma, ie)
    gc, gd, ge, gf = fg['gc'], fg['gd'], fg['ge'], fg['gf']
    gh, gi = fg['gh'], fg['gi']
    gj_c, gk_c = fg['gj'], fg['gk']
    gr, gs = fg['gr'], fg['gs']
    hb, hc_c, he_c = fg['hb'], fg['hc'], fg['he']

    # --- Inflection angle and geometry (lines 174-189) ---
    eta = eta_deg / conv  # radians
    if ie == 0:
        se = eta
    else:
        se = 2.0 * np.sin(0.5 * eta)
    cse = np.cos(eta)
    rt = rc + 1.0

    # Exit Mach properties
    cbet = np.sqrt(cmach**2 - 1.0)
    frc = ((g6 + g5 * cmach**2)**ga / cmach)**qt
    tye = frc * se

    # --- Inflection point Mach from radial flow (lines 211-229) ---
    bbet = np.sqrt(bmach**2 - 1.0)
    bpsi = g2 * np.arctan(g4 * bbet) - np.arctan(bbet)

    # fmach=0 path (line 213 → 8 → 9 → 10)
    fmach_init = -bpsi / eta
    fpsi = -fmach_init * eta  # = bpsi
    fmach = mach_from_prandtl_meyer(fpsi, gamma)

    epsi = fpsi - 2.0 * eta / qt
    emach = mach_from_prandtl_meyer(epsi, gamma)
    we = g2 * emach / np.sqrt(emach**2 + g9)
    dw = we - 1.0  # wi = 1.0

    d_src = source_derivatives(we, gamma, ie)
    xe = d_src[0]
    wep = d_src[1]
    wepp = d_src[2]
    wrppp = d_src[3]

    # --- Throat parameters (lines 247-281) ---
    # tk = mass ratio y*/y0 (line 247)
    tk = (1.0 - g7 * (1.0 + (ge + gf / rt) / rt)
          / rt**2 / (15 + ie) / 3.0)**qt
    yo = se / tk
    aa = np.sqrt(qt * (gamma + 1.0) * rt)

    # Throat velocity derivatives (line 251)
    wipp = (1.0 - gamma / 1.5 + gj_c / rt) / (aa * yo)**2

    # Throat velocity gradient (line 270)
    wip = (1.0 - (gc - gd / rt) / rt) / yo / aa
    whp = wip

    # Mach derivatives at throat (lines 273-274)
    amp = g7 * wip
    ampp = g7 * (wipp + 3.0 * g8 * wip**2)

    # Inflection point offset (line 275)
    xoi = yo * np.sqrt(g7 / 2.0 / (9 - ie) / rt) * (1.0 + (gh + gi / rt) / rt)

    # Throat center velocity (line 279)
    wo = 1.0 - (0.5 / (3 - ie) + (gr + gs / rt) / rt) / rt

    # Throat Mach (subsonic, for reference)
    om = wo / np.sqrt(g7 - g8 * wo**2)

    # Third derivative of velocity (line 281)
    woppp = gk_c / (aa * yo)**3

    # --- Call throat characteristic (line 285) ---
    lr = lr_sign * abs(n_axis)
    nn_tc = abs(lr)

    fc, tc_result = sivells_throat_characteristic_with_wo(
        rt, wo, gamma=gamma, ie=ie, n_points=nn_tc, tk=tk
    )

    # Retrieve trans.f outputs
    am = tc_result['amn']     # Mach at axis end
    wi = tc_result['w']       # velocity at axis end
    awp = tc_result['awp']    # dW/dx
    awpp = tc_result['awpp']  # d²W/dx²
    awppp = tc_result['cwoppp']  # corrected d³W/dx³
    xi = tc_result['axn']     # x at axis end

    # --- Rescale derivatives by se (lines 288-295) ---
    amp_s = tc_result['amp'] / se
    ampp_s = tc_result['ampp'] / se**2
    wap = awp / se
    wapp = awpp / se**2
    woppp_s = awppp / se**3

    dw = we - wi  # update dw with actual wi from trans
    xoi_s = xi * se  # rescaled inflection offset

    # --- Emach iteration (lines 384-456) ---
    # This iterates to find the inflection point velocity we
    # that gives a consistent solution
    wip_iter = wap  # use throat characteristic derivatives
    wipp_iter = wapp
    nocon = 0
    max_iter = 100

    for _it in range(max_iter):
        nocon += 1
        if nocon > max_iter:
            raise RuntimeError(f"No convergence in {nocon} iterations")

        # 3rd-degree distribution (ix=0, line 408)
        if ix == 0:
            denom_sq = (wip_iter + wep + wep)**2 - 6.0 * dw * wepp
            if denom_sq < 0:
                denom_sq = abs(denom_sq)
            xie = 6.0 * dw / (np.sqrt(denom_sq) + wip_iter + wep + wep)
            fxw = 0.5 * xie * (wepp + wipp_iter) / (wep - wip_iter) if abs(wep - wip_iter) > 1e-15 else 0.0

            if fxw <= 0:
                ew = we + 0.1
            elif fxw < 1.0 and ie != 0:
                ew = wi + dw * (4.0 + fxw**2) / 5.0
            else:
                ew = wi + dw * (9.0 + fxw) / 10.0
        else:
            # 4th-degree distribution (ix≠0, line 415)
            ea = woppp_s
            eb = 5.0 * wipp_iter + wepp
            ec = 12.0 * wip_iter
            ed = -12.0 * dw
            xie = cubic_solve(ea, eb, ec, ed)
            if xie <= 0:
                ew = we - 0.1
                if ew <= wi:
                    raise RuntimeError("RC is too large to allow a solution")
            else:
                ew = wi + 0.5 * xie * (wip_iter + wep + xie * (wipp_iter - wepp) / 6.0)

        we = ew
        if we > g2:
            raise RuntimeError("Velocity greater than theoretical maximum")

        if abs(ew - dw - wi) < 1e-9:
            break

        dw = we - wi
        d_src = source_derivatives(we, gamma, ie)
        xe = d_src[0]
        wep = d_src[1]
        wepp = d_src[2]
        wrppp = d_src[3]

    # --- Converged emach (lines 450-456) ---
    emach = we / np.sqrt(g7 - g8 * we**2)
    ebet = np.sqrt(emach**2 - 1.0)
    epsi = g2 * np.arctan(g4 * ebet) - np.arctan(ebet)
    fpsi = epsi + 2.0 * eta / qt
    fmach = mach_from_prandtl_meyer(fpsi, gamma)

    # --- Upstream distribution coefficients (lines 444-487) ---
    # Radial flow: 3rd or 4th/5th degree polynomial
    h_val = 3.0 * (wep + wip_iter) / (wipp_iter - wepp)
    hh = 12.0 * dw / (wipp_iter - wepp)
    xie = hh / (np.sqrt(h_val**2 + hh) + h_val)

    # Polynomial coefficients c(1..6)
    c = np.zeros(6)
    c[0] = wi  # c(1) in Fortran
    c[1] = xie * wip_iter  # c(2)
    c[2] = 0.5 * wipp_iter * xie**2  # c(3)
    c[3] = (10.0 * dw - xie * (4.0 * wep - 0.5 * xie * wepp)
            - 6.0 * c[1] - 3.0 * c[2])  # c(4)
    c[4] = (xie * (7.0 * wep + 8.0 * wip_iter
            - xie * (wepp - 1.5 * wipp_iter)) - 15.0 * dw)  # c(5)
    c[5] = (6.0 * dw - 3.0 * xie * (wep + wip_iter)
            + 0.5 * xie**2 * (wepp - wipp_iter))  # c(6)

    # For 3rd-degree: c(5)=0 and c(6)=0 (line 483-485)
    if ix == 0:
        c[4] = 0.0
    c[5] = 0.0  # sfoa=0 always for this path

    # Velocity derivatives at inflection (for output)
    eoe = epsi / eta
    wippp = 6.0 * c[3] / xie**3
    weppp = 6.0 * (c[3] + 4.0 * c[4] + 10.0 * c[5]) / xie**3

    xi_final = xe - xie  # xi (inflection x-position, line 471)
    xo = xi_final - xoi_s  # xo (throat x-position)
    x1 = xo + xoi  # x1 (Mach 1 position)

    # --- wwo and wwop (param module, line 308-309) ---
    wwo = 1.0 + (1.0 / (ie + 3) - (hb - hc_c / rt) / rt) / rt
    wwop = (1.0 + (1.0 - ie / 8.0 - he_c / rt) / rt) / yo / aa

    # --- Axis Mach distribution (amach, gmach, etc.) ---
    gpsi = fpsi - eta / qt
    gmach = mach_from_prandtl_meyer(gpsi, gamma)
    rg = ((g6 + g5 * gmach**2)**ga / gmach)**qt
    apsi = bpsi - eta / qt
    amach = mach_from_prandtl_meyer(apsi, gamma)
    ra = ((g6 + g5 * amach**2)**ga / amach)**qt
    xa = ra * cse

    # --- Fill axis array (lines 657-728) ---
    # Fortran k=1..n, axis(1..5,k) with 1-based indexing
    # Our k=0..n_axis-1 with 0-based indexing
    axis = np.zeros((5, n_axis))
    fn = float(n_axis - 1)

    for k in range(n_axis):
        # Point spacing: q goes from 1.0 (exit, k=0) to 0.0 (inflection, k=n-1)
        # Fortran k_f = k+1, so (n - k_f)/fn = (n_axis - 1 - k)/fn
        frac = (n_axis - 1 - k) / fn
        if nx == 0:
            q = frac**2  # Quadratic spacing (line 660)
        else:
            q = frac**(nx * 0.1)  # Power-law spacing (line 661)
        axis[0, k] = xie * q + xi_final  # x position (line 665)

        # Velocity from polynomial (lines 700-703, ip=0 path)
        w_val = c[0] + q * (c[1] + q * (c[2] + q * (c[3] + q * (c[4] + q * c[5]))))
        wp = (c[1] + q * (2 * c[2] + q * (3 * c[3] + q * (4 * c[4] + q * 5 * c[5])))) / xie
        wpp = 2 * (c[2] + q * (3 * c[3] + q * (6 * c[4] + q * 10 * c[5]))) / xie**2
        wppp = 6 * (c[3] + q * (4 * c[4] + 10 * q * c[5])) / xie**3

        # Mach from velocity (lines 704-717)
        gww = g7 - w_val**2 * g8
        if gww <= 0:
            raise RuntimeError("Velocity greater than theoretical maximum")
        gw = np.sqrt(gww)
        xm = w_val / gw

        # Mach derivatives
        xmw = g7 / gw / gww
        xmp = xmw * wp
        xmpp = xmw * (wpp + 3.0 * g8 * w_val * wp**2 / gww)

        # Store (lines 723-727)
        axis[1, k] = 0.0  # y = 0 (axis)
        axis[2, k] = xm   # Mach
        xbet = np.sqrt(max(xm**2 - 1.0, 0.0))
        axis[3, k] = g2 * np.arctan(g4 * xbet) - np.arctan(xbet)  # psi
        axis[4, k] = ie * 0.5 * (xm - 1.0 / xm) * wp / w_val  # dtheta/dy

    return {
        # Throat characteristic
        'fc': fc,
        'tc_result': tc_result,
        # Axis distribution
        'axis': axis,
        'c': c,
        # Geometry
        'rt': rt, 'se': se, 'tk': tk, 'yo': yo,
        'xi': xi_final, 'xo': xo, 'xoi': xoi_s, 'xie': xie, 'xe': xe,
        'xa': xa, 'sf_default': 1.0 / yo,
        # Throat
        'wo': wo, 'om': om, 'wwo': wwo, 'wwop': wwop,
        # Inflection point
        'emach': emach, 'fmach': fmach, 'amach': amach, 'gmach': gmach,
        'epsi': epsi, 'fpsi': fpsi, 'eoe': eoe,
        # Velocities at throat char axis
        'wi': wi, 'wip': wap, 'wipp': wapp,
        'wippp': wippp, 'woppp': woppp_s,
        # Velocities at inflection
        'we': we, 'wep': wep, 'wepp': wepp, 'weppp': weppp, 'wrppp': wrppp,
        # Derivatives
        'amp': amp_s, 'ampp': ampp_s,
        # Iteration count
        'nocon': nocon,
        # MOC parameters
        'n_char': n_char, 'n_axis': n_axis,
        'frc': frc, 'tye': tye, 'cbet': cbet, 'cse': cse,
        'bpsi': bpsi, 'rg': rg, 'ra': ra,
    }


# ==========================================================================
# perfc.f — MOC contour generation (upstream)
# ==========================================================================

def sivells_perfc(axial_result, gamma=1.4, ie=0):
    """Compute the upstream nozzle contour using MOC with mass flow integration.

    Port of perfc.f (634 lines). For the upstream contour (ip=0), this:
    1. Scales the throat characteristic and copies to fclast
    2. Sets up the initial radial flow characteristic at inflection
    3. Marches characteristics from inflection toward throat using ofeld
    4. Integrates mass flow along each characteristic (Simpson's rule)
    5. Finds wall position where integrated mass equals total mass
    6. Matches remaining wall points against the throat characteristic
    7. Integrates wall slopes (dy/dx = tan(theta)) for y-coordinates

    All loop variables (nn, last, line, etc.) use Fortran 1-indexed convention.
    Arrays are padded with an extra element at index 0 so that Fortran indices
    can be used directly: a[:, j] in Python = a(:, j) in Fortran.

    Parameters
    ----------
    axial_result : dict
        Output from sivells_axial().
    gamma : float
    ie : int
        0 for planar, 1 for axisymmetric.

    Returns
    -------
    result : dict
    """
    g = contur_gas_constants(gamma)
    g2, g4, g5, g6, g8, ga = g['g2'], g['g4'], g['g5'], g['g6'], g['g8'], g['ga']
    g7 = g['g7']
    qt = 1.0 / (1 + ie)
    conv = 180.0 / np.pi
    zro = 0.0
    one = 1.0
    two = 2.0
    half = 0.5
    six = 6.0
    thr = 3.0

    # --- Unpack axial_result ---
    fc_in = axial_result['fc']        # (6, lq) throat characteristic, 0-indexed
    axis_0 = axial_result['axis']     # (5, n_axis) 0-indexed

    n = axial_result['n_axis']        # Fortran n
    m = axial_result['n_char']        # Fortran m

    eta_rad = np.arccos(axial_result['cse'])
    se = eta_rad if ie == 0 else 2.0 * np.sin(0.5 * eta_rad)

    xo = axial_result['xo']
    yo = axial_result['yo']
    epsi = axial_result['epsi']
    bpsi = axial_result['bpsi']
    cbet = axial_result['cbet']
    tye = axial_result['tye']
    cse = axial_result['cse']
    wwo = axial_result['wwo']
    wwop = axial_result['wwop']
    rc = axial_result['rt'] - 1.0  # rto = rc + 1, so rc = rto - 1
    sdo = (1.0 / rc) / yo  # rrc/yo where rrc = 1/rc (axial.f line 310-311)

    # Convert axis to 1-indexed: taxi(:, 1..n) = axis_0(:, 0..n-1)
    taxi = np.zeros((5, n + 1))  # 1-indexed
    for k in range(n):
        taxi[:, k + 1] = axis_0[:, k]

    # --- Phase 1: Scale throat characteristic, copy to fclast ---
    # perfc.f lines 102-128
    lq = fc_in.shape[1]
    nl = n + lq - 1

    # 1-indexed arrays for fc, fclast, su
    fc = np.zeros((6, lq + 1))
    fclast = np.zeros((5, lq + 1))
    su = np.zeros(200)

    for j in range(lq):
        fc[0, j + 1] = fc_in[0, j] * se + xo  # qm=1 path
        fc[1, j + 1] = fc_in[1, j] * se
        fc[2, j + 1] = fc_in[2, j]
        fc[3, j + 1] = fc_in[3, j]
        fc[4, j + 1] = fc_in[4, j]
        fc[5, j + 1] = fc_in[5, j]
        for k in range(5):
            fclast[k, j + 1] = fc[k, j + 1]
        su[j + 1] = fc[5, j + 1]  # sumax=1 for qm=1

    # --- Phase 2: Initial radial flow characteristic ---
    # perfc.f lines 156-168 (ise=0, ip=0 path)
    em = eta_rad / (m - 1)
    # 1-indexed: a(:, 1..m)
    MAXPTS = max(m, lq) + m + 5
    a = np.zeros((5, MAXPTS + 1))
    b = np.zeros((5, MAXPTS + 1))

    for k_f in range(1, m + 1):  # Fortran k=1,m
        t = (k_f - 1) * em
        psi_k = epsi + t / qt
        xm = mach_from_prandtl_meyer(psi_k, gamma)
        r = ((g6 + g5 * xm**2)**ga / xm)**qt
        xbet = np.sqrt(xm**2 - one)
        a[0, k_f] = r * np.cos(t)
        a[1, k_f] = r * np.sin(t)
        a[2, k_f] = xm
        a[3, k_f] = g2 * np.arctan(g4 * xbet) - np.arctan(xbet)
        a[4, k_f] = t

    if ie == 1:
        a[4, 1] = taxi[4, 1]

    # --- Phase 3: Initialize wall ---
    # perfc.f lines 171-191
    wall = np.zeros((5, nl + 3))  # 1-indexed: wall(:, 1..nl+1)
    for j in range(5):
        wall[j, 1] = a[j, m]  # wall(:,1) = a(:,m)

    line = 1
    su[1] = zro
    nn = 1

    # Copy a → b
    for k_f in range(1, m + 1):
        for j in range(5):
            b[j, k_f] = a[j, k_f]
    last = m - 1  # perfc.f line 190: last=m-1

    # --- Mass integration and wall finding ---
    # s, fs are 1-indexed working arrays
    s = np.zeros(MAXPTS + 1)
    fs = np.zeros(MAXPTS + 1)

    def _mass_integrate_and_find_wall():
        """Mass integration along current b characteristic.

        Uses outer variables: b, nn, last, line, s, fs, su, wall, ie, se,
        g5, g6, ga, conv.

        Implements perfc.f labels 20-34.
        Returns (total_mass_or_None, updated_last).
        """
        nonlocal last
        lastp = last + 1

        # Compute s and fs arrays (label 21)
        for j_f in range(nn, lastp + 1):  # Fortran: do j=nn,lastp
            bx_loc = one / se if ie == 0 else two * b[1, j_f] / se**2
            xm = b[2, j_f]
            if xm < 1.0:
                xm = max(xm, 1.001)
            xmur = np.arcsin(one / xm)
            # ip=0: integration with respect to y
            s[j_f] = b[1, j_f] - b[1, nn]
            dsx = one / np.sin(xmur + b[4, j_f])
            if b[1, j_f] == zro:
                dsx = xm
            fs[j_f] = dsx * bx_loc / (g6 + g5 * xm**2)**ga

        # Simpson integration (label 258-287 / 290-303)
        sa, sb, sc = zro, zro, zro
        sum1 = su[nn]
        kan = (lastp - nn) // 2
        kt = nn
        k = nn  # will be overwritten

        for j_loc in range(1, kan + 1):  # Fortran: do j=1,kan
            k = nn + 2 * j_loc
            kt = k
            a_s = s[k - 1] - s[k - 2]
            b_s = s[k] - s[k - 1]
            c_s = a_s + b_s

            if abs(a_s) < 1e-30 or abs(b_s) < 1e-30:
                continue

            s1 = (two - b_s / a_s) * c_s / six
            s3 = (two - a_s / b_s) * c_s / six
            s2 = c_s - s1 - s3
            add = s1 * fs[k - 2] + s2 * fs[k - 1] + s3 * fs[k]
            sum1 = add + sum1

            if line == 1:
                continue  # goto 28

            delta = one - sum1
            if delta < 0:
                # Label 30: mass exceeded, interpolate
                s2_r = b_s * (two + c_s / a_s) / six
                s3_r = b_s * (two + a_s / c_s) / six
                s1_r = b_s - s2_r - s3_r
                bdd = s1_r * fs[k - 2] + s2_r * fs[k - 1] + s3_r * fs[k]

                if bdd + delta < 0:
                    # Label 31: between k-2 and k-1
                    dn = two * (add + delta) / a_s
                    disc = fs[k - 2]**2 + (fs[k - 1] - fs[k - 2]) * dn
                    sb = dn / (fs[k - 2] + np.sqrt(max(disc, 0)))
                    sa = one - sb
                    sc = zro
                elif bdd + delta == 0:
                    # Label 32
                    sa, sb, sc = zro, one, zro
                else:
                    # Label 33: between k-1 and k
                    dn = two * delta / b_s
                    disc = fs[k]**2 + (fs[k] - fs[k - 1]) * dn
                    sc = one + dn / (fs[k] + np.sqrt(max(disc, 0)))
                    sb = one - sc
                    sa = zro

                # Label 34: store wall point
                for jj in range(5):
                    wall[jj, line] = (b[jj, kt - 2] * sa
                                      + b[jj, kt - 1] * sb
                                      + b[jj, kt] * sc)
                last = kt
                return None

            elif delta == 0:
                # Label 29
                sa, sb, sc = zro, zro, one
                for jj in range(5):
                    wall[jj, line] = b[jj, kt] * sc
                last = kt
                return None

        # End of Simpson loop
        if line == 1:
            # First characteristic: return total mass, goto 16
            return sum1

        # Wall not found within loop — extrapolate (lines 282-287)
        if k + 1 <= lastp:
            b_s_ext = s[k + 1] - s[k]
            kt = k + 1
        else:
            b_s_ext = s[k] - s[k - 1]
            kt = k
        delta = one - sum1
        if abs(b_s_ext) > 1e-30 and abs(delta) > 1e-30:
            dn = two * delta / b_s_ext
            disc = fs[kt]**2 + (fs[kt] - fs[kt - 1]) * dn
            if disc > 0:
                sc = dn / (fs[kt] + np.sqrt(disc))
            else:
                sc = half
            sb = one - sc
            sa = zro
        else:
            sa, sb, sc = zro, zro, one

        for jj in range(5):
            wall[jj, line] = (b[jj, kt - 2] * sa
                              + b[jj, kt - 1] * sb
                              + b[jj, kt] * sc)
        last = kt
        return None

    # --- First characteristic (line=1) ---
    # perfc.f: goto 20 → mass integration → label 16
    total_mass = _mass_integrate_and_find_wall()

    # Label 16: last=m, line=2
    last = m
    line = 2

    # --- Axis march: lines 2..n ---
    # perfc.f labels 17, 20, 36
    while True:
        # Label 17: b(:,1) = taxi(:,line)
        for j in range(5):
            b[j, 1] = taxi[j, line]

        # March: do j=1,last: ofeld(a(:,j), b(:,j), b(:,j+1))
        for j_f in range(1, last + 1):
            c_pt, conv_flag = ofeld(a[:, j_f], b[:, j_f], gamma=gamma, ie=ie)
            if not conv_flag:
                last = j_f
                break
            b[:, j_f + 1] = c_pt

        # Label 20: mass integration
        _mass_integrate_and_find_wall()

        # Label 309: if (n-line) 42,41,36
        if n - line < 0:
            break  # goto 42
        elif n - line == 0:
            break  # goto 41 → 42 (for ip=0, lr≠0, it=0)
        else:
            # Label 36: advance to next line
            line = line + 1
            for k in range(5):
                for l in range(1, MAXPTS + 1):
                    a[k, l] = b[k, l]
            # ip=0: goto 17 (loop continues)

    # --- Throat characteristic matching: lines n+1..nl-1 ---
    # perfc.f labels 42, 46, 20
    while line < nl:
        # Label 42
        nn = nn + 1
        line = line + 1

        # a = b, b = fclast
        for k in range(5):
            for l in range(1, MAXPTS + 1):
                a[k, l] = b[k, l]
        for k in range(5):
            b[k, :] = 0.0  # clear
            for l in range(1, lq + 1):
                b[k, l] = fclast[k, l]

        # Label 46: do j=nn,last: ofeld(a(:,j), b(:,j), b(:,j+1))
        for j_f in range(nn, last + 1):
            c_pt, conv_flag = ofeld(a[:, j_f], b[:, j_f], gamma=gamma, ie=ie)
            if not conv_flag:
                last = j_f
                break
            b[:, j_f + 1] = c_pt

        # Label 20: mass integration
        _mass_integrate_and_find_wall()

        # Check: if line == nl-1, goto 48 (which leads to slope integration)
        if line >= nl - 1:
            break

    # --- Add throat wall point (perfc.f lines 362-363) ---
    # wall(:, line+1) = [xo, ?, 0, 0, 0]
    wall[0, line + 1] = xo
    wall[4, line + 1] = zro

    # --- Slope integration (perfc.f lines 357-399) ---
    # ib=1, lt=0 for standard case
    ib = 1
    lt = 0
    nut = (line - 1) // ib + 2 - lt  # perfc.f line 361

    yi = np.zeros(nut + 2)  # 1-indexed: yi(1..nut)
    yi[nut] = wall[1, 1]  # yi(nut) = wall(2,1) = y of inflection wall point
    y = yi[nut]

    lin = 2 * ((line - lt) // 2)
    for j_f in range(2, lin + 1, 2):  # Fortran: do j=2,lin,2
        i = nut - j_f

        ss = wall[0, j_f] - wall[0, j_f - 1]
        tt = wall[0, j_f + 1] - wall[0, j_f]
        st = ss + tt

        if abs(st) < 1e-30 or abs(ss) < 1e-30 or abs(tt) < 1e-30:
            continue

        s1 = ss * (two + tt / st) / six
        s2 = ss * (two + st / tt) / six
        s3 = ss - s1 - s2
        t3 = tt * (two + ss / st) / six
        t2 = tt * (two + st / ss) / six
        t1 = tt - t2 - t3

        y = y + (s1 * np.tan(wall[4, j_f - 1])
                 + s2 * np.tan(wall[4, j_f])
                 + s3 * np.tan(wall[4, j_f + 1]))
        yi[i + 1] = y  # Fortran: yi(i+1)=y (ib=1)

        y = y + (t1 * np.tan(wall[4, j_f - 1])
                 + t2 * np.tan(wall[4, j_f])
                 + t3 * np.tan(wall[4, j_f + 1]))
        yi[i] = y  # Fortran: yi(i)=y

    # Throat extrapolation (perfc.f line 385-386)
    # Skip if lr≠0 and line==lin (perfc.f line 384)
    x_thr = wall[0, line - lt] - xo
    if abs(x_thr) > 1e-30:
        yi[1] = yi[2] - x_thr * (np.tan(wall[4, line - lt]) + half * x_thr * sdo) / thr
    else:
        yi[1] = yi[2]

    # --- Rearrange output arrays (perfc.f lines 387-399) ---
    # Fortran: do l=2,nut: jj=1+ib*(nut-l); wax(l)=wall(1,jj)
    wax = np.zeros(nut)
    way = np.zeros(nut)
    wmn = np.zeros(nut)
    wan = np.zeros(nut)
    waltan = np.zeros(nut)

    # l=1 (Fortran): throat point
    wax[0] = xo
    way[0] = yo
    wan[0] = zro
    wmn[0] = wwo / np.sqrt(g7 - g8 * wwo**2)
    waltan[0] = zro

    for l_f in range(2, nut + 1):  # Fortran: do l=2,nut
        jj = 1 + ib * (nut - l_f)  # Fortran 1-indexed wall index
        l = l_f - 1  # Python 0-indexed output index
        wax[l] = wall[0, jj]
        way[l] = wall[1, jj]  # will be overridden by yi
        wmn[l] = wall[2, jj]
        wan[l] = conv * wall[4, jj]
        waltan[l] = np.tan(wall[4, jj])

    # Override y with integrated values (yi is 1-indexed: yi(1..nut))
    way[0] = yo
    for l_f in range(2, nut + 1):
        way[l_f - 1] = yi[l_f]

    # Second derivatives
    secd = parabolic_derivative(wax, waltan)
    secd[0] = sdo
    secd[-1] = zro

    return {
        'wax': wax,
        'way': way,
        'wmn': wmn,
        'wan': wan,
        'waltan': waltan,
        'secd': secd,
        'wall_raw': wall[:, 1:nl + 2],  # convert to 0-indexed output
        'mass': total_mass,
        'n_wall': nut,
    }
