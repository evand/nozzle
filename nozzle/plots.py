"""Visualization for nozzle contours and MOC results.

All plot functions return (fig, ax) tuples for composability.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_contour(x, y, label=None, ax=None, title="Nozzle Contour",
                 show_axis=True, mirror=True):
    """Plot a nozzle contour (upper wall ± mirrored lower wall).

    Parameters
    ----------
    x, y : ndarray
        Contour coordinates (normalized by throat radius).
    label : str or None
        Legend label.
    ax : matplotlib Axes or None
        Existing axes to plot on.
    title : str
        Plot title.
    show_axis : bool
        Draw centerline at y=0.
    mirror : bool
        Mirror contour below axis.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(x, y, 'b-', linewidth=2, label=label)
    if mirror:
        ax.plot(x, -y, 'b-', linewidth=2)
    if show_axis:
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--')

    ax.set_xlabel("x / r*")
    ax.set_ylabel("y / r*")
    ax.set_title(title)
    ax.set_aspect('equal')
    if label:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_contour_comparison(contours, title="Nozzle Contour Comparison"):
    """Plot multiple contours overlaid.

    Parameters
    ----------
    contours : list of (x, y, label) tuples
    title : str

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    colors = plt.cm.tab10.colors

    for i, (x, y, label) in enumerate(contours):
        color = colors[i % len(colors)]
        ax.plot(x, y, '-', color=color, linewidth=2, label=label)
        ax.plot(x, -y, '-', color=color, linewidth=2)

    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel("x / r*")
    ax.set_ylabel("y / r*")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_char_mesh(mesh, ax=None, title="Characteristic Mesh",
                   show_wall=True, show_axis=True):
    """Plot the characteristic mesh from MOC solution.

    Parameters
    ----------
    mesh : CharMesh
        MOC solution mesh.
    ax : matplotlib Axes or None
    title : str
    show_wall : bool
        Highlight wall points.
    show_axis : bool
        Highlight axis points.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    else:
        fig = ax.figure

    # Plot C+ and C- characteristics as lines between connected points
    for pt in mesh.points:
        if pt.left_idx is not None:
            lp = mesh.points[pt.left_idx]
            ax.plot([lp.x, pt.x], [lp.y, pt.y], 'b-', linewidth=0.3, alpha=0.5)
        if pt.right_idx is not None:
            rp = mesh.points[pt.right_idx]
            ax.plot([rp.x, pt.x], [rp.y, pt.y], 'r-', linewidth=0.3, alpha=0.5)

    # Highlight special points
    xs = np.array([p.x for p in mesh.points])
    ys = np.array([p.y for p in mesh.points])

    if show_wall:
        wall_mask = np.array([p.is_wall for p in mesh.points])
        if wall_mask.any():
            ax.plot(xs[wall_mask], ys[wall_mask], 'k-', linewidth=2, label='Wall')

    if show_axis:
        axis_mask = np.array([p.is_axis for p in mesh.points])
        if axis_mask.any():
            ax.plot(xs[axis_mask], ys[axis_mask], 'g.', markersize=3, label='Axis')

    ax.set_xlabel("x / r*")
    ax.set_ylabel("y / r*")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_exit_distributions(y_exit, M_exit, theta_exit=None, ax=None,
                            title="Exit Plane Distributions"):
    """Plot Mach and flow angle distributions at exit plane.

    Parameters
    ----------
    y_exit : ndarray
        Radial positions at exit.
    M_exit : ndarray
        Mach numbers at exit.
    theta_exit : ndarray or None
        Flow angles at exit [radians].
    ax : Axes or None

    Returns
    -------
    fig, axes
    """
    n_plots = 2 if theta_exit is not None else 1
    if ax is None:
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]

    axes[0].plot(y_exit, M_exit, 'b-o', markersize=3)
    axes[0].set_xlabel("y / r*")
    axes[0].set_ylabel("Mach number")
    axes[0].set_title("Exit Mach Distribution")
    axes[0].grid(True, alpha=0.3)

    if theta_exit is not None and n_plots > 1:
        axes[1].plot(y_exit, np.degrees(theta_exit), 'r-o', markersize=3)
        axes[1].set_xlabel("y / r*")
        axes[1].set_ylabel("Flow angle [°]")
        axes[1].set_title("Exit Flow Angle")
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_performance_comparison(results, title="Performance Comparison"):
    """Bar chart comparing Cf values across nozzle types.

    Parameters
    ----------
    results : list of (name, Cf) tuples

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r[0] for r in results]
    cfs = [r[1] for r in results]

    bars = ax.bar(names, cfs, color=plt.cm.tab10.colors[:len(names)])
    ax.set_ylabel("Thrust Coefficient Cf")
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, cf in zip(bars, cfs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{cf:.4f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    return fig, ax


def plot_contour_delta(x_a, y_a, label_a, x_b, y_b, label_b, title=None):
    """Plot shape difference between two contours.

    Two panels: true-scale (same aspect as contour plot) and exaggerated
    (auto-scaled y to show detail).

    Parameters
    ----------
    x_a, y_a : ndarray
        First contour (reference).
    label_a : str
        Name of first contour.
    x_b, y_b : ndarray
        Second contour (comparison).
    label_b : str
        Name of second contour.
    title : str or None

    Returns
    -------
    fig, axes
    """
    # Common x grid over shared domain
    x_lo = max(x_a[0], x_b[0])
    x_hi = min(x_a[-1], x_b[-1])
    x_common = np.linspace(x_lo, x_hi, 500)

    ya_interp = np.interp(x_common, x_a, y_a)
    yb_interp = np.interp(x_common, x_b, y_b)
    dy = ya_interp - yb_interp

    if title is None:
        title = f"Shape Delta: {label_a} \u2212 {label_b}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: true scale
    ax = axes[0]
    ax.fill_between(x_common, 0, dy, where=(dy >= 0),
                    color='#4a8eff', alpha=0.4, label=f'{label_a} wider')
    ax.fill_between(x_common, 0, dy, where=(dy < 0),
                    color='#ff6b6b', alpha=0.4, label=f'{label_b} wider')
    ax.plot(x_common, dy, 'k-', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.set_xlabel("x / r*")
    ax.set_ylabel("\u0394y / r*")
    ax.set_title("True Scale")
    ax.set_aspect('equal')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Panel 2: exaggerated
    ax = axes[1]
    ax.fill_between(x_common, 0, dy, where=(dy >= 0),
                    color='#4a8eff', alpha=0.4, label=f'{label_a} wider')
    ax.fill_between(x_common, 0, dy, where=(dy < 0),
                    color='#ff6b6b', alpha=0.4, label=f'{label_b} wider')
    ax.plot(x_common, dy, 'k-', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.set_xlabel("x / r*")
    ax.set_ylabel("\u0394y / r*")
    ax.set_title("Exaggerated")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    # Auto-scale y to show detail
    dy_max = max(abs(dy.max()), abs(dy.min()), 1e-6)
    ax.set_ylim(-1.5 * dy_max, 1.5 * dy_max)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig, axes
