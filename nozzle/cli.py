"""Command-line interface for nozzle design tool.

Usage:
    nozzle run config.yaml [--output-dir DIR]
    nozzle example [--M-exit 2.0] [--plot]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from nozzle.config import load_config, build_nozzle_spec
from nozzle.contours import (
    conical_nozzle, rao_parabolic_nozzle, minimum_length_nozzle,
    truncated_ideal_contour, conical_divergence_loss, load_contour_csv,
    sivells_nozzle,
)
from nozzle.analysis import (
    conical_performance, rao_performance, moc_performance,
    quasi_1d_performance,
)
from nozzle.gas import area_mach_ratio, mach_from_area_ratio, thrust_coefficient_ideal
from nozzle.plots import (
    plot_contour, plot_contour_comparison, plot_performance_comparison,
    plot_contour_delta,
)


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='nozzle',
        description='MOC rocket nozzle design tool',
    )
    subparsers = parser.add_subparsers(dest='command')

    # --- run command ---
    run_parser = subparsers.add_parser('run', help='Run from config file')
    run_parser.add_argument('config', type=str, help='YAML config file')
    run_parser.add_argument('--output-dir', '-o', default=None,
                            help='Output directory (default: ./output)')

    # --- example command ---
    example_parser = subparsers.add_parser('example', help='Quick example')
    example_parser.add_argument('--M-exit', type=float, default=2.0,
                                help='Exit Mach number')
    example_parser.add_argument('--area-ratio', type=float, default=None,
                                help='Area ratio (computed from M if omitted)')
    example_parser.add_argument('--plot', action='store_true',
                                help='Show plots interactively')

    # --- web command ---
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--port', type=int, default=8080,
                            help='Port to serve on (default: 8080)')

    parsed = parser.parse_args(args)

    if parsed.command == 'run':
        return cmd_run(parsed)
    elif parsed.command == 'example':
        return cmd_example(parsed)
    elif parsed.command == 'web':
        return cmd_web(parsed)
    else:
        parser.print_help()
        return 1


def cmd_run(args):
    """Run nozzle design from YAML config file."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = load_config(config_path)
    configs = spec['configs']
    comparisons = spec.get('comparisons', [])
    outputs = spec.get('outputs', ['contour', 'performance'])

    # Process each config
    results = {}
    contours = []

    for name, cfg in configs.items():
        nozzle_spec = build_nozzle_spec(cfg)
        result = _run_single(nozzle_spec, name, output_dir, outputs)
        results[name] = result

        if result.get('x_wall') is not None:
            contours.append((result['x_wall'], result['y_wall'], name))

    # Print summary table
    if 'performance' in outputs and results:
        _print_summary_table(results)

    # Generate comparisons
    if len(contours) > 1 and 'contour' in outputs:
        fig, ax = plot_contour_comparison(contours)
        fig.savefig(output_dir / 'contour_comparison.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved contour comparison to {output_dir / 'contour_comparison.png'}")

        # Delta plots for each pair of contours
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                x_a, y_a, name_a = contours[i]
                x_b, y_b, name_b = contours[j]
                fig, _ = plot_contour_delta(x_a, y_a, name_a,
                                            x_b, y_b, name_b)
                fname = f'delta_{name_a}_vs_{name_b}.png'
                fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved delta plot to {output_dir / fname}")

    if len(results) > 1 and 'performance' in outputs:
        perf_data = [(name, r.get('Cf', 0)) for name, r in results.items()
                     if r.get('Cf') is not None]
        if perf_data:
            fig, ax = plot_performance_comparison(perf_data)
            fig.savefig(output_dir / 'performance_comparison.png', dpi=150,
                        bbox_inches='tight')
            plt.close(fig)
            print(f"Saved performance comparison to {output_dir / 'performance_comparison.png'}")

    # Export performance summary JSON
    if 'performance' in outputs:
        import json
        summary = {}
        for name, r in results.items():
            entry = {'type': r.get('type'), 'gamma': r.get('gamma'),
                     'area_ratio': r.get('area_ratio')}
            for key in ('Cf', 'Cf_ideal', 'efficiency', 'M_mean', 'M_exit',
                        'M_max', 'M_min', 'theta_max_deg', 'lambda',
                        'theta_n_deg', 'theta_e_deg'):
                if key in r:
                    entry[key] = r[key]
            summary[name] = entry
        json_path = output_dir / 'summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved performance summary to {json_path}")

    return 0


def _print_summary_table(results):
    """Print an aligned performance summary table."""
    # Compute 1D ideal for reference
    rows = []
    for name, r in results.items():
        cf = r.get('Cf')
        cf_ideal = r.get('Cf_ideal')
        if cf is None:
            continue
        pct = (cf / cf_ideal * 100) if cf_ideal else 0
        note = ''
        rtype = r.get('type', '')
        if rtype == 'conical':
            note = f"lambda={r.get('lambda', 0):.4f}"
        elif rtype == 'rao':
            note = (f"theta_n={r.get('theta_n_deg', 0):.1f} "
                    f"theta_e={r.get('theta_e_deg', 0):.1f}")
        elif rtype in ('mln', 'tic'):
            note = f"M_mean={r.get('M_mean', 0):.3f}"
        elif rtype == 'sivells':
            note = f"lambda={r.get('lambda', 0):.4f} (upstream only)"
        elif rtype == 'custom':
            note = f"lambda={r.get('lambda', 0):.4f} (quasi-1D)"
        rows.append((name, cf, pct, note))

    if not rows:
        return

    # Column widths
    w_name = max(len(r[0]) for r in rows)
    w_name = max(w_name, 6)  # "Nozzle" header

    print(f"\n{'Nozzle':<{w_name}}   {'Cf':>8}   {'% Ideal':>7}   Notes")
    print(f"{'-' * w_name}   {'--------':>8}   {'-------':>7}   -----")
    for name, cf, pct, note in rows:
        print(f"{name:<{w_name}}   {cf:>8.4f}   {pct:>6.1f}%   {note}")
    print()


def _write_contour_csv(path, x, y, name, spec):
    """Write contour coordinates to CSV.

    Includes dimensional columns (x_mm, y_mm) when throat_radius_m is set.
    """
    r_t = spec.get('throat_radius_m')
    with open(path, 'w') as f:
        f.write(f"# {name} nozzle contour\n")
        f.write(f"# type={spec['type']} gamma={spec['gamma']}"
                f" area_ratio={spec['area_ratio']:.4f}\n")
        if r_t is not None:
            f.write(f"# throat_radius={r_t*1e3:.4f} mm\n")
            f.write("# x/r*,y/r*,x_mm,y_mm\n")
            for xi, yi in zip(x, y):
                f.write(f"{xi:.8f},{yi:.8f},"
                        f"{xi * r_t * 1e3:.6f},{yi * r_t * 1e3:.6f}\n")
        else:
            f.write("# x/r*,y/r*\n")
            for xi, yi in zip(x, y):
                f.write(f"{xi:.8f},{yi:.8f}\n")


def _write_exit_plane_csv(path, y, M, theta, name):
    """Write exit plane distributions to CSV."""
    with open(path, 'w') as f:
        f.write(f"# {name} exit plane distribution\n")
        f.write("# y/r*,Mach,theta_deg\n")
        for yi, Mi, ti in zip(y, M, theta):
            f.write(f"{yi:.8f},{Mi:.8f},{np.degrees(ti):.8f}\n")


def _run_single(spec, name, output_dir, outputs):
    """Run a single nozzle configuration."""
    ntype = spec['type']
    gamma = spec['gamma']
    area_ratio = spec['area_ratio']
    result = {'type': ntype, 'gamma': gamma, 'area_ratio': area_ratio}

    if ntype == 'conical':
        half_angle = spec.get('half_angle_deg', 15)
        x_wall, y_wall = conical_nozzle(half_angle, area_ratio)
        perf = conical_performance(half_angle, area_ratio, gamma)
        result.update(perf)
        result['x_wall'] = x_wall
        result['y_wall'] = y_wall
        print(f"  Conical {half_angle}°: Cf={perf['Cf']:.4f}, "
              f"λ={perf['lambda']:.4f}, M_exit={perf['M_exit']:.3f}")

    elif ntype == 'rao':
        bell = spec.get('bell_fraction', 0.8)
        x_wall, y_wall, theta_n, theta_e = rao_parabolic_nozzle(
            area_ratio, bell, gamma
        )
        perf = rao_performance(area_ratio, bell, gamma)
        result.update(perf)
        result['x_wall'] = x_wall
        result['y_wall'] = y_wall
        print(f"  Rao {bell*100:.0f}% bell: θ_n={perf['theta_n_deg']:.1f}°, "
              f"θ_e={perf['theta_e_deg']:.1f}°, "
              f"Cf={perf['Cf']:.4f} (λ={perf['lambda']:.4f})")

    elif ntype == 'mln':
        M_exit = spec['M_exit']
        n_chars = spec.get('n_chars', 30)
        x_wall, y_wall, mesh = minimum_length_nozzle(M_exit, n_chars, gamma)
        perf = moc_performance(mesh, gamma)
        result.update(perf)
        result['x_wall'] = x_wall
        result['y_wall'] = y_wall
        result['mesh'] = mesh
        print(f"  MLN M={M_exit:.1f}: Cf={perf['Cf']:.4f}, "
              f"M_mean={perf['M_mean']:.3f}, efficiency={perf['efficiency']:.4f}")

    elif ntype == 'tic':
        M_exit = spec['M_exit']
        n_chars = spec.get('n_chars', 30)
        trunc_frac = spec.get('truncation_fraction', 0.8)
        x_wall, y_wall, mesh = truncated_ideal_contour(
            M_exit, trunc_frac, n_chars, gamma
        )
        perf = moc_performance(mesh, gamma)
        result.update(perf)
        result['x_wall'] = x_wall
        result['y_wall'] = y_wall
        result['mesh'] = mesh
        # TODO: Cf is evaluated at full MLN exit plane, not the truncated exit.
        # Fix as part of comparisons task (evaluate at x_trunc cross-section).
        print(f"  TIC {trunc_frac*100:.0f}% M={M_exit:.1f}: Cf={perf['Cf']:.4f}, "
              f"M_mean={perf['M_mean']:.3f} (Cf from full MLN mesh)")

    elif ntype == 'sivells':
        M_exit = spec['M_exit']
        x_wall, y_wall = sivells_nozzle(
            M_exit, gamma=gamma,
            rc=spec.get('rc', 1.5),
            inflection_angle_deg=spec.get('inflection_angle_deg'),
            n_char=spec.get('n_char', 41),
            n_axis=spec.get('n_axis', 21),
            nx=spec.get('nx', 13),
            ix=spec.get('ix', 0),
            ie=spec.get('ie', 0),
        )
        perf = quasi_1d_performance(x_wall, y_wall, gamma)
        result.update(perf)
        result['x_wall'] = x_wall
        result['y_wall'] = y_wall
        eta = spec.get('inflection_angle_deg')
        eta_str = f"{eta:.1f}" if eta is not None else "auto"
        print(f"  Sivells M={M_exit:.1f}: {len(x_wall)} pts, "
              f"θ_infl={eta_str}°, "
              f"Cf={perf['Cf']:.4f} (quasi-1D, λ={perf['lambda']:.4f})")

    elif ntype == 'custom':
        contour_file = spec.get('contour_file', '')
        n_chars = spec.get('n_chars', 20)
        if not contour_file:
            print(f"  Error: custom type requires 'contour_file'")
            result['x_wall'] = None
            result['y_wall'] = None
        else:
            x_wall, y_wall = load_contour_csv(contour_file)
            perf = quasi_1d_performance(x_wall, y_wall, gamma)
            result.update(perf)
            result['x_wall'] = x_wall
            result['y_wall'] = y_wall
            print(f"  Custom contour from {contour_file}: "
                  f"{len(x_wall)} points, AR={perf['area_ratio']:.2f}, "
                  f"Cf={perf['Cf']:.4f} (quasi-1D, "
                  f"λ={perf['lambda']:.4f})")

    else:
        print(f"  Unknown type: {ntype}")
        result['x_wall'] = None
        result['y_wall'] = None

    # Save individual contour plot
    if result.get('x_wall') is not None and 'contour' in outputs:
        fig, ax = plot_contour(result['x_wall'], result['y_wall'],
                               label=name, title=f"{name} Nozzle Contour")
        fig.savefig(output_dir / f'{name}_contour.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        # Export contour CSV
        csv_path = output_dir / f'{name}_contour.csv'
        _write_contour_csv(csv_path, result['x_wall'], result['y_wall'],
                           name, spec)

    # Export exit plane CSV for types that have MOC mesh
    if result.get('mesh') is not None and 'performance' in outputs:
        mesh = result['mesh']
        y_exit, M_exit, theta_exit = mesh.get_exit_plane()
        if len(y_exit) > 1:
            csv_path = output_dir / f'{name}_exit_plane.csv'
            _write_exit_plane_csv(csv_path, y_exit, M_exit, theta_exit, name)

    return result


def cmd_example(args):
    """Run a quick example comparing conical, Rao, and MLN nozzles."""
    M_exit = args.M_exit
    area_ratio = args.area_ratio
    if area_ratio is None:
        area_ratio = area_mach_ratio(M_exit)

    print(f"Nozzle Design Comparison")
    print(f"  M_exit = {M_exit:.2f}")
    print(f"  A/A*   = {area_ratio:.4f}")
    print(f"  γ      = 1.4")

    # 1. Conical 15°
    print(f"\n1. Conical 15° half-angle:")
    x_con, y_con = conical_nozzle(15, area_ratio)
    perf_con = conical_performance(15, area_ratio)
    print(f"   Cf = {perf_con['Cf']:.4f} (λ = {perf_con['lambda']:.4f})")

    # 2. Rao 80% bell
    print(f"\n2. Rao 80% bell:")
    x_rao, y_rao, theta_n, theta_e = rao_parabolic_nozzle(area_ratio, 0.8)
    perf_rao = rao_performance(area_ratio, 0.8)
    print(f"   θ_n = {np.degrees(theta_n):.1f}°, θ_e = {np.degrees(theta_e):.1f}°")
    print(f"   Cf = {perf_rao['Cf']:.4f} (λ = {perf_rao['lambda']:.4f})")
    print(f"   Length = {x_rao[-1]:.2f} · r* (vs {x_con[-1]:.2f} for conical)")

    # 3. MLN
    print(f"\n3. Minimum Length Nozzle:")
    x_mln, y_mln, mesh = minimum_length_nozzle(M_exit, n_chars=20)
    perf_mln = moc_performance(mesh)
    print(f"   Cf = {perf_mln['Cf']:.4f}")
    print(f"   M_mean = {perf_mln['M_mean']:.3f}")

    # 4. Summary
    Cf_ideal = thrust_coefficient_ideal(M_exit)
    print(f"\nSummary:")
    print(f"   1D Ideal Cf = {Cf_ideal:.4f}")
    print(f"   Conical  Cf = {perf_con['Cf']:.4f} ({perf_con['Cf']/Cf_ideal*100:.1f}%)")
    print(f"   Rao      Cf = {perf_rao['Cf']:.4f} ({perf_rao['Cf']/Cf_ideal*100:.1f}%)")
    print(f"   MLN      Cf = {perf_mln['Cf']:.4f} ({perf_mln['Cf']/Cf_ideal*100:.1f}%)")

    if args.plot:
        matplotlib.use('TkAgg')
        contours = [
            (x_con, y_con, f"Conical 15°"),
            (x_rao, y_rao, f"Rao 80% bell"),
            (x_mln, y_mln, f"MLN"),
        ]
        fig, ax = plot_contour_comparison(contours,
                                          title=f"Nozzle Comparison (M={M_exit})")
        plt.show()

    return 0


def cmd_web(args):
    """Launch the web interface.

    Serves web/ as the document root. If web/nozzle/ doesn't exist
    (build.py hasn't been run), falls back to serving nozzle/*.py
    from the source tree.
    """
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    web_dir = Path(__file__).resolve().parent.parent / 'web'
    src_dir = Path(__file__).resolve().parent  # nozzle/ package

    if not web_dir.exists():
        print(f"Error: web directory not found at {web_dir}")
        return 1

    # Auto-run build if web/nozzle/ is missing
    built_dir = web_dir / 'nozzle'
    if not built_dir.exists() or not (built_dir / 'gas.py').exists():
        print("Running web/build.py to copy source files...")
        import subprocess
        subprocess.run([sys.executable, str(web_dir / 'build.py')], check=True)

    class NozzleHandler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            # Serve everything from web/
            rel = path.lstrip('/')
            candidate = web_dir / rel
            # Fallback: if web/nozzle/X.py missing, try source nozzle/X.py
            if not candidate.exists() and rel.startswith('nozzle/'):
                source = src_dir / rel[len('nozzle/'):]
                if source.exists():
                    return str(source)
            return str(candidate)

        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            super().end_headers()

        def log_message(self, format, *a):
            pass  # Suppress request logging

    port = args.port
    server = HTTPServer(('localhost', port), NozzleHandler)
    print(f"Nozzle web interface: http://localhost:{port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

    return 0
