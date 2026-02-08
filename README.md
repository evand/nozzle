# Nozzle — MOC Rocket Nozzle Design Tool

Method of Characteristics solver for axisymmetric supersonic nozzle design.
Designs optimized contours (MLN, Rao, TIC, Sivells), generates parabolic
approximations, and compares them — all from YAML config files.

**[Try the live web demo](https://evand.github.io/nozzle/)**  — runs entirely in your browser via Pyodide.

## Nozzle Types

| Type | Method | Use Case |
|------|--------|----------|
| **Conical** | Straight divergent wall | Baseline / simplest design |
| **Rao parabolic** | Cubic bezier bell (Rao 1960) | Industry-standard short nozzle |
| **TIC** | Truncated ideal contour | Trade length vs. performance |
| **MLN** | Minimum length nozzle (MOC) | Theoretical optimum (uniform exit) |
| **Sivells** | MOC + mass flow (AEDC-TR-78-63) | High-fidelity design to M~6 |
| **Custom** | User CSV contour | Evaluate any arbitrary shape |

## Install

```bash
pip install -e .
```

Requires Python 3.9+ and numpy, scipy, matplotlib, pyyaml, unyt.

## Quick Start

```bash
# Compare all nozzle types at M=2.0
nozzle run examples/configs/all_types.yaml

# Quick example (no config file needed)
nozzle example --M-exit 2.5

# TIC truncation trade study
nozzle run examples/configs/tic_comparison.yaml

# High-Mach (M=4) with Sivells MOC
nozzle run examples/configs/high_mach.yaml

# Launch interactive web interface (Pyodide)
nozzle web
```

### Example Output

```
  Conical 15.0°: Cf=1.3992, λ=0.9830, M_exit=2.000
  Rao 80% bell: θ_n=17.0°, θ_e=8.5°, Cf=1.4156 (λ=0.9945)
  TIC 80% M=2.0: θ_e=3.1°, Cf=1.4177 (λ=0.9993)
  MLN M=2.0: Cf=1.4234, M_mean=2.000, efficiency=1.0000

Nozzle          Cf   % Ideal   Notes
-------   --------   -------   -----
conical     1.3992     98.3%   lambda=0.9830
rao         1.4156     99.5%   theta_n=17.0 theta_e=8.5
tic_80      1.4177     99.9%   lambda=0.9993
mln         1.4234    100.0%   M_mean=2.000
```

Outputs per config: contour PNG, contour CSV, exit plane CSV, and a
`summary.json` with all performance metrics. Comparison runs also produce
overlaid contour plots, shape delta plots, tolerance band analysis, and
performance bar charts.

## YAML Configuration

```yaml
configs:
  rao:
    type: rao
    gamma: 1.4
    M_exit: 2.0
    bell_fraction: 0.8

  mln:
    type: mln
    gamma: 1.4
    M_exit: 2.0
    n_chars: 30

outputs:
  - contour
  - performance
```

Features:
- **Config inheritance** — `base:` key merges parent config with overrides
- **Flexible exit conditions** — specify any one of `M_exit`, `area_ratio`, or `exit_radius`
- **Dimensional I/O** — set `throat_radius: 15 mm` for physical units in output CSVs

See `examples/configs/` for all available examples.

## Python API

```python
from nozzle.contours import minimum_length_nozzle, rao_parabolic_nozzle
from nozzle.analysis import moc_performance, quasi_1d_performance

# Design an MLN at M=3.0
x, y, mesh = minimum_length_nozzle(M_exit=3.0, n_chars=30)
perf = moc_performance(mesh)
print(f"Cf = {perf['Cf']:.4f}")

# Rao 80% bell at AR=10
x, y, theta_n, theta_e = rao_parabolic_nozzle(area_ratio=10, bell_fraction=0.8)
```

## Web Interface

The interactive web viewer runs the full Python solver in-browser via
[Pyodide](https://pyodide.org/). Toggle nozzle types, adjust parameters,
and see contour overlays, performance tables, tolerance band analysis, and
exit plane distributions — all without a server.

- **Live demo**: [evand.github.io/nozzle](https://evand.github.io/nozzle/)
- **Local**: `nozzle web` (or `python web/serve.py`)

## Validation

359 tests, validated against published references:

| Source | Coverage |
|--------|----------|
| Anderson *Modern Compressible Flow* Tables A.1/A.5 | Gas relations (exact to 4+ digits) |
| Sutton & Biblarz Tables 3-3/3-4 | Rao angles, conical lambda, Cf |
| CONTUR Fortran (Sivells AEDC-TR-78-63) | 122 tests at machine precision |
| Physics invariants | Monotonicity, conservation, ordering |

See `docs/VALIDATION.md` for the full module-by-module matrix.

```bash
pytest tests/
```

## Project Structure

```
nozzle/
  gas.py        — Isentropic + Prandtl-Meyer relations
  kernel.py     — Hall transonic initial data line
  moc.py        — MOC unit processes and mesh
  contours.py   — Nozzle contour generators
  sivells.py    — Sivells CONTUR port (AEDC-TR-78-63)
  analysis.py   — Performance analysis (Cf, efficiency)
  config.py     — YAML config with inheritance
  cli.py        — CLI entry point
  plots.py      — Visualization (contours, deltas, tolerance bands)
web/            — Browser-based interface (Pyodide)
tests/          — 359 tests
examples/       — YAML configs for common use cases
docs/           — Algorithms, references, validation matrix
```

## References

- Sivells, AEDC-TR-78-63, 1978 — Axisymmetric nozzle design via MOC
- Anderson, *Modern Compressible Flow*, 3rd ed., McGraw-Hill 2003
- Zucrow & Hoffman, *Gas Dynamics* Vol. 2, Wiley 1977
- Rao, *Jet Propulsion* 1958; *ARS Journal* 1960
- Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed.
- Hall, *QJMAM* 1962 — Transonic kernel
- NACA 1135, 1953 — Isentropic flow tables
