# Nozzle Design Tool — Specification

## Purpose

A method of characteristics solver for supersonic rocket nozzle design and
comparison. Fills the gap between simple analytical approximations and full CFD.
Constant gas properties, no combustion kinetics.

## Inputs

The tool accepts nozzle design parameters via YAML config files:

- **Gas properties**: gamma (ratio of specific heats)
- **Throat geometry**: throat radius (dimensional, e.g. mm/in) or normalized
- **Exit conditions**: exit Mach number, exit diameter, or expansion ratio
  (any one determines the others via isentropic relations)
- **Length constraint**: bell fraction (% of equivalent conical length) or
  absolute length
- **Nozzle type**: conical, Rao parabolic, MLN (minimum length), TIC
  (truncated ideal contour), or custom (arbitrary contour from CSV)

## Outputs

### Data
- Contour coordinates (x, y) as CSV, in both normalized (r*) and dimensional units
- Performance summary as JSON: Cf, efficiency, exit Mach uniformity, flow angles
- Exit plane distributions: M(y), theta(y)

### Plots
- Contour comparison: overlaid profiles (to scale, with mirrored lower wall)
- Shape delta: difference between two contours (to scale and exaggerated)
- Exit plane: Mach and flow angle distributions across the exit
- Performance bar chart: Cf comparison across nozzle types

### Console
- Tabular performance summary for each config

## Nozzle Types

| Type | Method | Performance |
|------|--------|-------------|
| Conical | Straight divergent wall at half-angle | Analytical (lambda) |
| Rao parabolic | Cubic bezier approximation of ideal bell | Analytical (lambda) |
| MLN | Method of characteristics, minimum length | MOC exit plane integration |
| TIC | Truncated ideal contour | Quasi-1D with divergence loss |
| Custom CSV | User-supplied (x, y) contour | MOC analysis (future) |

## Comparisons

Given multiple configs, the tool compares them:
- Overlaid contour plots
- Performance table (Cf, % of ideal, notes)
- Shape delta between any two contours (to scale + exaggerated)
- Arbitrary CSV contours participate in comparison on equal footing

## Dimensional I/O

Internally, the solver works in throat-radius-normalized coordinates (r* = 1).
The config specifies `throat_radius` with units; all outputs are available in
both normalized and dimensional form.

## Tolerance Band Visualization

For manufacturing validation, overlay multiple CSV contours (nominal + tolerance
limits) on a single plot, with the delta view showing the band width.

## Validation

- Gas relations: vs Anderson tables (exact)
- Unit processes: vs 2D planar at large y (exact)
- Conical Cf: vs analytical lambda formula (< 1%)
- Rao angles: vs Sutton tables (+/- 2 deg)
- MLN Cf: vs 1D ideal (< 0.1%)
- Sivells port: vs CONTUR Fortran output (< 1e-6)
- All algorithms cite published sources in code

## Interfaces

- **CLI**: `nozzle run config.yaml [-o output_dir]`, `nozzle example [--M-exit N]`
- **Web**: Static page via Pyodide (same Python source, no duplication)
- **Python API**: `from nozzle.contours import minimum_length_nozzle` etc.

## Non-Goals

- Combustion kinetics / variable composition
- CFD / Navier-Stokes
- 3D geometry (axisymmetric only)
- Boundary layer correction (deferred, Sivells Phase 6)
