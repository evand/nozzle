# Completed Tasks

Move items here from TODO.md when done. Most recent at top.

---

## Tier 1 features: exit pressure, convergent section, web enhancements
Four features landed together:
- **Exit pressure as input**: `exit_pressure_ratio` in config, lowest priority after
  exit_radius/M_exit/area_ratio. 8 new tests in test_config.py.
- **Convergent section geometry**: `convergent_section()` in contours.py — 4-segment
  wall (cylinder + upstream arc + straight + downstream arc) with auto-scaling radii.
  Optional `convergent:` sub-dict in config, CLI prepends to any nozzle type.
  13 new tests in test_convergent.py.
- **Custom contour in web viewer**: File upload card, CSV written to Pyodide FS,
  processed via load_contour_csv + quasi_1d_performance, appears in plots/table.
- **Physical units in web viewer**: Throat radius (mm) input, all plots/exports
  scale to mm when set, performance table shows exit diameter and throat area.
- **Expansion ratio in web viewer**: A/A* input syncs bidirectionally with M_exit.

380 tests, all passing.
*Completed 2026-02-08*

## Tolerance band visualization
`plot_tolerance_band()` in plots.py — 3-panel comparison (contour overlay, delta
radius, area ratio deviation %). CLI generates tolerance plots for every contour
pair alongside delta plots. Web viewer shows up to 3 pairwise tolerance plots
with dark theme styling. 4 new tests in test_plots.py, 359 total, all passing.
*Completed 2026-02-08*

## README and example configs
Updated README.md with install, quick start, example output, validation summary,
project structure, and Python API examples. Added 3 new example configs:
`all_types.yaml` (all nozzle types at M=2), `tic_comparison.yaml` (TIC trade
study at M=2.5), `high_mach.yaml` (M=4 with Sivells downstream). Created
`docs/VALIDATION.md` validation matrix mapping every module to its reference
source and tolerance.
*Completed 2026-02-08*

## TIC performance at truncated exit
Switched TIC from `moc_performance(mesh)` (evaluated at full MLN exit) to
`quasi_1d_performance(x_wall, y_wall)` (evaluated at truncated exit). Now
correctly computes M_exit from truncated area ratio and λ = (1+cos θ_e)/2
from wall angle at truncation. Summary table shows lambda for TIC.
3 new performance tests in test_tic.py, 355 total, all passing.
*Completed 2026-02-08*

## Sivells downstream contour
`sivells_axial_downstream()` and `sivells_perfc_downstream()` — downstream axis
polynomial + MOC march with reversed ofeld and mass integration. Integrated into
`sivells_nozzle(downstream=True)` with linear bridge at inflection gap.
Config/CLI support for downstream, ip, md, nd, nf params. 51 downstream tests,
352 total, all passing. Validated against CONTUR Mach 4 planar output.
*Completed 2026-02-08*

## Sivells axisymmetric validation
28 tests comparing sivells_axial and sivells_perfc with ie=1 against CONTUR F90
axisymmetric Mach 4 output. Matches to x~1e-8, M~1e-8, angle~0.02 deg.
*Completed 2026-02-07*

## Web viewer overhaul
Toggle-based UI for all 5 nozzle types (conical, rao, mln, tic, sivells) with
type-specific parameters. CSV per-type and YAML config export. Sivells module
added to Pyodide build. Dynamic computation based on enabled types.
*Completed 2026-02-07*

## Sivells CLI integration
`sivells_nozzle()` wrapper in contours.py, `type: sivells` in config/CLI.
Auto-defaults for inflection angle and bmach. Example config and integration
tests. Performance via quasi_1d_performance.
*Completed 2026-02-07*

## Structured console output
Aligned summary table with Cf, % Ideal, and type-specific notes printed after
all configs run. Replaces ad-hoc per-config print statements.
*Completed 2026-02-07*

## Custom contour quasi-1D performance
`quasi_1d_performance()` in analysis.py estimates Cf for CSV contours using
area-Mach relation and exit divergence angle (lambda). Custom contours now
participate in performance comparisons.
*Completed 2026-02-07*

## Input flexibility: M_exit / exit_radius / area_ratio
Config now accepts any one of M_exit, exit_radius, or area_ratio. The others
are computed via isentropic relations. Priority: exit_radius > M_exit > area_ratio.
*Completed 2026-02-07*

## Dimensional I/O
Config accepts `throat_radius` with units (mm/cm/m/in/ft). CSV exports include
dimensional columns (x_mm, y_mm) alongside normalized. Solver internals unchanged.
*Completed 2026-02-07*

## Shape delta plot
`plot_contour_delta()` in plots.py shows dy(x) between two contours in two
panels: true-scale (aspect='equal') and exaggerated (auto-scaled y). CLI
generates delta plots for every pair in comparison configs.
*Completed 2026-02-07*

## Data exports (CSV, JSON, exit plane)
CLI writes `{name}_contour.csv`, `{name}_exit_plane.csv` (for MLN/TIC), and
`summary.json` with all performance metrics. Dimensional columns included when
throat_radius is set.
*Completed 2026-02-07*

## Exit plane plot deduplication
Fixed `get_exit_plane` to deduplicate points at similar y-positions, and moved
interior mesh points from 0.97 to 0.80 of exit to avoid tolerance overlap.
Exit Mach/angle plots now clean for all Mach numbers.
*Completed 2026-02-07*

## Sivells Phase 2: perfc.f port
`sivells_perfc()` — upstream wall contour via MOC + mass flow integration.
Validated to x~1e-8, M~5e-8 vs CONTUR Mach 4 output. 15 tests.
*Completed 2026-02-07*

## Sivells Phase 1: axial.f port
`sivells_axial()` — transonic throat + axial Mach distribution.
Validated to machine precision vs CONTUR. 32 tests.
*Completed 2026-02-06*

## Sivells Phase 0: CONTUR equivalence validation
29 tests comparing gas functions, interior point, throat characteristic
against CONTUR Fortran output.
*Completed 2026-02-06*

## MLN design_mln rewrite
Replaced broken MOC wall march with direct mesh construction (analytical wall
+ quasi-1D flow properties). All 15 physics invariant tests pass. Cf matches
1D ideal to 4 decimal places for M=2.0 through M=4.0.
*Completed 2026-02-05*

## Rao contour fixes
Extended rao_angles table, added R_u scaling, monotonicity filter. AR 2.0-4.0
all produce valid contours.
*Completed 2026-02-04*

## Core modules
gas.py, kernel.py, moc.py (unit processes), contours.py (conical, Rao, MLN,
TIC), analysis.py (Cf integration), config.py (YAML), cli.py, plots.py,
web viewer (Pyodide). 273 tests, all passing.
*Completed through 2026-02-05*
