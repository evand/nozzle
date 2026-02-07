# Completed Tasks

Move items here from TODO.md when done. Most recent at top.

---

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
