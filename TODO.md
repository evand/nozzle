# TODO

When a task is finished, move it to `docs/COMPLETE.md` (most recent at top).
Tasks that need more than a couple lines of context get a folder under
`docs/tasks/` with an `onboarding.md`.

---

## Data I/O

### Add dimensional I/O to config and output
Config accepts `throat_radius` with units (mm/in). Outputs include dimensional
coordinates alongside normalized. See `docs/tasks/dimensional-io/onboarding.md`.

### Export contour CSV from CLI
`nozzle run` writes `{name}_contour.csv` (x, y columns) to output dir for each
config. Include both normalized and dimensional columns if throat_radius given.

### Export performance JSON from CLI
`nozzle run` writes `summary.json` with Cf, efficiency, M_mean, theta_max, etc.
for every config. Machine-readable companion to the console table.

### Export exit plane CSV
`nozzle run` writes `{name}_exit_plane.csv` (y, M, theta_deg) for MLN/TIC types.

## Plots & Visualization

### Shape delta plot
New plot type: difference between two contours (dy vs x). Show both to-scale and
exaggerated (auto-scaled) views side by side. See `docs/tasks/delta-plot/onboarding.md`.

### Tolerance band overlay
Overlay multiple CSV contours on one plot with shaded band between outer limits.
Useful for manufacturing tolerance checks.

### Web viewer: export buttons
Add "Download CSV" and "Download PNG" buttons to the web interface so users can
save results without the CLI.

## Analysis

### Custom contour performance via MOC analysis
Wire `analyze_contour` into the `type: custom` pipeline so CSV contours get Cf,
exit M distribution, etc. Requires stabilizing the near-sonic startup.
See `docs/tasks/custom-contour-analysis/onboarding.md`.

### Input flexibility: M_exit / exit_diameter / area_ratio interchangeable
Config should accept any one of `M_exit`, `exit_diameter`, or `area_ratio` and
compute the others. Currently only `area_ratio` + optional `M_exit`.

## Sivells Integration

### Wire Sivells as a contour type
Add `type: sivells` to config/CLI. Wrapper in contours.py calls `sivells_axial`
+ `sivells_perfc`, returns (x, y) contour.

### Sivells axisymmetric validation
Current Mach 4 test is planar (ie=0). Test with ie=1 for the real rocket case.

### Sivells downstream contour
Port ip!=0 branch of perfc.f for nozzles extending past the inflection point.

## Polish

### Structured console output
Print a proper aligned table (not ad-hoc print statements) for multi-config runs.

### Example configs for common use cases
Add examples: dimensional input, custom CSV comparison, tolerance band check,
high-Mach design (M=3-4).

### README.md for the repo
User-facing README with install, quick start, example output screenshots.
