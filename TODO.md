# TODO

When a task is finished, move it to `docs/COMPLETE.md` (most recent at top).
Tasks that need more than a couple lines of context get a folder under
`docs/tasks/` with an `onboarding.md`.

---

## Plots & Visualization

### Tolerance band overlay
Overlay multiple CSV contours on one plot with shaded band between outer limits.
Useful for manufacturing tolerance checks.

### Web viewer: export buttons
Add "Download CSV" and "Download PNG" buttons to the web interface so users can
save results without the CLI.

## Analysis

### Custom contour performance via full MOC analysis
`quasi_1d_performance` works now for simple estimates. For full 2D analysis,
stabilize `analyze_contour` near-sonic startup (Sauer transonic init).
See `docs/tasks/custom-contour-analysis/onboarding.md`.

## Sivells Integration

### Wire Sivells as a contour type
Add `type: sivells` to config/CLI. Wrapper in contours.py calls `sivells_axial`
+ `sivells_perfc`, returns (x, y) contour.

### Sivells axisymmetric validation
Current Mach 4 test is planar (ie=0). Test with ie=1 for the real rocket case.

### Sivells downstream contour
Port ip!=0 branch of perfc.f for nozzles extending past the inflection point.

## Polish

### Example configs for common use cases
Add examples: custom CSV comparison, tolerance band check, high-Mach design.

### README.md for the repo
User-facing README with install, quick start, example output screenshots.
(Current README is a good start, needs screenshots and more examples.)
