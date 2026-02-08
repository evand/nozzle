# TODO

When a task is finished, move it to `docs/COMPLETE.md` (most recent at top).
Tasks that need more than a couple lines of context get a folder under
`docs/tasks/` with an `onboarding.md`.

---

## Analysis

### TIC performance at truncated exit
moc_performance currently evaluates at the full MLN exit plane even for TIC.
Need to evaluate at the truncated x cross-section. Part of broader comparisons
task (comparing nozzles with unequal exit ratios, etc).

### Custom contour performance via full MOC analysis
`quasi_1d_performance` works now for simple estimates. For full 2D analysis,
stabilize `analyze_contour` near-sonic startup (Sauer transonic init).
See `docs/tasks/custom-contour-analysis/onboarding.md`.

## Plots & Visualization

### Tolerance band overlay
Overlay multiple CSV contours on one plot with shaded band between outer limits.
Useful for manufacturing tolerance checks.

## Polish

### Example configs for common use cases
Add examples: custom CSV comparison, tolerance band check, high-Mach design.

### README.md for the repo
User-facing README with install, quick start, example output screenshots.
