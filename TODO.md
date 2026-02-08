# TODO

When a task is finished, move it to `docs/COMPLETE.md` (most recent at top).
Tasks that need more than a couple lines of context get a folder under
`docs/tasks/` with an `onboarding.md`.

---

## Analysis

### Custom contour performance via full MOC analysis
`quasi_1d_performance` works now for simple estimates. For full 2D analysis,
stabilize `analyze_contour` near-sonic startup (Sauer transonic init).
See `docs/tasks/custom-contour-analysis/onboarding.md`.

## Polish

### Example configs for common use cases
Add examples for: custom CSV comparison, tolerance band check.
(Done: all_types.yaml, tic_comparison.yaml, high_mach.yaml)

### README.md for the repo — DONE
User-facing README with install, quick start, example output, validation, project structure.

### Example config for tolerance band check
Add an example config comparing two similar contours with tolerance analysis.
