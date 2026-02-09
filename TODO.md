# TODO

When a task is finished, move it to `docs/COMPLETE.md` (most recent at top).
Tasks that need more than a couple lines of context get a folder under
`docs/tasks/` with an `onboarding.md`.

---

## Tier 2 — Real design capability

### Length-constrained optimal nozzle
Given (target length, target AR), find the MLN design Mach M_design > M_exit
such that truncating MLN(M_design) at the target length gives the target AR.
1D root-find (bisect over M_design) wrapping existing `minimum_length_nozzle`.
This answers: "best nozzle I can build in X cm at this area ratio."

### Parametric sweep
Vary one parameter (bell %, truncation fraction, half-angle, design Mach) and
plot Cf vs. nozzle length curves. This is the real trade study tool — shows
the full design space rather than single-point comparisons.

### Constraint-based comparison
Compare nozzles at equal length, equal weight, or equal exit pressure rather
than just equal M_exit. Requires solving for the free parameter (bell fraction,
truncation, half-angle) that hits the constraint. Root-finder around existing
contour generators.

### CAD export
DXF (or IGES) of the wall contour as a revolved profile. Makes the tool
useful for actual manufacturing. Convergent section (Tier 1) should land first
so the export is a complete nozzle.

## Tier 3 — Engineering fidelity

### Profile tolerance sensitivity analysis
Given a tolerance band (±tol on radius), how much does performance vary?
Three levels:
- **Deterministic worst-case**: perturbation maximizing exit divergence angle.
- **Monte Carlo**: random smooth perturbations (sinusoidal modes, bounded
  amplitude), evaluate `quasi_1d_performance` on hundreds of samples, report
  Cf distribution and 95% confidence band.
- **Constrained manufacturing**: C0 continuity, no curvature reversal, within
  band. Realistic for machined/formed nozzles.
Connects to existing `plot_tolerance_band` for visualization.

### Boundary layer correction
Displacement thickness shifts the effective wall inward. NASA SP-8120 and
Rao's original papers have correlations. Without this, manufactured nozzles
underperform predictions. Could be a post-processing correction on the
contour (add δ* to wall radius).

### Off-design performance
Fixed contour at chamber pressures other than design point. The expansion
ratio is fixed but the flow structure changes. Requires analysis-mode MOC
(`analyze_contour`), which is currently unstable near-sonic.

## Analysis (existing)

### Custom contour performance via full MOC analysis
`quasi_1d_performance` works now for simple estimates. For full 2D analysis,
stabilize `analyze_contour` near-sonic startup (Sauer transonic init).
See `docs/tasks/custom-contour-analysis/onboarding.md`.

## Polish

### Example configs
Add examples for: custom CSV comparison, tolerance band check.
(Done: all_types.yaml, tic_comparison.yaml, high_mach.yaml)
