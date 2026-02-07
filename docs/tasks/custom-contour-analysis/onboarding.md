# Custom Contour Performance Analysis

## Problem

`type: custom` loads a CSV contour and plots it, but computes no performance.
The original spec asks to take "an arbitrary contour and compare that" — which
means it needs Cf, exit Mach distribution, etc., on equal footing with the
designed nozzles.

## Current state

`analyze_contour()` in moc.py exists but is numerically unstable near-sonic.
The Hall kernel gives M~1.02 at small x, Mach angles near 90 deg, and the
MOC march takes tiny steps that accumulate error.

## Options

### A: Fix analyze_contour startup (Recommended)
Use Sauer's transonic approximation or Sivells' throat characteristic to
provide a stable initial data line further downstream (M~1.1-1.2). March from
there. This is what CONTUR and TDK do.

- Pro: General-purpose, works for any contour shape
- Con: More work, need to handle arbitrary throat curvature from the CSV
- Effort: Medium

### B: Quasi-1D fallback
For contours that are "close enough" to 1D (small wall angles), compute M(x)
from the local area ratio A(x)/A* = y(x)^2 and integrate Cf from that.

- Pro: Simple, stable, no MOC needed
- Con: Ignores 2D effects (flow angle, non-uniform exit), so comparison with
  MOC-designed nozzles is apples-to-oranges
- Effort: Low

### C: Both
Use quasi-1D as the always-available baseline. Offer MOC analysis as an
opt-in when the user wants the full picture.

## Recommendation

Option C. Quasi-1D is easy and useful now. MOC analysis is the right long-term
answer but doesn't need to block custom contour comparison.

## Files to modify

- `nozzle/analysis.py` — new `quasi_1d_performance(x_wall, y_wall, gamma)`
- `nozzle/cli.py` — call it for custom contours
- `nozzle/moc.py` — fix `analyze_contour` startup (longer term)

## Tests

- Quasi-1D on a conical contour matches analytical lambda to ~1%
- Quasi-1D on an MLN contour matches 1D ideal Cf closely
- Custom CSV gets performance numbers in CLI output
