# Shape Delta Plot

## Problem

The original spec asks to "display the delta in shape (to scale and
exaggerated)" between contours. Currently we overlay contours but don't show
the actual difference.

## Design

Two-panel plot:
1. **To scale**: dy(x) = y_A(x) - y_B(x) plotted with the same x/y scale as
   the contour plot. This shows how the difference looks physically.
2. **Exaggerated**: Same dy(x) but y-axis auto-scaled to show detail. This
   reveals subtle shape differences that are invisible at true scale.

**Implementation:**
- Interpolate both contours to a common x grid (union of x arrays, or
  linspace over shared domain)
- Compute dy = y_A(x) - y_B(x) via np.interp
- Handle different x extents: shade or mark the region where only one contour
  exists
- Color: positive dy (A is wider) in one color, negative in another

**Where it fits:**
- `plots.py`: new function `plot_contour_delta(x_A, y_A, x_B, y_B, ...)`
- CLI: auto-generate for each comparison pair in the config
- Web: add as a third panel when comparing two contours

## Tolerance band variant

When 3+ contours are compared, show the envelope (min/max at each x) as a
shaded band. This is the manufacturing tolerance use case.

## Files to modify

- `nozzle/plots.py` — new `plot_contour_delta()` function
- `nozzle/cli.py` — generate delta plots for comparison pairs
- `web/index.html` — add delta panel (optional, can defer)
