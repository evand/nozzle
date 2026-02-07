# Dimensional I/O

## Problem

Everything internal is r*-normalized (throat radius = 1). Users work in real
units (mm, inches). There's no way to say "30mm throat, 120mm exit" and get
back dimensional coordinates.

## Design

**Config additions:**
```yaml
nozzle:
  throat_radius: 15 mm    # or: 0.59 in
  # exit can be specified as any one of:
  area_ratio: 10           # A_exit / A_throat
  exit_radius: 47.4 mm     # overrides area_ratio
  M_exit: 3.0              # overrides area_ratio via isentropic
```

**Implementation:**
- `config.py:build_nozzle_spec()` parses `throat_radius` with unyt, converts
  to SI (meters), stores as `r_throat_m`
- `config.py` resolves exit condition priority: `exit_radius` > `M_exit` >
  `area_ratio` (any one determines the others)
- Solver runs in normalized coords as today (no changes to moc.py, gas.py, etc.)
- Output functions multiply by `r_throat_m` to produce dimensional columns
- CSV export: columns `x_norm, y_norm, x_mm, y_mm` (or whatever unit was given)

**Key constraint:** unyt stays at the config boundary. Solver internals remain
pure numpy in normalized coordinates. This is the existing architecture and it
works well.

## Files to modify

- `nozzle/config.py` — parse throat_radius, resolve exit conditions
- `nozzle/cli.py` — pass dimensional info through to output functions
- New: output formatting helpers (maybe in `analysis.py` or a new `output.py`)

## Tests

- Config with throat_radius parses correctly
- exit_radius overrides area_ratio
- M_exit computes correct area_ratio
- CSV output has dimensional columns when throat_radius given
- CSV output is normalized-only when throat_radius omitted
