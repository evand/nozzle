# Nozzle — MOC Rocket Nozzle Design Tool

## What this is
Method of Characteristics solver for axisymmetric supersonic nozzle design.
Designs optimized contours (MLN, Rao, TIC), generates Rao parabolic approximations,
and compares them — all from YAML config files.

## Multi-session workflow
This project is developed across many sessions. Efficiency across sessions is a priority.

### Reference-first development
Before implementing any algorithm:
1. **Find the published algorithm** in the references below or via web search
2. **Document it** in `docs/algorithms/` as a concise summary (pseudocode, equations, key constraints)
3. **Write physics-based tests** that define correct behavior before writing implementation
4. **Then implement** against those tests

Do not reinvent well-studied algorithms. MOC nozzle design is a mature field with
extensive published literature from the 1950s–1970s.

### Documentation for continuity
- `docs/algorithms/` — Algorithm summaries in markdown (pseudocode, equations, validation targets)
- `docs/references/` — Notes on key reference texts and where to find specific algorithms
- `MEMORY.md` — Cross-session learnings, mistakes to avoid, validated results
- This file (`CLAUDE.md`) — Architecture, conventions, and workflow guidance

### Current status
**Working**: gas.py, kernel.py, unit processes, conical contours, Rao parabolic contours, analysis
**Broken**: `design_mln` wall extraction — interior mesh marches correctly but wall contour
extraction diverges. Root cause verified from Sivells AEDC-TR-78-63 and Guentert & Neumann
TR-R-33: published codes use **mass flow integration** along characteristics to find the wall,
NOT a "design wall point" unit process. See `docs/algorithms/axisymmetric_mln.md` for the
verified algorithm and implementation strategy. Physics invariant tests in
`tests/test_robustness.py::TestMLNPhysicsInvariants` define what "fixed" looks like
(10 intentional failures).

**Reference PDFs** available in `docs/references/pdfs/`:
- `sivells_AEDC-TR-78-63.pdf` — Primary reference, complete Fortran listing
- `guentert_neumann_TR-R-33.pdf` — Same approach, simpler presentation
- `beckwith_moore_NACA-TN-3322.pdf`, `NASA_SP-8120_nozzles.pdf`

## Architecture
- **Coordinates**: Throat at x=0, y=axis distance. Solver works in r_throat-normalized coords.
- **Mesh**: Flat list of `CharPoint` with index-based connectivity.
- **Units**: `unyt` at config boundary only. Solver internals are raw numpy in SI floats.
- **MOC**: Predictor-corrector by default (Zucrow & Hoffman standard).
- **Axis singularity**: L'Hopital's rule for sin(θ)/y → dθ/dy.

## Key modules
- `gas.py` — Isentropic + Prandtl-Meyer relations (Anderson MCF, NACA 1135)
- `kernel.py` — Transonic initial data line (Hall 1962)
- `moc.py` — Core MOC: unit processes, CharMesh, design/analysis modes
- `contours.py` — Nozzle generators: MLN, Rao, TIC, conical, parabolic
- `analysis.py` — Performance: Cf, Isp efficiency, exit uniformity
- `plots.py` — Visualization
- `config.py` — YAML config loading with inheritance
- `cli.py` — CLI entry point

## Key references
See `docs/references/README.md` for detailed index with page/equation numbers.
- **Sivells, AEDC-TR-78-63, 1978** — Primary reference for axisymmetric nozzle design (local PDF)
  - Complete Fortran listing, Mach 4 worked example, mass flow wall technique
- **Guentert & Neumann, NASA TR-R-33, 1959** — Same approach, simpler (local PDF)
- Zucrow & Hoffman, *Gas Dynamics* Vol. 2, Ch. 11–12, Wiley 1977 (NOT locally available)
- Anderson, *Modern Compressible Flow*, 3rd ed., Ch. 11 — 2D planar MLN overview
- Rao, *Jet Propulsion* 1958; *ARS Journal* 1960
- Hall, *QJMAM* 1962 — Transonic kernel for initial data line
- Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed.
- NACA 1135, 1953 — Isentropic relations tables

## Running
```bash
pip install -e .
nozzle run examples/configs/rao_vs_mln.yaml
pytest tests/
```

## Test structure
- `tests/test_gas.py` — Gas dynamics relations vs Anderson tables
- `tests/test_kernel.py` — Hall kernel vs known properties
- `tests/test_moc.py` — Unit processes, mesh construction, design/analysis modes
- `tests/test_contours.py` — Contour generators, Rao angles, geometry
- `tests/test_analysis.py` — Performance integration, Cf validation
- `tests/test_robustness.py` — Property tests: high-Mach, low-AR, physics invariants
  - `TestMLNPhysicsInvariants` — 15 tests defining correct MLN behavior (10 currently fail)
  - `TestMLNHighMach` — M=3.0–7.0 completion and monotonicity
  - `TestRaoLowAR` — AR=2.0–4.0 monotonicity and geometry
  - `TestContourProperties` — Cross-type throat/exit checks

## Validation hierarchy
Fancy results validate against simple ones:
- gas.py vs Anderson tables (exact)
- Unit processes at large y vs 2D planar MOC (exact)
- MLN Cf vs 1D ideal Cf (< 0.1%)
- MOC conical vs analytical λ (< 1%)
- Rao parabolic angles vs Sutton tables (±2°)

## Known issues
- `design_mln` wall extraction is broken (see Current status above)
- `analyze_contour` is numerically unstable near-sonic (x_start must be ≥ 0.1–0.3)
- `np.trapz` removed in numpy 2.0 — use `np.trapezoid` (handled via getattr fallback)
