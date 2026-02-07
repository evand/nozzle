# Port Sivells CONTUR to Python

## Goal

Replace the `design_mln` hack (analytical wall + quasi-1D fill) with a real
MOC solver ported from Sivells' CONTUR code. The validated F90 port at
`docs/references/external_codes/contur/src/` is the line-by-line reference.

## Validation Target

Reproduce the Mach 4 axisymmetric test case from Sivells AEDC-TR-78-63.
Expected output is in `docs/references/external_codes/contur/docs/output.txt`.
Input cards are in `docs/references/external_codes/contur/src/input.txt`.

## CONTUR Program Structure (3,209 lines F90)

```
main.f ─── axial.f ─── trans.f      Phase 1: Transonic + axial Mach distribution
       │              ├── cubic.f
       │              ├── fmv.f
       │              ├── toric.f
       │              ├── conic.f
       │              └── sorce.f
       │
       ├── perfc.f ─── ofeld.f      Phase 2: MOC march + mass flow wall finding
       │              ├── fmv.f
       │              ├── scond.f
       │              ├── twixt.f
       │              └── neo.f
       │
       ├── bound.f                   Phase 3: Boundary layer correction (optional)
       │
       └── splind.f / xyz.f          Output: spline interpolation for smooth contour
```

## Mapping to Our Existing Code

| CONTUR subroutine | Our code | Status | Action |
|-------------------|----------|--------|--------|
| `fmv` | `gas.py:mach_from_prandtl_meyer` | **Done** | Validate equivalence |
| `ofeld` | `moc.py:interior_point` | **Done** | Validate equivalence |
| `cubic` | — | **Missing** | Port (61 lines, Cardano solver) |
| `trans` | `kernel.py:hall_initial_line` | **Different method** | Port Sivells' version alongside Hall |
| `axial` | — | **Missing** | Port (813 lines, core algorithm) |
| `perfc` | `moc.py:analyze_contour` (broken) | **Must replace** | Port (634 lines, core algorithm) |
| `splind`/`xyz` | — | **Missing** | Port or use `scipy.interpolate` |
| `scond` | — | **Missing** | Port (25 lines, parabolic deriv) |
| `twixt` | — | **Missing** | Port (28 lines, Lagrange interp) |
| `neo` | — | **Missing** | Port (144 lines, contour smoothing) |
| `toric` | — | **Missing** | Port (20 lines) |
| `conic` | — | **Missing** | Port (24 lines) |
| `sorce` | — | **Missing** | Port (25 lines) |
| `bound` | — | **Missing** | Defer (571 lines, optional) |

## Phases

### Phase 0: Validate Existing Equivalences

Before porting anything, confirm our existing functions match CONTUR's.

**Tasks:**
- [ ] Compare `gas.py:mach_from_prandtl_meyer` vs `fmv.f` — same algorithm?
      Same input/output convention? Test with values from Mach 4 case.
- [ ] Compare `moc.py:interior_point` vs `ofeld.f` — same predictor-corrector?
      Same source term formulation? Same iteration count?
- [ ] Compare `kernel.py:hall_initial_line` vs `trans.f` — different methods
      (Hall vs Sivells), but do they produce similar output for same inputs?
- [ ] Extract the Mach 4 test case gas constants (γ=1.4, R=1716.563) and
      geometry (inflection angle 8.67°, RC=6) from input.txt for later use.

**Validation:** Write `tests/test_contur_equivalence.py` with known values
from the CONTUR Mach 4 output.

### Phase 1: Port Utility Functions

Small, self-contained functions that other subroutines depend on.

**Tasks:**
- [ ] Port `cubic.f` → `nozzle/sivells.py:cubic_solve` (Cardano's formula, 61 lines)
- [ ] Port `toric.f` → `nozzle/sivells.py:throat_curvature` (20 lines)
- [ ] Port `conic.f` → `nozzle/sivells.py:conic_derivatives` (24 lines)
- [ ] Port `sorce.f` → `nozzle/sivells.py:source_derivatives` (25 lines)
- [ ] Port `scond.f` → `nozzle/sivells.py:parabolic_derivative` (25 lines)
- [ ] Port `twixt.f` → `nozzle/sivells.py:lagrange_interp` (28 lines)
- [ ] Decide on `splind`/`xyz`: port or use `scipy.interpolate.CubicSpline`

**Validation:** Unit tests for each function against known values from
CONTUR output. `cubic_solve` can be tested against analytical roots.

### Phase 2: Port `trans.f` — Transonic Throat Solution

The throat characteristic computation. This replaces/supplements our
`kernel.py:hall_initial_line`.

**Tasks:**
- [ ] Read `trans.f` (222 lines) line by line against Z&H Ch. 12 and
      Sivells Appendix A equations
- [ ] Port to `nozzle/sivells.py:sivells_throat_characteristic`
- [ ] Returns 51 points on the throat characteristic (fc array)
- [ ] Compare output with Hall kernel at same conditions — document differences

**Validation:** Throat characteristic from Mach 4 test case matches
CONTUR output to 4+ significant figures.

### Phase 3: Port `axial.f` — Centerline Mach Distribution

The largest subroutine (813 lines). Computes the upstream and downstream
Mach/velocity distribution along the nozzle axis.

**Tasks:**
- [ ] Study `axial.f` structure: throat iteration (lines 11-267),
      upstream distribution (lines 327-382), downstream distribution
- [ ] Understand the input parameters: inflection angle, RC, polynomial
      order, downstream Mach number
- [ ] Port to `nozzle/sivells.py:axial_distribution`
- [ ] Input: γ, inflection_angle, R_curvature, M_exit
- [ ] Output: axial M(x), velocity derivatives, throat characteristic data

**Validation:** Axial Mach distribution from Mach 4 test case matches
CONTUR output. Intermediate values (c coefficients, iteration residuals)
should be checked.

**Key reference:** Sivells AEDC-TR-78-63 Section 3.0, Eqs. 29-39.

### Phase 4: Port `perfc.f` — MOC Contour Generation

The core MOC algorithm (634 lines). This is what actually computes the
nozzle wall contour via characteristic march + mass flow integration.

**Tasks:**
- [ ] Study `perfc.f` structure:
      - Initialization from throat characteristic (lines 100-169)
      - Main march loop (lines 174-451)
      - Mass flow integration (lines 259-303)
      - Wall finding by mass flow (lines 304-350)
      - Contour output (lines 524-595)
- [ ] Map `ofeld` calls to our `interior_point` — verify interface compatibility
- [ ] Port to `nozzle/sivells.py:sivells_contour` or rewrite `design_mln`
- [ ] Input: axial distribution from Phase 3, geometry parameters
- [ ] Output: wall contour (x, y, slope, Mach), characteristic mesh

**Validation:** Wall contour coordinates from Mach 4 test case match
CONTUR output. This is the primary validation target.

**Key reference:** Sivells Section 4.0; G&N Eq. 22 for mass flow;
Moger & Ramsay for the mass flow technique.

### Phase 5: Integration and Cleanup

Wire the ported code into `design_mln` and verify all existing tests pass.

**Tasks:**
- [ ] Rewrite `design_mln` to call the ported Sivells algorithm
- [ ] Ensure `CharMesh` output is compatible with existing test helpers
      (`_real_wall_points`, `_real_axis_points`, `_real_mesh`)
- [ ] All 197 existing tests pass
- [ ] Add new tests: `test_sivells_mach4.py` validating against published output
- [ ] Port `neo.f` contour smoothing if needed for contour quality
- [ ] Run `python web/build.py`

### Phase 6 (Future): Boundary Layer and Extensions

- [ ] Port `bound.f` for boundary layer correction
- [ ] Support variable γ (G&N approach)
- [ ] Support planar nozzles (ie=0 flag in CONTUR)
- [ ] Lower Mach number designs (M=2-3, noted as harder by Rona/Korte)

## Key References for Each Phase

| Phase | Primary reference | Pages to read |
|-------|-------------------|---------------|
| 0 | CONTUR output.txt | All |
| 1 | CONTUR F90 source | utility .f files |
| 2 | Sivells Appendix A; Z&H Ch. 12 | Eqs. A-1 to A-40 |
| 3 | Sivells Section 3.0 | Eqs. 29-39, p.14-25 |
| 4 | Sivells Section 4.0; G&N Eq. 22 | p.25-38 |
| 5 | Our test suite | tests/test_robustness.py |

## File Organization

New code goes in `nozzle/sivells.py` — keeps the port separate from the
existing `moc.py` until it's validated and ready to replace it.

## What NOT To Do

- Don't try to make `analyze_contour` work by patching it. It's a different
  algorithm (IDL-based march) that fundamentally fails near the wall.
- Don't guess at the algorithm from textbook descriptions. Read the F90 code
  line by line. The devil is in the iteration details, convergence checks,
  and edge cases that Sivells handled in the 1970s.
- Don't port `bound.f` until the inviscid contour is validated. BL correction
  is a separate concern.
