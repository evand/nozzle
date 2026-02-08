# Validation Matrix

Maps every module to its reference sources, tolerances, and test files.
Identifies gaps where no external reference exists.

**355 tests total** — 0 failures as of 2026-02-08.

## Summary by Reference Source

| Source | Tests | Description |
|--------|-------|-------------|
| CONTUR F90 (Sivells AEDC-TR-78-63) | ~122 | Fortran output at machine precision |
| Anderson MCF 3rd ed., Tables A.1/A.5 | ~25 | Isentropic relations, Prandtl-Meyer |
| Sutton & Biblarz 9th ed., Tables 3-3/3-4 | ~12 | Rao angles, conical lambda, Cf |
| Analytical formulas | ~10 | Geometry, lambda=(1+cos theta)/2 |
| Self-consistency / physics constraints | ~186 | Monotonicity, roundtrips, ordering |

## Module-by-Module Detail

### gas.py — Isentropic & Prandtl-Meyer Relations

| Function | Reference | Tolerance | Test File |
|----------|-----------|-----------|-----------|
| `pressure_ratio` | Anderson MCF Table A.1, NACA 1135 | rel=1e-3 | test_gas.py |
| `temperature_ratio` | Anderson MCF Table A.1, NACA 1135 | rel=1e-3 | test_gas.py |
| `density_ratio` | Anderson MCF Table A.1, NACA 1135 | rel=1e-3 | test_gas.py |
| `area_mach_ratio` | Anderson MCF Table A.1, NACA 1135 | rel=1e-3 | test_gas.py |
| `mach_from_area_ratio` | Roundtrip vs `area_mach_ratio` | rel=1e-10 | test_gas.py |
| `mach_angle` | Analytical: arcsin(1/M) | rel=1e-10 | test_gas.py |
| `prandtl_meyer` | Anderson MCF Table A.5 | abs=0.05 deg | test_gas.py |
| `mach_from_prandtl_meyer` | Roundtrip + CONTUR F90 | abs=1e-4 | test_gas.py, test_contur_equivalence.py |
| `thrust_coefficient_ideal` | Sutton Table 3-3 | range check | test_gas.py |

**Status: Well-validated.** Every function has a published reference.

### kernel.py — Hall Transonic Initial Data Line

| Function | Reference | Tolerance | Test File |
|----------|-----------|-----------|-----------|
| `hall_initial_line` | Physics constraints only | constraints | test_kernel.py |

Constraints tested: M>1 everywhere, theta=0 on axis, M monotonic axis-to-wall,
theta monotonic, wall position matches parabolic throat, constant x.

**Gap: No published reference output.** Hall (1962) QJMAM paper has the theory
but no tabulated output. Kennedy's F90 code (DTIC_ADA578559) or Guentert &
Neumann TR-R-33 could provide reference values.

### moc.py — Unit Processes (Interior, Axis, Wall Points)

| Function | Reference | Tolerance | Test File |
|----------|-----------|-----------|-----------|
| `interior_point` | 2D planar limit: K+/K- conserved | abs=0.01 rad | test_unit_processes.py, test_contur_equivalence.py |
| `axis_point` | Symmetry: y=0, theta=0 | abs=1e-12 | test_unit_processes.py |
| `wall_point` | Boundary: theta=arctan(dy/dx) | rel=1e-10 | test_unit_processes.py |

**Moderate.** Planar Riemann invariant test validates the core algorithm.
No axisymmetric reference output for unit processes in isolation (but see
Sivells tests below for full axisymmetric validation).

### moc.py — design_mln (MLN Mesh Construction)

| Property | Reference | Tolerance | Test File |
|----------|-----------|-----------|-----------|
| theta_max = nu(M)/2 | Anderson MCF Eq. 11.33 | abs=0.1 deg | test_mln.py |
| Exit A/A* matches 1D | Isentropic area-Mach | constraint | test_mln.py, test_robustness.py |
| Cf = Cf_ideal | 1D ideal (Sutton Eq. 3-30) | rel=0.05 | test_robustness.py |
| Axis M monotonic | Zucrow & Hoffman Ch. 11 | tol=1e-6 | test_robustness.py |
| Mesh convergence | n=15 vs n=30 | rel=0.10 | test_robustness.py |

**Moderate.** Physics invariants are well-tested. The implementation uses
quasi-1D flow properties by construction, so Cf=Cf_ideal is somewhat circular.
True validation would require comparison to a published MOC mesh (e.g., Anderson
Example 11.2 or Sivells).

### contours.py — Conical Nozzle

| Property | Reference | Tolerance | Test File |
|----------|-----------|-----------|-----------|
| Geometry | Analytical (straight line) | exact | test_conical.py |
| lambda = (1+cos theta)/2 | Sutton & Biblarz Eq. 3-32 | abs=1e-4 | test_conical.py |
| lambda(15 deg) = 0.9830 | Sutton Table 3-3 | abs=1e-4 | test_conical.py, test_performance.py |

**Status: Well-validated.** Analytical — no numerical algorithm to go wrong.

### contours.py — Rao Parabolic Nozzle

| Property | Reference | Tolerance | Test File |
|----------|-----------|-----------|-----------|
| theta_n, theta_e (AR=10, 80%) | Sutton & Biblarz Table 3-4 | abs=2.0 deg | test_rao.py |
| theta_n, theta_e (AR=4, 60%) | Sutton & Biblarz Table 3-4 | abs=2.0 deg | test_rao.py |
| theta_n > theta_e | Rao (1960) | constraint | test_rao.py |
| Cf ordering: conical < rao < ideal | Sutton §3.4 | constraint | test_rao.py, test_performance.py |
| Exit radius = sqrt(AR) | Geometry | rel=0.05 | test_rao.py |
| Low-AR (2.0-4.0) monotonicity | Robustness | constraint | test_robustness.py |

**Status: Good.** Angles validated vs Sutton tables at +/-2 deg tolerance
(table interpolation limits accuracy). Performance ordering correct.

### contours.py — TIC (Truncated Ideal Contour)

| Property | Reference | Tolerance | Test File |
|----------|-----------|-----------|-----------|
| TIC(100%) = MLN | Definition | array_equal | test_tic.py |
| TIC shorter than MLN | Design intent | constraint | test_tic.py |
| Cf(TIC) < Cf(MLN) | Divergence loss | constraint | test_tic.py |
| Cf increases with fraction | Physics: smaller exit angle | constraint | test_tic.py |
| Cf(100%) ~ Cf(MLN) | Limit behavior | rel=0.01 | test_tic.py |

**Gap: No published TIC performance data.** TIC is just a truncated MLN,
so correctness follows from MLN correctness + geometry. Could validate
against NASA SP-8120 or Rao (1960) TIC performance curves if available.

### sivells.py — CONTUR Port (Throat + Upstream + Downstream)

| Component | Reference | Tolerance | Test File |
|-----------|-----------|-----------|-----------|
| Gas constants (gamma=1.4) | CONTUR main.f | abs=1e-10 | test_sivells.py |
| FG constants (ie=0, ie=1) | CONTUR axial.f | abs=1e-6 | test_sivells.py |
| Cubic solver | Cardano + CONTUR output | abs=1e-5 | test_sivells.py |
| Conic derivatives | Area-Mach at M=2, M=4 | abs=1e-8 | test_sivells.py |
| Lagrange interpolation | Exact for degree 3 | abs=1e-8 | test_sivells.py |
| `ofeld` (interior point) | 2D planar K+/K- exact | abs=1e-10 | test_sivells.py |
| Throat characteristic (planar) | CONTUR F90 Mach 4 output | abs=1e-5 | test_sivells.py |
| `sivells_axial` (planar) | CONTUR F90: wo, tk, yo, polynomials | abs=1e-5 | test_sivells.py |
| `sivells_perfc` (planar) | CONTUR F90: 41 wall points (x,y,M,theta) | x: 1e-6, y: 1e-4, M: 1e-6, theta: 0.01 deg | test_sivells.py |
| `sivells_axial` (axi, ie=1) | CONTUR F90 axi output | abs=1e-5 | test_sivells_axi.py |
| `sivells_perfc` (axi, ie=1) | CONTUR F90 axi output | x: 1e-6, y: 1e-4, theta: 0.02 deg | test_sivells_axi.py |
| `sivells_axial_downstream` | CONTUR F90 downstream output | abs=1e-5 | test_sivells_contour.py |
| `sivells_perfc_downstream` | CONTUR F90 downstream output | x: 1e-4, y: 1e-3, M: 1e-3 | test_sivells_contour.py |
| `sivells_nozzle` wrapper | Coordinate transform + geometry | constraints | test_sivells_contour.py |

**Status: Excellent.** 122 tests against CONTUR F90 output. Covers planar,
axisymmetric, upstream, and downstream. Machine-precision agreement on
most quantities.

### analysis.py — Performance Metrics

| Function | Reference | Tolerance | Test File |
|----------|-----------|-----------|-----------|
| `conical_performance` | Sutton lambda formula | abs=1e-4 | test_conical.py, test_performance.py |
| `rao_performance` | Sutton Table 3-4 + lambda approx | abs=2 deg (angles) | test_rao.py |
| `moc_performance` | MLN Cf vs 1D ideal | rel=0.05 | test_robustness.py |
| `quasi_1d_performance` | Lambda divergence correction | rel=0.01 (TIC limit) | test_tic.py |

**Moderate.** Conical/Rao use well-known formulas. MOC integration has no
independent reference (validated indirectly through Cf ~ Cf_ideal).

### config.py / cli.py — Configuration & I/O

| Feature | Reference | Test File |
|---------|-----------|-----------|
| CSV load/save roundtrip | Self-consistency | test_csv_contour.py |
| Throat normalization | Coordinate convention | test_csv_contour.py |
| Config inheritance (base:) | Functional spec | (example configs) |

**Status: Adequate.** I/O tests are functional, not physics-based.

## Known Gaps (No External Reference)

| Area | What's Missing | Potential Reference |
|------|----------------|---------------------|
| Hall kernel output | Tabulated M, theta at specific (x, R_wall) | Kennedy F90, Guentert & Neumann TR-R-33 |
| MOC unit processes (axisymmetric) | Reference output for individual point computations | Zucrow & Hoffman worked examples |
| TIC performance curves | Published Cf vs truncation fraction | NASA SP-8120, Rao (1960) |
| MLN mesh (non-quasi-1D) | Independent MOC mesh at M=2 or 2.4 | Anderson Example 11.2, AeroMOC |
| `moc_performance` integration | Independent Cf from exit plane integration | TDK manual (JANNAF standard) |
