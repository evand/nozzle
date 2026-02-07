# References Index

Key texts for MOC nozzle design, organized by topic.
Local PDFs in `pdfs/` subdirectory. External code repos in `external_codes/`.

## Primary MOC References

### Zucrow & Hoffman — Gas Dynamics Vol. 2 (1977)
**`zucrow_hoffman_vol2_1977.pdf`** — locally available
- **Ch. 11**: 2D planar MOC fundamentals
  - Unit processes (interior, axis, wall)
  - Predictor-corrector formulation
  - Design mode vs analysis mode
  - MLN design for planar flow
- **Ch. 12**: Axisymmetric MOC
  - Source term Q = sin(θ)·sin(μ) / (y·cos(θ±μ))
  - Axis singularity treatment (L'Hopital)
  - Axisymmetric MLN design procedure
  - Worked examples with nozzle coordinates

### Anderson — Modern Compressible Flow, 3rd ed. (2003)
- **Ch. 11 §11.11**: MLN design conditions
  - Along each C- in the expansion fan: K- = θ - ν = const
  - Wall angle from arriving C+: θ_wall determined by compatibility
  - Good conceptual overview, but 2D planar examples only

## Local PDFs (in `pdfs/`)

### Sivells — AEDC-TR-78-63, 1978 — CONTUR program
**`sivells_AEDC-TR-78-63.pdf`** (148 pages)
- Complete Fortran listing (Appendix D, p.61-138)
- Worked Mach 4 axisymmetric nozzle example (Section 7.0, p.38)
- **Key subroutines**:
  - AXIAL: Centerline velocity/Mach distribution as polynomial
  - PERFC: Characteristic network + contour by mass flow integration
  - OFELD: Interior point (standard predictor-corrector)
  - BOUND: Turbulent boundary layer correction
  - TRANS: Transonic throat region (series expansion, Appendix A)
- **Wall determination** (PERFC lines PER 173-221): Mass flow integration
  along each C+ characteristic. Wall = where ∫mass_flow = total_flow.
  From Moger & Ramsay (Ref. 17, AEDC-TDR-64-110, 1964).
- **Wall coordinates** (PERFC lines PER 249-303): Integrate dy/dx = tan(θ_wall)
  from inflection point using cubic integration coefficients.
- Transonic equations: Appendix A (Eqs. A-1 to A-40, planar + axisymmetric)
- Integration factors: Appendix B (cubic integration for uneven spacing)

### Kennedy — DTIC ADA578559 / TR-RDMR-SS-12-11, 2013
**`DTIC_ADA578559_MOC_solver.pdf`** (127 pages)
- MOC Nozzle Flowfield Solver — User's Guide and Input Manual
- U.S. Army AMRDEC, Redstone Arsenal
- **Fortran 90**, analysis mode (given wall → compute flowfield)
- 2nd-order modified Euler predictor-corrector with iteration
- **Transonic**: Sauer's method for sonic line, or user-input starting profile
- **Wall**: Circular arc + quadratic, then mass flow rate determination
- **Boundary layer**: 1/8 power law correction
- Complementary to Sivells: analysis mode vs design mode

### Guentert & Neumann — NASA TR-R-33, 1959
**`guentert_neumann_TR-R-33.pdf`** (20 pages)
- Axisymmetric exhaust nozzle design with variable isentropic exponent
- **Same fundamental approach as Sivells**: centerline pressure distribution + mass flow
- Key equations:
  - Interior point: Eqs. 5-12 (predictor-corrector with source terms)
  - Mass flow across characteristic: Eq. 22: w = 2π∫ρV[cosθ - sin²θ/tan(θ+μ)]·y·dy
  - Vacuum specific impulse: Eq. 23 (integrated along C+ characteristics)
- Pressure distribution options: cosine (Eq. 19), parabolic (Eq. 20), cubic (Eq. 21)
- Important finding (p.7): "extremely difficult to specify flow distribution along
  nozzle axis and obtain solutions yielding short-length nozzles" — characteristics
  coalesce if expansion is too fast
- Computational procedure: Appendix B (pp. 16-17)

### Beckwith & Moore — NACA TN-3322
**`beckwith_moore_NACA-TN-3322.pdf`**
- Accurate nozzle design method
- Tabulated nozzle coordinates for various Mach numbers

### NASA SP-8120 — Liquid Rocket Engine Nozzles
**`NASA_SP-8120_nozzles.pdf`**
- Comprehensive monograph on rocket nozzle design
- Performance data, design guidelines, contour methods

### NASA RP-1104 — Perfect Bell Nozzle Parametric Data, 1983
**`NASA_RP-1104_bell_nozzle.pdf`** (31 pages)
- Tuttle & Blount, Marshall Space Flight Center
- Nozzle contour data for bell nozzles, area ratios up to 6100, γ=1.2
- Optimization curves for maximum Cf within length/area constraints
- Validation data for Rao-type bell nozzles

### Rona & Zavalan — SSRN 4061508, 2022
**`rona_zavalan_SSRN-4061508_contur.pdf`** (12 pages)
- SoftwareX article documenting the CONTUR code reconstruction
- OCR of Sivells' F77 listing, translation to F90
- Verified against Mach 4 test case (exact match to 16th decimal)
- Notes low Mach designs are harder (short radial flow region, per Korte)

### Wang & Jiang — Chinese J. Aeronautics 35(1), 2022
**`wang_jiang_hypersonic_nozzle_review.pdf`** (25 pages)
- Comprehensive review: MOC, graphic design, Sivells method, BL correction, CFD optimization
- Confirms "the Sivells method is still one of the internationally-recognized design
  methods for high-velocity nozzles that feature high accuracy"
- Covers real-gas effects, boundary layer, and limitations of MOC at M > 8

### TDK — Two-Dimensional Kinetic Reference Code, 1986 manual
**`TDK_manual_19860007470.pdf`**
- Nickerson, Coats, Bartz (Software & Engineering Associates)
- JANNAF standard nozzle performance prediction code
- Design + analysis modes, real gas / equilibrium / frozen / finite-rate kinetics
- Modern versions are commercial (SEA Inc.)
- 1986 NASA CR version with documentation is public domain

## External Code Repos (in `external_codes/`)

### aldorona/contur — Sivells F90 port (MIT license)
**`external_codes/contur/`**
- OCR reconstruction of Sivells' original F77 from AEDC-TR-78-63
- Ported to Fortran 90 by Aldo Rona & Florentina-Luiza Zavalan (U. Leicester)
- **Validated** against Sivells' published Mach 4 test case
- `sivells/` — original F77 source files + makefile
- `src/` — F90 port with modules + makefile
- `docs/output.txt` — validated Mach 4 output
- Paper: Rona & Zavalan, SSRN 4061508 (not locally available)
- https://github.com/aldorona/contur

### noahess/conturpy — Python wrapper for CONTUR F90
**`external_codes/conturpy/`**
- Python wrapper around Rona's F90 port
- ConturSettings/ConturResult classes, batch runs, CSV export, plotting
- Precompiled binaries for Windows x86-64 and Apple Silicon
- `conturpy/` — Python wrapper code
- `src/` — F90 source (same as contur)
- https://github.com/noahess/conturpy

### YangYunjia/AeroMOC — Python axisymmetric MOC
**`external_codes/AeroMOC/`**
- Independent Python implementation of axisymmetric MOC
- Maximum thrust (ideal) and minimum length nozzle design
- Non-isentropic flow support, boundary layer correction
- `aeromoc/moc.py` — core MOC solver
- https://github.com/YangYunjia/AeroMOC

### nasa/Three-Dimensional-Nozzle-Design-Code — NASA MOC/STT
**`external_codes/Three-Dimensional-Nozzle-Design-Code/`**
- Official NASA Glenn code (LEW-20180-1)
- 2D MOC grid generation + 3D streamline tracing
- C++ with MFC GUI (Windows-specific)
- `MOC_Grid_BDE/` — 2D MOC nozzle contour generation
- https://github.com/nasa/Three-Dimensional-Nozzle-Design-Code

## Transonic Region

### Hall — QJMAM 15(4), 1962
- Transonic kernel for initial data line
- Avoids the sonic singularity at the throat
- Our `kernel.py` implements this

### Sauer's Method
- Alternative transonic treatment used by Sivells and Kennedy
- Series expansion for sonic line determination
- Implemented in CONTUR (TRANS subroutine) and Kennedy's code

## Nozzle Design Methods

### Sivells — J. Spacecraft and Rockets 7(11), 1970
- "Aerodynamic design of axisymmetric hypersonic wind-tunnel nozzles"
- Complete procedure for axisymmetric MLN including boundary layer correction
- Hypersonic validation data (M=8, 10, 20)
- Uses Sauer's transonic approximation for initial line

### Rao — Jet Propulsion (1958), ARS Journal (1960)
- Thrust-optimized contour via calculus of variations
- Our `contours.py` implements the parabolic approximation
- Angle tables from Sutton & Biblarz

### Moger & Ramsay — AEDC-TDR-64-110, 1964
- "Supersonic Axisymmetric Nozzle Design by Mass Flow Techniques"
- The mass flow technique used by Sivells' PERFC subroutine
- NOT locally available (try DTIC AD-601589)

### Foelsch — J. Aeronautical Sciences 16(3), 1949
- Analytical method for axisymmetric nozzle design
- Precursor to numerical MOC approaches

## Isentropic Relations

### NACA 1135 (1953)
- "Equations, Tables, and Charts for Compressible Flow"
- Our `gas.py` validates against these tables

### Sutton & Biblarz — Rocket Propulsion Elements, 9th ed.
- Rao parabolic nozzle angle tables (Table 3-4)
- Performance comparison: conical vs bell nozzle

## Expected Validation Targets

| Design | M_exit | γ   | Cf (ideal) | Expected Cf/Cf_ideal |
|--------|--------|-----|------------|---------------------|
| MLN    | 2.0    | 1.4 | 1.4234     | 100% (±0.1%)        |
| MLN    | 3.0    | 1.4 | 1.5678     | 100% (±0.1%)        |
| Conical 15° | 2.0 | 1.4 | 1.4234  | 98.3% (= λ)        |
| Rao 80%| 2.0    | 1.4 | 1.4234     | 98.8%               |

## Nozzle Length Reference Data

From Sutton & Biblarz and general MOC literature:

| M_exit | A/A*  | L_conical_15°/r* | L_MLN/r* (approx) |
|--------|-------|-------------------|--------------------|
| 2.0    | 1.688 | 2.6               | ~1.5               |
| 3.0    | 4.235 | 6.4               | ~3.5               |
| 4.0    | 10.72 | 15.8              | ~8                  |
| 5.0    | 25.00 | 36.7              | ~18                 |

L_conical = (√(A/A*) - 1) / tan(15°)
L_MLN ≈ 55-65% of L_conical (varies with M and method)

Sivells Mach 4 example: inflection angle 8.67°, RC=6, upstream ~14 in,
downstream to point J ~31 in (total ~45 in for r*=12.25 in → L/r*≈3.7)

## Validation Data Sources (not yet extracted)

- **Sivells Mach 4**: Input cards + output in contur repo (`docs/output.txt`)
- **G&N TR-R-33**: Tabulated nozzle coordinates for multiple cases
- **Beckwith & Moore TN-3322**: Streamline coordinates
- **NASA RP-1104**: Bell nozzle parametric contours at γ=1.2
- **Adams thesis (2021)**: Mach 3 wind tunnel nozzle designed with CONTUR,
  validated with CFD (U. Tennessee, https://trace.tennessee.edu/utk_gradthes/4276/)
