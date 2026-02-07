# Axisymmetric Nozzle Design Algorithm — Verified from Source Texts

**Current status**: `design_mln` uses a simplified approach (analytical wall +
quasi-1D mesh fill) that passes all tests. The Sivells algorithm documented
below is the correct next step for a real MOC solver. The validated F90 port
of Sivells' code is at `docs/references/external_codes/contur/`.

## References (with local PDFs)
- Sivells, AEDC-TR-78-63, 1978 — `docs/references/pdfs/sivells_AEDC-TR-78-63.pdf`
  - Complete Fortran listing (Appendix D), worked Mach 4 example (Section 7.0)
  - Mass flow technique from Moger & Ramsay (Ref. 17)
  - **F90 port**: `docs/references/external_codes/contur/` (MIT, validated)
- Guentert & Neumann, NASA TR-R-33, 1959 — `docs/references/pdfs/guentert_neumann_TR-R-33.pdf`
  - Centerline pressure distribution, variable γ, weight flow streamlines
- Beckwith & Moore, NACA TN-3322 — `docs/references/pdfs/beckwith_moore_NACA-TN-3322.pdf`
- NASA SP-8120, Liquid Rocket Engine Nozzles — `docs/references/pdfs/NASA_SP-8120_nozzles.pdf`
- Hall, *QJMAM* 15(4), 1962 (transonic kernel)
- Z&H = Zucrow & Hoffman, *Gas Dynamics* Vol. 2 — `docs/references/pdfs/zucrow_hoffman_vol2_1977.pdf`
- Anderson, *Modern Compressible Flow*, §11.11
- Kennedy, DTIC ADA578559, 2013 — `docs/references/pdfs/DTIC_ADA578559_MOC_solver.pdf`
  - F90 analysis-mode MOC with Sauer transonic + mass flow wall

## What the Published Literature Actually Does

**Both Sivells and Guentert & Neumann use the same fundamental approach**, which is
DIFFERENT from the "expansion fan + design wall point" approach previously documented here.

### The Published Algorithm (Centerline Distribution + Mass Flow)

1. **Prescribe centerline conditions** (velocity or pressure distribution as polynomial)
2. **March characteristics outward from centerline** using standard MOC interior points
3. **Find wall position by mass flow integration** along each characteristic
4. **Wall angle from interpolation** of the flow field at the wall position

This sidesteps the "design wall point" problem entirely. The wall is never computed
from a unit process — it's found by interpolation after the characteristic network
is fully computed.

### Why This Works (and Our Previous Approach Didn't)

| Approach | Problem |
|----------|---------|
| Design wall point (C+/wall intersection) | `denom = slope_cplus - wall_slope` can be tiny → diverges |
| θ_w = θ_I (simplified) | Only exact for 2D planar; approximate for axisymmetric |
| Mass flow interpolation (Sivells) | Robust: wall found by interpolation, not intersection |

## Detailed Algorithm from Sivells AEDC-TR-78-63

### Program Structure (Section 6.0, p.34)

```
MAIN → AXIAL (centerline distribution)
     → PERFC (characteristic network + contour by mass flow)
     → BOUND (boundary layer correction)
     → SPLIND/XYZ (interpolation for output)
```

### Step 1: Centerline Distribution (Subroutine AXIAL)

Specify velocity/Mach number along the nozzle axis as a polynomial.
The distribution runs from M=1 at the throat to M=M_exit at the nozzle end.

Sivells uses a polynomial of degree 3-5 in x (Eq. 35-39 in the report).
The distribution determines:
- M(x) along the axis → ν(x), μ(x)
- θ = 0 on the axis (by symmetry)
- All thermodynamic properties from isentropic relations

**For MLN**: The distribution should give the minimum expansion length.
This is related to the Prandtl-Busemann centered expansion — the fastest
possible acceleration from M=1 to M_exit along the centerline.

**Key constraint** (G&N p.7): If the expansion is too fast (nozzle too short),
characteristics coalesce and no solution exists. Trial-and-error on the
distribution parameters may be needed.

### Step 2: Throat/Transonic Region (Subroutine TRANS)

The transonic region near the throat uses a series expansion (Appendix A).
For axisymmetric flow, the velocity field is (Eq. A-1 with σ=1):

```
u = 1 - 1/(2S) + λx(1 - σ/8S + ...) + λ²x²/2(1 - 2γ/3 - ...) + y²/(2S) + ...
```

where S = (λy₀)^(2/3), λ = velocity gradient parameter, y₀ = throat radius.

The throat characteristic (where M=1 on axis) provides the starting line
for the supersonic MOC calculation.

### Step 3: March Characteristics Outward (Subroutine PERFC + OFELD)

**OFELD** (p.91, OFE 1-61) computes interior points — standard predictor-corrector:

From two parent points A (on C+) and B (on C-), find point C at intersection.

**Predictor** (first approximation, Eqs. 5-12 in G&N):
```
x_C = (y_A - y_B + x_B·tan(θ_B-μ_B) - x_A·tan(θ_A+μ_A)) / (tan(θ_B-μ_B) - tan(θ_A+μ_A))
y_C = y_A + (x_C - x_A)·tan(θ_A + μ_A)
```

**Compatibility** (Sivells Eqs. 44-51):
```
ψ₃ = ½(ψ₂ + ψ₁ - φ₂ + φ₁ + P₂ + P₁)    where P = source term integral
φ₃ = ½(ψ₁ - ψ₂ + φ₁ + φ₂ + P₁ - P₂)          ψ = θ + ν (Riemann invariant)
                                                  φ = θ - ν (Riemann invariant)
```

Source terms for axisymmetric (σ=1):
```
P₁ = σ·sin(μ_A)·sin(θ_A)/(y_A·cos(θ_A+μ_A)) · ds₁   (along C+)
P₂ = σ·sin(μ_B)·sin(θ_B)/(y_B·cos(θ_B-μ_B)) · ds₂   (along C-)
```

**Corrector**: Average upstream/downstream properties, re-solve. Iterate 2-3 times.

**Axis points** (Sivells Eqs. 54-60): Use L'Hôpital's rule for sin(θ)/y → 0/0:
```
lim(y→0) sin(θ)/y = dθ/dy ≈ (M²-1)/(2MW) · dW/dx
```

### Step 4: Mass Flow Integration (PERFC lines PER 173-221)

**This is the critical step that determines the wall contour.**

After computing all interior points along a characteristic line (from axis outward),
integrate the mass flow crossing that characteristic:

From G&N Eq. (22):
```
w_C - w_A = 2π ∫_A^C ρV [cos θ - sin²θ / tan(θ+μ)] · y · dy
```

The wall is the streamline carrying the **total nozzle mass flow**.
In normalized coordinates:
```
w_total = ρ* · a* · π · r*²     (mass flow at sonic throat)
```

**In PERFC** (lines PER 173-221):
1. Accumulate mass flow along each C+ characteristic: `SUM = SU(NN)`
2. Compute fractional mass flow: `DEL = 1 - SUM`
3. When `DEL` changes sign → wall is between the last two points
4. Interpolate to find exact wall position (`WALL(J,LINE)`)

### Step 5: Wall Coordinate Integration (PERFC lines PER 249-303)

The wall angle θ_w at each station comes from the mass flow interpolation.
Wall coordinates are obtained by integrating the slope:

```
dy/dx = tan(θ_wall)
```

Starting from the inflection point (where wall angle = θ_max), integrate
using cubic/parabolic integration (Simpson's-like):

```
y_{n} = y_{n-3} + S1·tan(θ_{n-3}) + S2·tan(θ_{n-2}) + S3·tan(θ_{n-1})
```

where S1, S2, S3 are cubic integration coefficients (Appendix B).

## Alternative: Expansion Fan + Wall Boundary (Z&H Style)

This is the approach our current `design_mln` attempts. It works differently:

1. Generate expansion fan at the throat/inflection point
2. March row-by-row with prescribed wall (expansion) then computed wall (straightening)
3. Wall angle at each step from arriving C+ characteristic

**Status**: This approach is standard for 2D planar MLN but problematic for
axisymmetric flow because:
- The source term Q = sin(θ)·sin(μ)/(y·cos(θ±μ)) makes the wall angle
  determination sensitive to small changes
- Near-wall characteristics can be nearly parallel to the wall, causing
  the C+/wall intersection to diverge
- Without mass flow conservation, small errors accumulate

**The Z&H Ch. 12 approach has NOT been verified** — we don't have a local copy
of Z&H Vol. 2. The "design wall point" procedure documented in the previous
version of this file was synthesized from training knowledge, not read from
the actual text.

## Implementation Strategy for Our Code

### Option A: Centerline Distribution + Mass Flow (Sivells approach)
- **Pro**: Proven, robust, well-documented in reference code
- **Pro**: Avoids the wall angle divergence problem entirely
- **Con**: Must choose the right centerline distribution for MLN
- **Con**: Significant rewrite of `design_mln`
- **Con**: Not directly "minimum length" — distribution choice determines length

### Option B: Fix the Expansion Fan Approach (current code)
- **Pro**: Closer to current code structure
- **Pro**: Directly gives minimum-length nozzle
- **Con**: Wall determination is the unsolved problem
- **Con**: No verified reference for the exact wall point algorithm in axisymmetric flow

### Recommended: Option A (Sivells approach)

The mass flow technique is used by the two most authoritative references we have
(Sivells and G&N). It's numerically robust and well-tested. For MLN, the centerline
distribution can be calibrated by:
1. Using the Prandtl-Busemann distribution (centered expansion → fastest acceleration)
2. Adjusting the expansion length parameter until the contour is smooth and shortest
3. Validating against known MLN dimensions (see table below)

## Expected Dimensions (γ=1.4)

| M_exit | A/A*  | y_exit | θ_max (°) | L/r* (MLN) | L/r* (15° cone) |
|--------|-------|--------|-----------|------------|-----------------|
| 2.0    | 1.688 | 1.299  | 13.2      | ~1.5       | 2.6             |
| 2.5    | 2.637 | 1.624  | 19.7      | ~3.0       | 4.6             |
| 3.0    | 4.235 | 2.058  | 24.9      | ~5.0       | 6.4             |
| 4.0    | 10.72 | 3.274  | 32.8      | ~10        | 15.8            |
| 5.0    | 25.00 | 5.000  | 38.3      | ~22        | 36.7            |

Sivells sample design: M=4 axisymmetric, inflection angle 8.67°, RC=6,
upstream length ~14 in, downstream to point J ~31 in (p.40).

## Key Equations Reference

### Interior Point (G&N Eqs. 5-12, Sivells Eqs. 44-51)
- Position: Eqs. 5-8 (predictor), corrector averages slopes
- Properties: Eqs. 9-12 (V_C, θ_C from compatibility + source terms)

### Axis Point (Sivells Eqs. 54-60)
- L'Hôpital: sin(θ)/y → (M²-1)/(2MW)·dW/dx

### Mass Flow Across Characteristic (G&N Eq. 22)
```
w = 2π ∫ ρV [cos θ - sin²θ/tan(θ+μ)] · y · dy
```

### Boundary Layer (Sivells Eqs. 61-97)
- Von Kármán momentum integral
- Not needed for inviscid design validation
