# Nozzle — MOC Rocket Nozzle Design Tool

Method of Characteristics solver for axisymmetric supersonic nozzle design.
Designs optimized contours (MLN, Rao, TIC), generates Rao parabolic
approximations, and compares them — all from YAML config files.

## Quick Start

```bash
pip install -e .
nozzle example
nozzle run examples/configs/rao_vs_mln.yaml
nozzle web   # Launch interactive web interface (Pyodide)
```

## Nozzle Types

| Type | Description | Method |
|------|-------------|--------|
| **Conical** | Simple cone with analytical divergence loss | Geometry + λ = (1+cos α)/2 |
| **Rao parabolic** | Bell nozzle from Rao (1960) approximation | Sutton Table 3-4 + parabolic construction |
| **MLN** | Minimum-length nozzle via MOC | Expansion fan + straightening section |
| **TIC** | Truncated ideal contour | MLN truncated at specified fraction |
| **Custom** | User-supplied wall from CSV | Load (x, y) coordinates |

## Example Output

```
$ nozzle example --M-exit 2.0

Nozzle Design Comparison
  M_exit = 2.00, A/A* = 1.6875, γ = 1.4

1. Conical 15°:  Cf = 1.3992 (98.3% of ideal)
2. Rao 80% bell: Cf = 1.4066 (98.8% of ideal)
3. MLN:          Cf = 1.4234 (100.0% of ideal)
```

## YAML Configuration

Configs support inheritance via `base:` key:

```yaml
configs:
  baseline:
    type: rao
    gamma: 1.4
    area_ratio: 10
    bell_fraction: 0.8

  longer_bell:
    base: baseline
    bell_fraction: 1.0

  mln:
    type: mln
    gamma: 1.4
    area_ratio: 10
    M_exit: 2.0
    n_chars: 30

  tic_80:
    type: tic
    gamma: 1.4
    area_ratio: 10
    M_exit: 2.0
    truncation_fraction: 0.8

comparisons:
  - name: bell_vs_mln
    configs: [baseline, mln]

outputs:
  - contour
  - performance
```

## Web Interface

Launch an interactive browser-based design tool powered by Pyodide:

```bash
nozzle web --port 8080
```

Then open http://localhost:8080. The interface lets you adjust Mach number,
gamma, cone angle, and bell fraction and instantly compare contours and
performance.

## References

- Zucrow & Hoffman, *Gas Dynamics* Vol. 2, Wiley 1977
- Anderson, *Modern Compressible Flow*, 3rd ed., McGraw-Hill 2003
- Rao, *Jet Propulsion* 1958; *ARS Journal* 1960
- Hall, *QJMAM* 1962
- Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed.
- NACA 1135, 1953

## Testing

```bash
pytest tests/ -v
```
