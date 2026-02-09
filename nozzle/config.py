"""YAML configuration loading with inheritance.

Supports a `base:` key for config inheritance with deep merge,
matching the pattern from the water_hammer sibling project.
"""

import yaml
import copy
from pathlib import Path


def _deep_merge(base, overrides):
    """Recursively merge overrides into base dict.

    - Scalars in overrides replace base values
    - Dicts are merged recursively
    - None values in overrides remove the key
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_config(name, raw, all_raw, resolved_cache=None):
    """Resolve a single config, following base references.

    Parameters
    ----------
    name : str
        Config name.
    raw : dict
        Raw config dict.
    all_raw : dict
        All raw configs (for base resolution).
    resolved_cache : dict
        Cache of already-resolved configs.

    Returns
    -------
    dict : Resolved config (base fields merged in).
    """
    if resolved_cache is None:
        resolved_cache = {}

    if name in resolved_cache:
        return resolved_cache[name]

    if 'base' in raw:
        base_name = raw['base']
        if base_name not in all_raw:
            raise ValueError(f"Config '{name}' references unknown base '{base_name}'")
        base_resolved = _resolve_config(
            base_name, all_raw[base_name], all_raw, resolved_cache
        )
        # Merge: base + overrides (excluding the 'base' key itself)
        overrides = {k: v for k, v in raw.items() if k != 'base'}
        resolved = _deep_merge(base_resolved, overrides)
    else:
        resolved = copy.deepcopy(raw)

    resolved_cache[name] = resolved
    return resolved


def load_config(path):
    """Load a nozzle design configuration from YAML.

    Supports:
    - Single config: `nozzle:` top-level key
    - Multiple configs: `configs:` top-level key with inheritance
    - Comparisons: `comparisons:` for side-by-side plots
    - Output control: `outputs:` list

    Parameters
    ----------
    path : str or Path
        Path to YAML config file.

    Returns
    -------
    dict with keys:
        configs : dict of {name: resolved_config}
        comparisons : list of comparison specs
        outputs : list of output types
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty config file: {path}")

    # Handle single-config shorthand
    if 'nozzle' in raw and 'configs' not in raw:
        configs_raw = {'default': raw['nozzle']}
    elif 'configs' in raw:
        configs_raw = raw['configs']
    else:
        # Treat entire file as a single config
        configs_raw = {'default': raw}

    # Resolve inheritance
    resolved_cache = {}
    configs = {}
    for name, cfg in configs_raw.items():
        configs[name] = _resolve_config(name, cfg, configs_raw, resolved_cache)

    comparisons = raw.get('comparisons', [])
    outputs = raw.get('outputs', ['contour', 'performance'])

    return {
        'configs': configs,
        'comparisons': comparisons,
        'outputs': outputs,
    }


def _parse_length(value):
    """Parse a length value with optional unit suffix.

    Supports: mm, cm, m, in, ft.  Plain numbers are treated as mm.
    Returns value in meters.
    """
    if isinstance(value, (int, float)):
        return float(value) * 1e-3  # default: mm

    s = str(value).strip()
    units = {'mm': 1e-3, 'cm': 1e-2, 'm': 1.0,
             'in': 0.0254, 'ft': 0.3048}
    for suffix, factor in sorted(units.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            num = s[:-len(suffix)].strip()
            return float(num) * factor
    # No unit suffix — treat as mm
    return float(s) * 1e-3


def build_nozzle_spec(cfg):
    """Convert a resolved config dict into a nozzle specification.

    Parameters
    ----------
    cfg : dict
        Resolved config from load_config.

    Returns
    -------
    dict with standardized keys:
        type : str — 'conical', 'rao', 'mln', 'tic', 'sivells', 'custom'
        gamma : float
        area_ratio : float
        throat_radius_m : float or None — physical throat radius in meters
        Additional keys depend on type.
    """
    from nozzle.gas import area_mach_ratio, mach_from_area_ratio, mach_from_pressure_ratio

    gamma = float(cfg.get('gamma', 1.4))

    # --- Resolve throat radius (optional) ---
    throat_radius_m = None
    if 'throat_radius' in cfg:
        throat_radius_m = _parse_length(cfg['throat_radius'])

    # --- Resolve exit condition: exit_radius > M_exit > area_ratio ---
    area_ratio = None
    M_exit = None

    if 'exit_radius' in cfg:
        exit_radius_m = _parse_length(cfg['exit_radius'])
        if throat_radius_m is None:
            raise ValueError("exit_radius requires throat_radius to be set")
        area_ratio = (exit_radius_m / throat_radius_m) ** 2
    if 'M_exit' in cfg:
        M_exit = float(cfg['M_exit'])
        if area_ratio is None:
            area_ratio = area_mach_ratio(M_exit, gamma)
    if 'area_ratio' in cfg:
        if area_ratio is None:
            area_ratio = float(cfg['area_ratio'])
        # If M_exit not set, compute from area_ratio
        if M_exit is None:
            M_exit = mach_from_area_ratio(area_ratio, gamma=gamma)
    if 'exit_pressure_ratio' in cfg:
        p_ratio = float(cfg['exit_pressure_ratio'])
        if p_ratio <= 0 or p_ratio >= 1:
            raise ValueError(
                f"exit_pressure_ratio must be between 0 and 1 (exclusive), "
                f"got {p_ratio}"
            )
        if M_exit is None:
            M_exit = mach_from_pressure_ratio(p_ratio, gamma)
        if area_ratio is None:
            area_ratio = area_mach_ratio(M_exit, gamma)
    if area_ratio is None:
        area_ratio = 10.0  # default
    if M_exit is None:
        M_exit = mach_from_area_ratio(area_ratio, gamma=gamma)

    spec = {
        'type': cfg.get('type', 'conical'),
        'gamma': gamma,
        'area_ratio': area_ratio,
        'M_exit': M_exit,
        'throat_radius_m': throat_radius_m,
    }

    ntype = spec['type']

    if ntype == 'conical':
        spec['half_angle_deg'] = float(cfg.get('half_angle_deg', 15))

    elif ntype == 'rao':
        spec['bell_fraction'] = float(cfg.get('bell_fraction', 0.8))

    elif ntype in ('mln', 'tic'):
        spec['n_chars'] = int(cfg.get('n_chars', 30))
        if ntype == 'tic':
            spec['truncation_fraction'] = float(
                cfg.get('truncation_fraction', 0.8))

    elif ntype == 'sivells':
        spec['rc'] = float(cfg.get('rc', 1.5))
        spec['inflection_angle_deg'] = (
            float(cfg['inflection_angle_deg'])
            if 'inflection_angle_deg' in cfg else None
        )
        spec['n_char'] = int(cfg.get('n_char', 41))
        spec['n_axis'] = int(cfg.get('n_axis', 21))
        spec['nx'] = int(cfg.get('nx', 13))
        spec['ix'] = int(cfg.get('ix', 0))
        spec['ie'] = int(cfg.get('ie', 0))
        spec['downstream'] = bool(cfg.get('downstream', False))
        spec['ip'] = int(cfg.get('ip', 10))
        spec['md'] = int(cfg['md']) if 'md' in cfg else None
        spec['nd'] = int(cfg['nd']) if 'nd' in cfg else None
        spec['nf'] = int(cfg['nf']) if 'nf' in cfg else None

    elif ntype == 'custom':
        spec['contour_file'] = cfg.get('contour_file', '')
        spec['n_chars'] = int(cfg.get('n_chars', 30))

    # Optional convergent section (applies to any nozzle type)
    if 'convergent' in cfg and isinstance(cfg['convergent'], dict):
        conv = cfg['convergent']
        spec['convergent'] = {
            'contraction_ratio': float(conv.get('contraction_ratio', 3.0)),
            'convergent_half_angle_deg': float(conv.get('half_angle_deg', 30.0)),
            'rc_upstream': float(conv.get('rc_upstream', 1.5)),
            'rc_downstream': float(conv.get('rc_downstream', 0.382)),
        }

    return spec
