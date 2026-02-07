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


def build_nozzle_spec(cfg):
    """Convert a resolved config dict into a nozzle specification.

    Parameters
    ----------
    cfg : dict
        Resolved config from load_config.

    Returns
    -------
    dict with standardized keys:
        type : str — 'conical', 'rao', 'mln', 'tic', 'custom'
        gamma : float
        area_ratio : float
        Additional keys depend on type.
    """
    spec = {
        'type': cfg.get('type', 'conical'),
        'gamma': float(cfg.get('gamma', 1.4)),
        'area_ratio': float(cfg.get('area_ratio', 10)),
    }

    ntype = spec['type']

    if ntype == 'conical':
        spec['half_angle_deg'] = float(cfg.get('half_angle_deg', 15))

    elif ntype == 'rao':
        spec['bell_fraction'] = float(cfg.get('bell_fraction', 0.8))

    elif ntype == 'mln':
        spec['n_chars'] = int(cfg.get('n_chars', 30))
        spec['M_exit'] = float(cfg.get('M_exit', 2.0))

    elif ntype == 'tic':
        spec['n_chars'] = int(cfg.get('n_chars', 30))
        spec['M_exit'] = float(cfg.get('M_exit', 2.0))
        spec['truncation_fraction'] = float(cfg.get('truncation_fraction', 0.8))

    elif ntype == 'custom':
        spec['contour_file'] = cfg.get('contour_file', '')
        spec['n_chars'] = int(cfg.get('n_chars', 30))

    return spec
