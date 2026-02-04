"""
Problems package: OCPs and config. Each problem module provides attach_config(config, overrides).
create_config(system=..., **overrides) builds a config by dispatching to the right attach_config.
"""
import os
from pathlib import Path
from types import SimpleNamespace

# Project root: AdHoc/ (results go under AdHoc/experiments/results/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from . import allen_cahn
from . import burgers
from . import kdv
# Registry: system name -> attach_config(config, overrides). Add new problems here.
_REGISTRY = {
    "burgers": burgers.attach_config,
    "allen_cahn": allen_cahn.attach_config,
    "kdv": kdv.attach_config,
}

# Shared defaults (overridden by overrides or by attach_config defaults)
default_system = "burgers"
default_seed = 32
default_fp_tol = 1e-02


def create_config(**overrides):
    """Build config: shared defaults + problem attach_config + overrides."""
    config = SimpleNamespace(
        seed=overrides.get("seed", default_seed),
        system=overrides.get("system", default_system),
        fp_tol=overrides.get("fp_tol", default_fp_tol),
    )
    # Apply overrides before attach so OCP is built with correct params (e.g. perturbation_type)
    for k, v in overrides.items():
        if k not in ("seed", "system", "fp_tol"):
            setattr(config, k, v)
    attach = _REGISTRY.get(config.system)
    if attach is None:
        raise ValueError(f"Unknown system: {config.system!r}. Known: {list(_REGISTRY)}")
    attach(config, overrides)
    return config


def get_results_dir(config, subdir=None):
    """Results under experiments: experiments/results/{system}/seed_{seed}/{subdir}/.
    subdir is e.g. 'data', 'saved_models', 'tables', 'plots'.
    """
    base = _PROJECT_ROOT / "experiments" / "results" / config.system / f"seed_{config.seed}"
    out = base / subdir if subdir else base
    return str(out)


__all__ = ["create_config", "get_results_dir", "default_system", "default_seed", "default_fp_tol"]
