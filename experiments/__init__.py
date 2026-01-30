"""
Experiments package: ensures project root is on sys.path and re-exports common imports
so notebooks can do:  from experiments import create_config, P, tables
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from problems import create_config
import figures

# Aliases used by notebooks (e.g. P.plot_3d, tables.save_config_table)
P = figures
tables = figures

__all__ = ["create_config", "figures", "P", "tables"]
