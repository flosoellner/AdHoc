"""
Notebook helper: put project root on sys.path and re-export common imports.

Usage: ``from experiments import create_config, figures`` (and ``P``, ``tables`` as aliases).
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from problems import create_config
import figures

P = figures
tables = figures

__all__ = ["create_config", "figures", "P", "tables"]
