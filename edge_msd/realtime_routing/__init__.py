"""Realtime node-selection routing rebuild (Stage A).

Fixed two-tier deployment, node-selection-only actions, ICC physical semantics.
This package does not modify the upstream modules; it reuses them as the physical
baseline and adds a fixed-action environment plus checkpointing.
"""

from .fixed_placement import FixedPlacement, build_fixed_placement, load_placement, save_placement
from .scenario import build_scenario, load_paper_scenario

__all__ = [
    "FixedPlacement",
    "build_fixed_placement",
    "save_placement",
    "load_placement",
    "build_scenario",
    "load_paper_scenario",
]
