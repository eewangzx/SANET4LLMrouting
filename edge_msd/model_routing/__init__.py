"""Experimental, profile-driven routing with constrained state reporting.

The bundled model profiles and workload are synthetic, not measurements of LLMs.
"""

from .environment import RoutingEnv
from .types import RoutingConfig

__all__ = ["RoutingConfig", "RoutingEnv"]
