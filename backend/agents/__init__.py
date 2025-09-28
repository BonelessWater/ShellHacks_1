"""
backend.agents package initializer.

This file makes the `backend.agents` package importable when running the
application in different execution contexts (e.g. via `python -m` or when
the working directory is the project root). It is intentionally minimal.
"""

from .vendor_agent import VendorAgent
from .totals_agent import TotalsAgent
from .pattern_agent import PatternAgent
from .coordinator import AgentCoordinator
from .invoice_mapper import infer_mapping_from_columns, apply_mapping

__all__ = [
    "VendorAgent",
    "TotalsAgent",
    "PatternAgent",
    "AgentCoordinator",
    "infer_mapping_from_columns",
    "apply_mapping",
]
