"""
Archived data models for compatibility with legacy imports.
Minimal implementations to allow imports to work.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json


@dataclass
class DataValidator:
    """Minimal data validator"""
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Basic validation - always returns True for compatibility"""
        return True


@dataclass
class Invoice:
    """Basic invoice data model"""
    invoice_id: str
    vendor: str
    total: float
    items: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []


@dataclass
class PatternAnalysisResult:
    """Pattern analysis result data model"""
    pattern_type: str
    confidence: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class TotalsCheckResult:
    """Totals check result data model"""
    is_valid: bool
    expected_total: float
    actual_total: float
    variance: float = 0.0


@dataclass
class VendorCheckResult:
    """Vendor check result data model"""
    is_valid: bool
    vendor_name: str
    confidence: float
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


# Additional compatibility exports
__all__ = [
    'DataValidator',
    'Invoice', 
    'PatternAnalysisResult',
    'TotalsCheckResult',
    'VendorCheckResult'
]