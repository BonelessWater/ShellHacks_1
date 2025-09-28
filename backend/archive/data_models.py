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

    @staticmethod
    def validate_invoice(data: Dict[str, Any]) -> Optional['Invoice']:
        """Create an Invoice instance from a dict-like payload (archived shim)."""
        if not isinstance(data, dict):
            return None
        invoice_id = data.get("invoice_id") or data.get("id")
        vendor = data.get("vendor")
        total = data.get("total")
        items = data.get("items") or []
        date = data.get("date") or data.get("invoice_date")
        if not invoice_id or vendor is None or total is None:
            return None
        try:
            total = float(total)
        except Exception:
            return None
        return Invoice(invoice_id=invoice_id, vendor=vendor, total=total, items=items, date=date)


@dataclass
class Invoice:
    """Basic invoice data model"""
    invoice_id: str
    vendor: str
    total: float
    items: List[Dict[str, Any]] = None
    date: Optional[str] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []

    def to_dict(self) -> Dict[str, Any]:
        # Ensure items are plain dicts
        items_out = []
        for it in self.items:
            if hasattr(it, "__dict__"):
                items_out.append(it.__dict__)
            else:
                items_out.append(it)
        return {
            "invoice_id": self.invoice_id,
            "vendor": self.vendor,
            "date": self.date,
            "items": items_out,
            "total": self.total,
        }


@dataclass
class PatternAnalysisResult:
    """Pattern analysis result data model (flexible compat)."""
    def __init__(self, anomalies_found: int = 0, anomaly_details: List[str] = None, risk_factor: str = "LOW", pattern_types: List[str] = None, severity_score: float = 0.0, **kwargs):
        self.anomalies_found = anomalies_found
        self.anomaly_details = anomaly_details or []
        self.risk_factor = risk_factor
        self.pattern_types = pattern_types or []
        self.severity_score = severity_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomalies_found": self.anomalies_found,
            "anomaly_details": self.anomaly_details,
            "risk_factor": self.risk_factor,
            "pattern_types": self.pattern_types,
            "severity_score": self.severity_score,
        }


@dataclass
class TotalsCheckResult:
    """Totals check result data model"""
    is_valid: bool
    expected_total: float
    actual_total: float
    variance: float = 0.0

    def __post_init__(self):
        # compatibility: provide risk_factor attribute
        if not hasattr(self, "risk_factor"):
            self.risk_factor = "LOW"

    @property
    def difference(self) -> float:
        return self.actual_total - self.expected_total

    @property
    def totals_match(self) -> bool:
        return abs(self.difference) <= (self.variance or 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "expected_total": self.expected_total,
            "actual_total": self.actual_total,
            "difference": self.difference,
            "variance": self.variance,
        }


@dataclass
class InvoiceItem:
    """Compatibility item representation used by tests and agents."""
    description: str
    quantity: float
    unit_price: float
    total: Optional[float] = None

    def __post_init__(self):
        # If total not provided, infer from quantity * unit_price
        try:
            if self.total is None:
                self.total = float(self.quantity) * float(self.unit_price)
        except Exception:
            # Fallback to 0.0 if conversion fails
            self.total = 0.0
        # Provide alias used by agents
        try:
            self.line_total = float(self.total)
        except Exception:
            self.line_total = 0.0


def create_test_invoice(invoice_id: str = "TEST-001", vendor_name: str = "Test Vendor", total: float = 100.0, include_issues: bool = False) -> Invoice:
    """Helper to quickly create a minimal Invoice instance for tests.

    include_issues is accepted for compatibility with older test helpers; when True
    it may include simple issue markers in returned invoice metadata (currently unused).
    """
    item = InvoiceItem(description="Test item", quantity=1, unit_price=float(total), total=float(total))
    inv = Invoice(invoice_id=invoice_id, vendor={"name": vendor_name}, total=float(total), items=[item.__dict__])
    if include_issues:
        # attach a minimal issues field expected by some tests/agents
        try:
            inv.issues = ["test_issue"]
        except Exception:
            pass
    return inv


@dataclass
class VendorCheckResult:
    """Vendor check result data model"""
    def __init__(self, is_valid: bool = True, vendor_name: str = None, vendor: str = None, vendor_valid: bool = True, confidence: float = 0.0, issues: List[str] = None, **kwargs):
        # Accept both 'vendor' and 'vendor_name' and 'vendor_valid' for compatibility
        if vendor_name is None and vendor is not None:
            vendor_name = vendor
        self.is_valid = is_valid
        self.vendor_name = vendor_name
        self.vendor_valid = vendor_valid if vendor_valid is not None else is_valid
        self.confidence = confidence
        self.issues = issues or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "vendor_name": self.vendor_name,
            "vendor_valid": self.vendor_valid,
            "confidence": self.confidence,
            "issues": self.issues,
        }


# Additional compatibility exports
__all__ = [
    'DataValidator',
    'Invoice', 
    'PatternAnalysisResult',
    'TotalsCheckResult',
    'VendorCheckResult'
]