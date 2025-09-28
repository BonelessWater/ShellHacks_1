"""Top-level shim for backend.data_models used by legacy imports."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class DataValidator:
    def validate(self, data: Dict[str, Any]) -> bool:
        return True

    @staticmethod
    def validate_invoice(data: Dict[str, Any]) -> Optional['Invoice']:
        """Create an Invoice instance from a dict-like payload.

        This is a minimal compatibility helper used by integration tests and
        legacy code paths. It performs light validation and returns None on
        missing required fields.
        """
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

    @staticmethod
    def normalize_vendor_name(name: str) -> str:
        if not name:
            return ""
        # Preserve original casing for approved vendors; only trim whitespace
        return str(name).strip()

@dataclass
class Invoice:
    invoice_id: str
    vendor: str
    total: float
    items: List[Dict[str, Any]] = None
    date: Optional[str] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []

    @property
    def calculated_total(self) -> float:
        """Calculate total from items if available (quantity * unit_price)."""
        total = 0.0
        for it in self.items:
            # Support either dict-like items or objects with attributes
            if isinstance(it, dict):
                try:
                    qty = float(it.get("quantity", 1))
                except Exception:
                    qty = 1
                try:
                    price = float(it.get("unit_price", it.get("price", 0.0)))
                except Exception:
                    price = 0.0
            else:
                try:
                    qty = float(getattr(it, "quantity", 1))
                except Exception:
                    qty = 1
                try:
                    price = float(getattr(it, "unit_price", getattr(it, "price", 0.0)))
                except Exception:
                    price = 0.0
            total += qty * price
        return total

    def to_dict(self) -> Dict[str, Any]:
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
class InvoiceItem:
    description: str
    quantity: float
    unit_price: float
    total: Optional[float] = None

    def __post_init__(self):
        try:
            if self.total is None:
                self.total = float(self.quantity) * float(self.unit_price)
        except Exception:
            self.total = 0.0


def create_test_invoice(invoice_id: str = "TEST-001", vendor_name: str = "Test Vendor", total: float = 100.0, include_issues: bool = False) -> Invoice:
    item = InvoiceItem(description="Test item", quantity=1, unit_price=float(total), total=float(total))
    inv = Invoice(invoice_id=invoice_id, vendor={"name": vendor_name}, total=float(total), items=[item.__dict__])
    if include_issues:
        try:
            inv.issues = ["test_issue"]
        except Exception:
            pass
    return inv

@dataclass
class PatternAnalysisResult:
    """Flexible pattern analysis result used by agents/tests."""
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
    # Keep dataclass for simple usage, but allow alternate kwarg names in helper
    def __init__(self, **kwargs):
        # Accept a broad set of field names used across the codebase
        self.is_valid = kwargs.get("is_valid", kwargs.get("valid", True))
        # reported_total or expected_total naming differences
        self.reported_total = kwargs.get("reported_total", kwargs.get("actual_total", kwargs.get("reported", 0.0)))
        self.calculated_total = kwargs.get("calculated_total", kwargs.get("expected_total", kwargs.get("calculated", 0.0)))
        # Some callers pass 'actual_total' meaning reported; keep both synonyms
        self.actual_total = kwargs.get("actual_total", self.reported_total)
        self.expected_total = kwargs.get("expected_total", self.calculated_total)
        self.difference = kwargs.get("difference", self.actual_total - self.expected_total if (self.actual_total and self.expected_total) else 0.0)
        self.totals_match = kwargs.get("totals_match", abs(self.difference) <= kwargs.get("variance", 0.0))
        self.variance = kwargs.get("variance", kwargs.get("tolerance", 0.0))
        self.risk_factor = kwargs.get("risk_factor", "LOW")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "reported_total": self.reported_total,
            "calculated_total": self.calculated_total,
            "difference": self.difference,
            "variance": self.variance,
            "totals_match": self.totals_match,
            "risk_factor": self.risk_factor,
        }

class VendorCheckResult:
    def __init__(self, is_valid: bool = True, vendor_name: str = None, vendor: str = None, vendor_valid: bool = True, confidence: float = 0.0, issues: List[str] = None, **kwargs):
        if vendor_name is None and vendor is not None:
            vendor_name = vendor
        self.is_valid = is_valid
        self.vendor_name = vendor_name
        self.vendor_valid = vendor_valid if vendor_valid is not None else is_valid
        self.confidence = confidence
        self.issues = issues or []
        # compatibility
        self.risk_factor = kwargs.get("risk_factor", "LOW")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "vendor_name": self.vendor_name,
            "vendor_valid": self.vendor_valid,
            "confidence": self.confidence,
            "issues": self.issues,
        }


class FraudDetectionResult:
    """Minimal result object used by integration tests and shims."""
    def __init__(self, fraud_score: float = 0.0, reasons: List[str] = None, metadata: Dict[str, Any] = None, **kwargs):
        """Compatibility-friendly constructor.

        Accepts common pipeline keyword args (status, iterations, processing_time,
        llm_working, results, summary, review) and assigns them to the
        instance so calling code can access them directly (e.g. result.summary.get(...)).
        """
        self.fraud_score = fraud_score
        self.reasons = reasons or []
        self.metadata = metadata or {}

        # Populate commonly-used pipeline fields from kwargs for compatibility
        self.status = kwargs.get("status", getattr(self, "status", "ok"))
        self.iterations = kwargs.get("iterations", getattr(self, "iterations", 0))
        self.processing_time = kwargs.get("processing_time", getattr(self, "processing_time", 0.0))
        self.llm_working = kwargs.get("llm_working", kwargs.get("llm", False))
        self.results = kwargs.get("results", getattr(self, "results", {}))
        # summary may be a dict or a FraudDetectionSummary object; store as-is
        self.summary = kwargs.get("summary", getattr(self, "summary", None))
        self.review = kwargs.get("review", {})

    # Compatibility attributes used by pipeline
    status: str = "ok"
    iterations: int = 0
    processing_time: float = 0.0
    summary: Optional[Dict[str, Any]] = None
    results: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fraud_score": self.fraud_score,
            "reasons": self.reasons,
            "metadata": self.metadata,
            "status": getattr(self, "status", "ok"),
            "iterations": getattr(self, "iterations", 0),
            "processing_time": getattr(self, "processing_time", 0.0),
            "summary": getattr(self, "summary", {}),
            "results": getattr(self, "results", {}),
        }

    # Backwards-compatible properties expected by tests and callers
    @property
    def fraud_risk(self):
        if isinstance(self.summary, dict):
            return self.summary.get("fraud_risk")
        if hasattr(self.summary, "fraud_risk"):
            return getattr(self.summary, "fraud_risk")
        return None

    @property
    def confidence_score(self):
        if isinstance(self.summary, dict):
            return self.summary.get("confidence_score")
        if hasattr(self.summary, "confidence_score"):
            return getattr(self.summary, "confidence_score")
        return None


class FraudDetectionSummary:
    """Summary object used by agents during integration tests and pipeline.

    Accept a broad set of keyword names to remain compatible with archived
    code paths (fraud_risk, conclusion, risk_factors, vendor_valid, etc.).
    """
    def __init__(self, total_checks: int = 0, fraud_cases: int = 0, average_score: float = 0.0, fraud_risk: str = None, conclusion: str = None, risk_factors: List[str] = None, vendor_valid: bool = True, totals_match: bool = True, anomalies_found: int = 0, confidence_score: float = 0.0, **kwargs):
        self.total_checks = total_checks
        self.fraud_cases = fraud_cases
        self.average_score = average_score
        self.fraud_risk = fraud_risk
        self.conclusion = conclusion
        self.risk_factors = risk_factors or []
        self.vendor_valid = vendor_valid
        self.totals_match = totals_match
        self.anomalies_found = anomalies_found
        self.confidence_score = confidence_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_checks": self.total_checks,
            "fraud_cases": self.fraud_cases,
            "average_score": self.average_score,
            "fraud_risk": self.fraud_risk,
            "conclusion": self.conclusion,
            "risk_factors": self.risk_factors,
            "vendor_valid": self.vendor_valid,
            "totals_match": self.totals_match,
            "anomalies_found": self.anomalies_found,
            "confidence_score": self.confidence_score,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


class JSONParser:
    """Very small helper used by legacy code to parse/validate JSON payloads."""

    @staticmethod
    def parse(raw: str) -> Dict[str, Any]:
        import json as _json

        try:
            return _json.loads(raw)
        except Exception:
            return {}





__all__ = [
    'DataValidator',
    'Invoice', 
    'PatternAnalysisResult',
    'TotalsCheckResult',
    'VendorCheckResult',
    'FraudDetectionResult',
    'FraudDetectionSummary',
]
