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

        if not invoice_id or vendor is None or total is None:
            return None

        try:
            total = float(total)
        except Exception:
            return None

        return Invoice(invoice_id=invoice_id, vendor=vendor, total=total, items=items)

    @staticmethod
    def normalize_vendor_name(name: str) -> str:
        if not name:
            return ""
        return str(name).strip().lower()

@dataclass
class Invoice:
    invoice_id: str
    vendor: str
    total: float
    items: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []

    @property
    def calculated_total(self) -> float:
        """Calculate total from items if available (quantity * unit_price)."""
        total = 0.0
        for it in self.items:
            try:
                qty = float(it.get("quantity", 1))
            except Exception:
                qty = 1
            try:
                price = float(it.get("unit_price", it.get("price", 0.0)))
            except Exception:
                price = 0.0
            total += qty * price
        return total

@dataclass
class PatternAnalysisResult:
    pattern_type: str
    confidence: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class TotalsCheckResult:
    # Keep dataclass for simple usage, but allow alternate kwarg names in helper
    is_valid: bool
    expected_total: float
    actual_total: float
    variance: float = 0.0

    def __init__(self, is_valid: bool = True, expected_total: float = 0.0, actual_total: float = 0.0, reported_total: float = None, variance: float = 0.0, **kwargs):
        # Accept alternate kwarg names used across legacy code (reported_total)
        if reported_total is not None:
            actual_total = reported_total
        # assign
        self.is_valid = is_valid
        self.expected_total = expected_total
        self.actual_total = actual_total
        self.variance = variance
        # Compatibility: some agents expect a 'risk_factor' attribute
        self.risk_factor = kwargs.get("risk_factor", "LOW")

    @property
    def difference(self) -> float:
        return self.actual_total - self.expected_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "expected_total": self.expected_total,
            "actual_total": self.actual_total,
            "difference": self.difference,
            "variance": self.variance,
        }

    @property
    def totals_match(self) -> bool:
        return abs(self.difference) <= (self.variance or 0.0)

class VendorCheckResult:
    def __init__(self, is_valid: bool = True, vendor_name: str = None, vendor: str = None, confidence: float = 0.0, issues: List[str] = None, **kwargs):
        # Accept either 'vendor' or 'vendor_name'
        if vendor_name is None and vendor is not None:
            vendor_name = vendor
        self.is_valid = is_valid
        self.vendor_name = vendor_name
        self.confidence = confidence
        self.issues = issues or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "vendor_name": self.vendor_name,
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
