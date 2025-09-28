"""Shim to expose archived data models as `backend.data_models`.

Some modules import `backend.data_models` directly; during the merge the
authoritative implementations live under `backend.archive.data_models`.
This shim re-exports the archived implementations to preserve backward
compatibility for tests and other imports.
"""

"""Temporary shim for `backend.data_models`.

This module re-exports public symbols from `backend.archive.data_models` so
existing imports continue to work during/after a large merge. It's a
short-term compatibility layer; see `backend/compat.py` for the re-export
implementation and add a cleanup PR to remove this shim once callers are
updated.
"""

from backend import compat

# Re-export public symbols from backend.archive.data_models (or fallbacks)
compat.reexport_module("data_models", globals())
#!/usr/bin/env python3
"""
Data models and validation for Invoice Fraud Detection System
"""

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class InvoiceItem:
    """Represents a single line item in an invoice"""

    description: str
    quantity: float
    unit_price: float

    @property
    def line_total(self) -> float:
        return self.quantity * self.unit_price

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Invoice:
    """Represents a complete invoice"""

    invoice_id: str
    vendor: str
    date: str
    items: List[InvoiceItem]
    total: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Invoice":
        """Create Invoice from dictionary"""
        items = []
        for item_data in data.get("items", []):
            if isinstance(item_data, dict):
                items.append(
                    InvoiceItem(
                        description=str(item_data.get("description", "")),
                        quantity=float(item_data.get("quantity", 0)),
                        unit_price=float(item_data.get("unit_price", 0)),
                    )
                )

        return cls(
            invoice_id=str(data.get("invoice_id", "")),
            vendor=str(data.get("vendor", "")),
            date=str(data.get("date", "")),
            items=items,
            total=float(data.get("total", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invoice_id": self.invoice_id,
            "vendor": self.vendor,
            "date": self.date,
            "items": [item.to_dict() for item in self.items],
            "total": self.total,
        }

    @property
    def calculated_total(self) -> float:
        """Calculate total from line items"""
        return sum(item.line_total for item in self.items)

    @property
    def total_mismatch(self) -> float:
        """Amount of mismatch between reported and calculated total"""
        return abs(self.total - self.calculated_total)


@dataclass
class VendorCheckResult:
    """Results from vendor validation"""

    vendor: str
    vendor_valid: bool
    risk_factor: str
    confidence: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TotalsCheckResult:
    """Results from totals validation"""

    reported_total: float
    calculated_total: float
    difference: float
    totals_match: bool
    risk_factor: str
    tolerance_used: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternAnalysisResult:
    """Results from pattern analysis"""

    anomalies_found: int
    anomaly_details: List[str]
    risk_factor: str
    pattern_types: List[str] = None
    severity_score: float = 0.0

    def __post_init__(self):
        if self.pattern_types is None:
            self.pattern_types = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FraudDetectionSummary:
    """Summary of fraud detection analysis"""

    fraud_risk: str  # HIGH, MEDIUM, LOW
    conclusion: str
    risk_factors: List[str]
    vendor_valid: bool
    totals_match: bool
    anomalies_found: int
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FraudDetectionResult:
    """Complete fraud detection result"""

    status: str
    iterations: int
    llm_working: bool
    results: Dict[str, Any]
    summary: FraudDetectionSummary
    review: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert summary to dict if it's an object
        if hasattr(self.summary, "to_dict"):
            result["summary"] = self.summary.to_dict()
        return result

    # Backwards-compatible attribute accessors used by tests
    @property
    def fraud_risk(self) -> str:
        if hasattr(self.summary, "fraud_risk"):
            return getattr(self.summary, "fraud_risk")
        return "UNKNOWN"

    @property
    def confidence_score(self) -> float:
        if hasattr(self.summary, "confidence_score"):
            return getattr(self.summary, "confidence_score")
        return 0.0


class JSONParser:
    """Utility class for robust JSON parsing from LLM responses"""

    @staticmethod
    def extract_json_from_text(text: str) -> Any:
        """Extract JSON from text, handling various formats"""
        if not text:
            return None

        # Clean the text first
        text = text.strip()

        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except:
            pass

        # Try to find JSON patterns
        json_patterns = [
            r"```json\s*([^`]+)\s*```",  # Code block
            r"```\s*([^`]+)\s*```",  # Generic code block
            r"(\[.*?\])",  # Array anywhere in text
            r"(\{.*?\})",  # Object anywhere in text
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            for match in matches:
                try:
                    cleaned = match.strip()
                    if cleaned:
                        return json.loads(cleaned)
                except:
                    continue

        # Last resort: try to extract from lines that look like JSON
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if (line.startswith("[") and line.endswith("]")) or (
                line.startswith("{") and line.endswith("}")
            ):
                try:
                    return json.loads(line)
                except:
                    continue

        return None

    @staticmethod
    def safe_json_parse(text: str, default: Any = None) -> Any:
        """Safely parse JSON with fallback"""
        result = JSONParser.extract_json_from_text(text)
        if result is not None:
            return result

        import logging

        log = logging.getLogger("fraud_detection_parser")
        log.warning(f"Could not parse JSON from: {text[:100]}...")
        return default


class DataValidator:
    """Validates and normalizes input data"""

    @staticmethod
    def validate_invoice(data: Union[Dict[str, Any], Invoice]) -> Optional[Invoice]:
        """Validate and normalize invoice data"""
        try:
            if isinstance(data, Invoice):
                return data

            if not isinstance(data, dict):
                return None

            # Required fields check
            required_fields = ["invoice_id", "vendor", "items", "total"]
            for field in required_fields:
                if field not in data:
                    return None

            return Invoice.from_dict(data)

        except (ValueError, TypeError, KeyError):
            return None

    @staticmethod
    def normalize_vendor_name(vendor: str) -> str:
        """Normalize vendor name for comparison"""
        if not vendor:
            return ""

        # Basic normalization
        normalized = vendor.strip()
        normalized = re.sub(r"\s+", " ", normalized)  # Multiple spaces to single
        normalized = normalized.title()  # Title case

        return normalized

    @staticmethod
    def validate_tasks(tasks: Any) -> List[str]:
        """Validate and normalize task list"""
        valid_tasks = ["CheckVendor", "CheckTotals", "AnalyzePatterns"]

        if not isinstance(tasks, list):
            return []

        normalized = []
        for task in tasks:
            if isinstance(task, str) and task in valid_tasks:
                normalized.append(task)

        return normalized


# Test data generators
def create_test_invoice(include_issues: bool = False) -> Invoice:
    """Create a test invoice for development/testing"""

    if include_issues:
        # Invoice with multiple issues
        items = [
            InvoiceItem("Office supplies", 10, 15.50),
            InvoiceItem("Gift cards", 5, 100.00),  # Suspicious
            InvoiceItem("Consulting services", 20, 75.00),
        ]
        total = 2200.00  # Intentional mismatch (should be 2155.00)
        vendor = "Unknown Vendor"  # Not in approved list
    else:
        # Clean invoice
        items = [
            InvoiceItem("Office supplies", 10, 15.50),
            InvoiceItem("Software license", 1, 500.00),
        ]
        total = 655.00  # Correct total
        vendor = "ACME Corp"  # Approved vendor

    return Invoice(
        invoice_id="INV-2024-001",
        vendor=vendor,
        date="2024-09-26",
        items=items,
        total=total,
    )
