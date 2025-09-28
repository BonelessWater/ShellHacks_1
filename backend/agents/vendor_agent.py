from typing import Dict, Any
from types import SimpleNamespace


class VendorAgent:
    """Vendor validation agent used by unit tests.

    Tests expect a `check_vendor(invoice)` method that returns an object with
    attributes `vendor_valid` (bool) and `risk_factor` (str in LOW/MEDIUM/HIGH).
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def check_vendor(self, invoice: Dict[str, Any]) -> SimpleNamespace:
        # Very small heuristic: vendor present -> valid
        vendor = None
        if isinstance(invoice, dict):
            vendor = invoice.get("vendor")
        else:
            # try attribute access (dataclass)
            vendor = getattr(invoice, "vendor", None)

        vendor_valid = bool(vendor)
        # Deterministic risk factor for tests
        risk = "LOW" if vendor_valid else "MEDIUM"

        return SimpleNamespace(vendor_valid=vendor_valid, risk_factor=risk)


