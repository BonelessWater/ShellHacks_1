from typing import Dict, Any
from types import SimpleNamespace


class PatternAgent:
    """Pattern analysis agent.

    Tests expect `analyze_patterns(invoice)` returning an object with
    `anomalies_found` (int) and `risk_factor` (str).
    """

    def analyze_patterns(self, invoice: Dict[str, Any]) -> SimpleNamespace:
        notes = invoice.get("notes") if isinstance(invoice, dict) else getattr(invoice, "notes", None)
        text = str(notes or invoice.get("description") if isinstance(invoice, dict) else getattr(invoice, "description", ""))
        anomalies = 1 if "suspicious" in text.lower() else 0
        risk = "HIGH" if anomalies else "LOW"
        return SimpleNamespace(anomalies_found=anomalies, risk_factor=risk)

