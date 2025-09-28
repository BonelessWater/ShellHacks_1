from typing import Dict, Any
from types import SimpleNamespace


class TotalsAgent:
    """Totals validation agent for tests.

    Tests expect `check_totals(invoice)` returning object with `totals_match` and
    `difference` attributes.
    """

    def check_totals(self, invoice: Dict[str, Any]) -> SimpleNamespace:
        # Support dict-like invoices or dataclass-like objects
        def _get_attr(obj, attr, default=None):
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        # Compute subtotal from items if available
        items = _get_attr(invoice, "items", []) or []
        subtotal = 0.0
        for it in items:
            if isinstance(it, dict):
                qty = float(it.get("quantity", 1))
                price = float(it.get("unit_price", it.get("price", 0.0)))
            else:
                qty = float(getattr(it, "quantity", 1))
                price = float(getattr(it, "unit_price", getattr(it, "price", 0.0)))
            subtotal += qty * price

        tax = float(_get_attr(invoice, "tax_amount", _get_attr(invoice, "tax", 0)) or 0)
        total = float(
            _get_attr(invoice, "total", _get_attr(invoice, "total_amount", _get_attr(invoice, "amount", 0))) or 0
        )

        expected = subtotal + tax
        diff = total - expected
        match = abs(diff) < 1e-6
        return SimpleNamespace(totals_match=match, difference=diff)

