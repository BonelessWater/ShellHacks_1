"""Lightweight ML and graph-based agents for fraud detection.

These agents are intentionally small and accept injected predictors/graph
builders so they remain test-friendly and do not require TensorFlow or
graph libraries at import time.
"""
from typing import Any, Dict, List, Optional


class TransactionAnomalyAgent:
    """Agent that uses a provided predictor function to score invoices.

    predictor: callable taking a list of feature dicts and returning a list
    of anomaly scores (0..1). The agent accepts the predictor at init so
    tests can inject a simple stub.
    """

    def __init__(self, predictor=None):
        self.predictor = predictor

    def _extract_features(self, invoice) -> List[Dict[str, Any]]:
        # Minimal feature extraction: line amounts, total, vendor length
        features = []
        for item in getattr(invoice, "items", []):
            amt = getattr(item, "line_total", None)
            if amt is None:
                try:
                    amt = float(item.quantity) * float(item.unit_price)
                except Exception:
                    amt = 0.0
            features.append({"amount": float(amt)})

        # Add invoice-level features
        features.append({"total": float(getattr(invoice, "total", 0.0))})
        features.append({"vendor_len": len(getattr(invoice, "vendor", ""))})
        return features

    def run(self, invoice) -> Dict[str, Any]:
        try:
            features = self._extract_features(invoice)
            if not features:
                return {"risk_score": 0.0, "details": "no_features"}

            if self.predictor:
                # Predictor may accept list of feature dicts and return scores
                scores = self.predictor(features)
                # Normalize to 0..1
                avg_score = float(sum(scores)) / max(1, len(scores))
                return {"risk_score": min(max(avg_score, 0.0), 1.0), "details": "model_based"}

            # Default heuristic: flag if a single line is > 50% of total
            total = float(getattr(invoice, "total", 0.0))
            max_line = 0.0
            for f in features:
                if f.get("amount", 0) > max_line:
                    max_line = f.get("amount", 0)

            if total and max_line / total > 0.5:
                return {"risk_score": 0.8, "details": "large_single_line"}

            return {"risk_score": 0.0, "details": "heuristic_ok"}

        except Exception as e:
            return {"risk_score": 1.0, "details": str(e)}


class GraphFraudAgent:
    """Agent performing lightweight graph analysis on vendor relationships.

    graph_builder: callable that accepts a list of edges and returns an object
    that supports a `score_node(node_id)` method. In tests this can be a stub.
    """

    def __init__(self, graph_builder=None):
        self.graph_builder = graph_builder

    def run(self, invoice) -> Dict[str, Any]:
        try:
            vendor = getattr(invoice, "vendor", None)
            if not vendor:
                return {"risk_score": 0.0, "details": "no_vendor"}

            # Build minimal edge list from invoice items (vendor -> payee)
            edges = []
            for item in getattr(invoice, "items", []):
                payee = getattr(item, "payee", None) or getattr(item, "vendor", None)
                if payee:
                    edges.append((vendor, payee))

            if not edges:
                return {"risk_score": 0.0, "details": "no_edges"}

            if self.graph_builder:
                graph = self.graph_builder(edges)
                try:
                    score = graph.score_node(vendor)
                    return {"risk_score": min(max(float(score), 0.0), 1.0), "details": "graph_score"}
                except Exception:
                    return {"risk_score": 0.0, "details": "graph_builder_failed"}

            # Default heuristic: if >3 distinct payees, raise suspicion
            payees = {p for (_, p) in edges}
            if len(payees) > 3:
                return {"risk_score": 0.6, "details": f"many_payees:{len(payees)}"}

            return {"risk_score": 0.0, "details": "graph_heuristic_ok"}

        except Exception as e:
            return {"risk_score": 1.0, "details": str(e)}
