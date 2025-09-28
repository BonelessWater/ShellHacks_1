from typing import List, Dict, Any


class AgentCoordinator:
    def __init__(self, agents: List[Any] = None):
        self.agents = agents or []

    def run(self, invoice: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for a in self.agents:
            name = a.__class__.__name__
            try:
                results[name] = a.analyze(invoice)
            except Exception as e:
                results[name] = {"error": str(e)}
        return results

    def execute_tasks(self, invoice: Dict[str, Any], tasks: List[str]) -> Dict[str, Any]:
        """Execute named tasks and return a mapping used by unit tests.

        The test calls execute_tasks(invoice, ["CheckVendor", "CheckTotals", "AnalyzePatterns"]).
        Map these to VendorAgent.check_vendor, TotalsAgent.check_totals, PatternAgent.analyze_patterns
        if those agents are present in self.agents.
        """
        results = {}
        # If no agents provided, instantiate basic ones for the coordinator to call
        if not self.agents:
            try:
                # Lazy import to avoid cycles
                from .vendor_agent import VendorAgent
                from .totals_agent import TotalsAgent
                from .pattern_agent import PatternAgent

                self.agents = [VendorAgent(), TotalsAgent(), PatternAgent()]
            except Exception:
                # If imports fail, proceed with empty list
                self.agents = []
        for t in tasks:
            try:
                if t == "CheckVendor":
                    # find first VendorAgent
                    agent = next((a for a in self.agents if a.__class__.__name__ == "VendorAgent"), None)
                    if agent:
                        res = agent.check_vendor(invoice)
                        results["vendor"] = res.__dict__ if hasattr(res, "__dict__") else res
                elif t == "CheckTotals":
                    agent = next((a for a in self.agents if a.__class__.__name__ == "TotalsAgent"), None)
                    if agent:
                        res = agent.check_totals(invoice)
                        results["totals"] = res.__dict__ if hasattr(res, "__dict__") else res
                elif t == "AnalyzePatterns":
                    agent = next((a for a in self.agents if a.__class__.__name__ == "PatternAgent"), None)
                    if agent:
                        res = agent.analyze_patterns(invoice)
                        results["patterns"] = res.__dict__ if hasattr(res, "__dict__") else res
            except Exception as e:
                results[t] = {"error": str(e)}

        return results
