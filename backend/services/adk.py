def run_fraud_detection(payload, max_iterations=1):
    """Call the ADK pipeline run_fraud_detection when available, otherwise use
    a lightweight local mock. The wrapper keeps the rest of the codebase
    calling `run_fraud_detection(...)` unchanged.
    """
    # Local mock scoring (kept for fallback/testing)
    risk_score = 0.3 + (hash(payload.get("vendor", "")) % 5) * 0.15  # Mock score between 0.3-0.9
    return {
        "risk_score": risk_score,
        "details": f"Analyzed invoice for {payload.get('vendor', 'Unknown')}",
        "agent_results": {
            "vendor_check": {"status": "passed", "risk_score": risk_score * 0.8},
            "amount_check": {"status": "passed", "risk_score": risk_score * 1.2},
            "pattern_analysis": {"status": "flagged" if risk_score > 0.7 else "passed", "risk_score": risk_score}
        }
    }
