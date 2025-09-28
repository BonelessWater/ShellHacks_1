#!/usr/bin/env python3
"""
DSPy signature definitions for LLM-based reasoning in fraud detection
Enhanced with robust error handling and validation
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import dspy
from data_models import DataValidator, JSONParser

log = logging.getLogger("fraud_detection_signatures")


class PlanSignature(dspy.Signature):
    """Plan which fraud detection tasks to run based on the invoice.

    You must respond with ONLY a JSON array containing task names.
    Valid tasks: "CheckVendor", "CheckTotals", "AnalyzePatterns"

    Analysis Guidelines:
    - ALWAYS include "CheckVendor" and "CheckTotals" for basic validation
    - Include "AnalyzePatterns" for invoices with:
      * Multiple line items (>1)
      * High total values (>$500)
      * Complex vendor relationships
      * Previous anomaly indicators

    Consider the invoice content and any previous errors when planning:
    - If known_error mentions "vendor", prioritize CheckVendor
    - If known_error mentions "total" or "calculation", prioritize CheckTotals
    - If known_error mentions "pattern" or "anomaly", include AnalyzePatterns

    Example response: ["CheckVendor", "CheckTotals", "AnalyzePatterns"]
    """

    invoice_json: str = dspy.InputField(desc="Invoice data as JSON string")
    known_error: str = dspy.InputField(
        desc="Previous error to fix (if any)", default=""
    )

    tasks: str = dspy.OutputField(
        desc='Respond with ONLY a JSON array like: ["CheckVendor", "CheckTotals", "AnalyzePatterns"]'
    )


class ValidateSignature(dspy.Signature):
    """Validate and correct the planned tasks.

    You must respond with ONLY a JSON array containing valid task names.
    Valid tasks: "CheckVendor", "CheckTotals", "AnalyzePatterns"

    Validation Rules:
    - CheckVendor: ALWAYS include for unknown/new vendors
    - CheckTotals: ALWAYS include for invoices with line items
    - AnalyzePatterns: Include for:
      * High-value invoices (>$1000)
      * Invoices with many items (>5)
      * Invoices with suspicious characteristics
      * Complex vendor scenarios

    Quality Checks:
    - Ensure logical task combination
    - Verify all necessary validations are covered
    - Remove redundant or inappropriate tasks
    - Add missing critical tasks

    Example response: ["CheckVendor", "CheckTotals", "AnalyzePatterns"]
    """

    invoice_json: str = dspy.InputField(desc="Invoice data as JSON string")
    proposed_tasks: str = dspy.InputField(desc="Proposed tasks as JSON array")

    validated_tasks: str = dspy.OutputField(
        desc="Respond with ONLY a corrected JSON array of valid tasks"
    )


class SummarizeSignature(dspy.Signature):
    """Combine agent results into comprehensive fraud assessment.

    You must respond with ONLY a JSON object containing the summary.
    Required fields:
    - fraud_risk: "HIGH", "MEDIUM", or "LOW"
    - conclusion: string describing the overall assessment
    - risk_factors: array of identified risk factors

    Risk Assessment Framework:

    HIGH RISK indicators:
    - Unknown/invalid vendor
    - Significant total mismatches (>$10 or >10%)
    - Multiple suspicious patterns (≥3 anomalies)
    - Blacklisted vendors or items
    - Mathematical impossibilities

    MEDIUM RISK indicators:
    - Minor total discrepancies ($1-$10)
    - 1-2 suspicious patterns
    - Vendor with fuzzy match only
    - Unusual but explainable patterns

    LOW RISK indicators:
    - Approved vendor with exact match
    - Perfect or near-perfect total match
    - No suspicious patterns detected
    - All validations pass

    Example: {"fraud_risk": "HIGH", "conclusion": "Multiple red flags detected including unknown vendor and significant total mismatch", "risk_factors": ["unknown_vendor", "total_mismatch", "suspicious_patterns"]}
    """

    vendor_result: str = dspy.InputField(desc="Vendor check results as JSON")
    totals_result: str = dspy.InputField(desc="Totals check results as JSON")
    patterns_result: str = dspy.InputField(desc="Pattern analysis results as JSON")

    summary: str = dspy.OutputField(
        desc="Respond with ONLY a JSON object containing fraud_risk, conclusion, and risk_factors"
    )


class ReviewSignature(dspy.Signature):
    """Review final results for consistency, completeness, and logical coherence.

    You must respond with ONLY a JSON object.

    Review Criteria:

    CONSISTENCY CHECKS:
    - Risk level matches findings (HIGH risk needs significant issues)
    - Conclusion aligns with identified risk factors
    - Individual agent results support overall assessment

    COMPLETENESS CHECKS:
    - All critical fraud indicators addressed
    - Risk factors adequately explained
    - No missing analysis for high-risk scenarios

    LOGICAL COHERENCE:
    - No contradictory findings
    - Risk progression makes sense
    - Conclusion follows logically from evidence

    QUALITY STANDARDS:
    - Appropriate detail level for risk level
    - Clear, actionable conclusions
    - Professional fraud assessment standards

    For success: {"status": "pass"}
    For failure: {"status": "fail", "error": "specific_error_code"}

    Error codes:
    - "inconsistent_risk": Risk level doesn't match findings severity
    - "incomplete_analysis": Missing critical fraud indicators or analysis
    - "invalid_format": Response format issues or malformed data
    - "contradictory_findings": Conflicting results between agents
    - "insufficient_evidence": Risk level not supported by evidence
    - "missing_risk_factors": High/medium risk without identified factors
    """

    invoice_json: str = dspy.InputField(desc="Original invoice data")
    summary: str = dspy.InputField(desc="Analysis summary as JSON")

    review: str = dspy.OutputField(
        desc='Respond with ONLY JSON: {"status": "pass"} or {"status": "fail", "error": "error_code"}'
    )


class ErrorAnalysisSignature(dspy.Signature):
    """Analyze processing errors and suggest specific corrections.

    You must respond with ONLY a JSON object.

    Error Analysis Process:
    1. Identify root cause of the error
    2. Determine impact on fraud detection accuracy
    3. Suggest specific corrective actions
    4. Recommend task adjustments for retry

    Common Error Categories:
    - "data_validation": Invalid or malformed invoice data
    - "calculation_error": Mathematical computation issues
    - "vendor_lookup": Vendor validation problems
    - "pattern_analysis": Anomaly detection failures
    - "llm_parsing": LLM response parsing issues
    - "system_error": Technical/infrastructure problems

    Response format:
    {
      "error_type": "category",
      "root_cause": "specific issue description",
      "impact": "effect on analysis accuracy",
      "suggestion": "specific corrective action",
      "retry_tasks": ["task1", "task2"],
      "priority": "HIGH|MEDIUM|LOW"
    }
    """

    invoice_json: str = dspy.InputField(desc="Invoice data that caused the error")
    error_message: str = dspy.InputField(desc="Error message or description")
    failed_step: str = dspy.InputField(
        desc="Which step failed (planning, validation, execution, etc.)"
    )

    analysis: str = dspy.OutputField(
        desc="JSON object with comprehensive error analysis and suggestions"
    )


class FallbackPlanSignature(dspy.Signature):
    """Generate a conservative, comprehensive fallback plan when primary planning fails.

    You must respond with ONLY a JSON array containing task names.

    Fallback Strategy:
    When primary planning fails, we must err on the side of thoroughness to ensure
    no fraud indicators are missed. This is a safety-first approach.

    Default Comprehensive Tasks:
    - ALWAYS include "CheckVendor" (vendor validation is critical)
    - ALWAYS include "CheckTotals" (mathematical accuracy is essential)
    - ALWAYS include "AnalyzePatterns" (pattern detection catches subtle fraud)

    Exception Handling:
    - If specific error context suggests vendor issues → prioritize CheckVendor
    - If calculation errors mentioned → ensure CheckTotals is first
    - If anomaly detection failed → emphasize AnalyzePatterns

    This comprehensive approach sacrifices efficiency for completeness,
    ensuring robust fraud detection even when AI planning components fail.

    Example response: ["CheckVendor", "CheckTotals", "AnalyzePatterns"]
    """

    invoice_json: str = dspy.InputField(desc="Invoice data as JSON string")
    error_context: str = dspy.InputField(desc="Context about why fallback is needed")

    fallback_tasks: str = dspy.OutputField(
        desc="Respond with ONLY a comprehensive JSON array of tasks ensuring complete fraud coverage"
    )


class ConfidenceSignature(dspy.Signature):
    """Assess confidence level in fraud detection results.

    You must respond with ONLY a JSON object.

    Confidence Assessment Factors:

    HIGH CONFIDENCE (0.8-1.0):
    - Clear, unambiguous indicators
    - Multiple agent agreement
    - Well-established fraud patterns
    - Sufficient data quality

    MEDIUM CONFIDENCE (0.5-0.8):
    - Some ambiguous indicators
    - Mixed agent results
    - Incomplete data
    - Edge case scenarios

    LOW CONFIDENCE (0.0-0.5):
    - Highly ambiguous results
    - Conflicting agent outputs
    - Poor data quality
    - Novel/unknown patterns

    Response format:
    {
      "confidence_score": 0.85,
      "confidence_level": "HIGH",
      "factors": ["clear_indicators", "agent_agreement"],
      "concerns": ["minor_data_gaps"],
      "recommendation": "Accept results with normal review"
    }
    """

    summary: str = dspy.InputField(desc="Fraud detection summary as JSON")
    agent_results: str = dspy.InputField(desc="All agent results as JSON")

    confidence: str = dspy.OutputField(
        desc="JSON object with confidence assessment and factors"
    )


# DSPy Module Management
class DSPyModuleManager:
    """Manages DSPy modules with error handling and validation"""

    def __init__(self):
        self.modules = {}
        self.initialized = False
        self.last_error = None

    def initialize_modules(self) -> bool:
        """Initialize all DSPy modules"""
        try:
            self.modules = {
                "planner": dspy.Predict(PlanSignature),
                "validator": dspy.Predict(ValidateSignature),
                "summarizer": dspy.Predict(SummarizeSignature),
                "reviewer": dspy.Predict(ReviewSignature),
                "error_analyzer": dspy.Predict(ErrorAnalysisSignature),
                "fallback_planner": dspy.Predict(FallbackPlanSignature),
                "confidence_assessor": dspy.Predict(ConfidenceSignature),
            }

            self.initialized = True
            self.last_error = None
            log.info("✅ DSPy modules initialized successfully")
            return True

        except Exception as e:
            self.last_error = str(e)
            self.initialized = False
            log.error(f"❌ Failed to initialize DSPy modules: {e}")
            return False

    def get_module(self, module_name: str):
        """Get a specific DSPy module"""
        if not self.initialized:
            log.warning("DSPy modules not initialized")
            return None

        return self.modules.get(module_name)

    def is_available(self) -> bool:
        """Check if modules are available"""
        return self.initialized and bool(self.modules)

    def get_status(self) -> Dict[str, Any]:
        """Get module status information"""
        return {
            "initialized": self.initialized,
            "modules_count": len(self.modules),
            "available_modules": list(self.modules.keys()) if self.initialized else [],
            "last_error": self.last_error,
        }


# Global module manager instance
module_manager = DSPyModuleManager()


# Module Creation Functions
def create_dspy_modules() -> Optional[Dict[str, Any]]:
    """Create and return DSPy module instances"""
    if module_manager.initialize_modules():
        return module_manager.modules
    return None


def get_dspy_module(module_name: str):
    """Get a specific DSPy module"""
    return module_manager.get_module(module_name)


def reinitialize_modules() -> bool:
    """Reinitialize all modules"""
    return module_manager.initialize_modules()


# Response Processing Functions
def extract_tasks_from_response(response: Any) -> List[str]:
    """Extract and validate task list from DSPy response"""
    try:
        # Handle different response types
        text = None
        if hasattr(response, "tasks"):
            text = response.tasks
        elif hasattr(response, "validated_tasks"):
            text = response.validated_tasks
        elif hasattr(response, "fallback_tasks"):
            text = response.fallback_tasks
        else:
            text = str(response)

        # Parse and validate
        parsed = JSONParser.safe_json_parse(text, [])
        validated = DataValidator.validate_tasks(parsed)

        if not validated:
            log.warning(f"No valid tasks extracted from response: {text[:100]}")
            return ["CheckVendor", "CheckTotals"]  # Safe fallback

        return validated

    except Exception as e:
        log.error(f"Error extracting tasks from response: {e}")
        return ["CheckVendor", "CheckTotals"]  # Safe fallback


def extract_summary_from_response(response: Any) -> Dict[str, Any]:
    """Extract and validate summary dict from DSPy response"""
    try:
        text = None
        if hasattr(response, "summary"):
            text = response.summary
        else:
            text = str(response)

        parsed = JSONParser.safe_json_parse(text, {})

        # Validate required fields
        if not isinstance(parsed, dict):
            log.warning("Summary response is not a dictionary")
            return {}

        # Ensure required fields exist
        if "fraud_risk" not in parsed:
            parsed["fraud_risk"] = "MEDIUM"

        if "conclusion" not in parsed:
            parsed["conclusion"] = "Analysis completed with limited data"

        if "risk_factors" not in parsed:
            parsed["risk_factors"] = []

        # Validate fraud_risk value
        parsed["fraud_risk"] = DataValidator.validate_risk_level(parsed["fraud_risk"])

        return parsed

    except Exception as e:
        log.error(f"Error extracting summary from response: {e}")
        return {
            "fraud_risk": "MEDIUM",
            "conclusion": "Error in summary extraction",
            "risk_factors": ["processing_error"],
        }


def extract_review_from_response(response: Any) -> Dict[str, Any]:
    """Extract and validate review dict from DSPy response"""
    try:
        text = None
        if hasattr(response, "review"):
            text = response.review
        else:
            text = str(response)

        parsed = JSONParser.safe_json_parse(text, {"status": "pass"})

        # Validate format
        if not isinstance(parsed, dict):
            return {"status": "pass"}

        # Ensure status field exists and is valid
        status = parsed.get("status", "pass").lower()
        if status not in ["pass", "fail"]:
            status = "pass"

        result = {"status": status}

        # Include error if status is fail
        if status == "fail" and "error" in parsed:
            result["error"] = str(parsed["error"])

        return result

    except Exception as e:
        log.error(f"Error extracting review from response: {e}")
        return {"status": "pass"}


def extract_confidence_from_response(response: Any) -> Dict[str, Any]:
    """Extract confidence assessment from DSPy response"""
    try:
        text = None
        if hasattr(response, "confidence"):
            text = response.confidence
        else:
            text = str(response)

        parsed = JSONParser.safe_json_parse(text, {})

        # Provide defaults
        default_confidence = {
            "confidence_score": 0.7,
            "confidence_level": "MEDIUM",
            "factors": ["analysis_completed"],
            "concerns": [],
            "recommendation": "Review results",
        }

        if not isinstance(parsed, dict):
            return default_confidence

        # Validate and normalize
        result = default_confidence.copy()
        result.update(parsed)

        # Ensure confidence_score is valid
        try:
            score = float(result["confidence_score"])
            result["confidence_score"] = max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            result["confidence_score"] = 0.7

        # Validate confidence_level
        if result["confidence_level"] not in ["HIGH", "MEDIUM", "LOW"]:
            result["confidence_level"] = "MEDIUM"

        return result

    except Exception as e:
        log.error(f"Error extracting confidence from response: {e}")
        return {
            "confidence_score": 0.5,
            "confidence_level": "LOW",
            "factors": ["extraction_error"],
            "concerns": ["processing_error"],
            "recommendation": "Manual review required",
        }


# Validation Functions
def validate_signature_response(
    signature_name: str, response: str, expected_type: type = dict
) -> bool:
    """Validate that a signature response meets expectations"""
    try:
        parsed = JSONParser.extract_json_from_text(response)

        if parsed is None:
            log.warning(f"⚠️ {signature_name}: Could not parse JSON from response")
            return False

        if expected_type == list and not isinstance(parsed, list):
            log.warning(f"⚠️ {signature_name}: Expected list, got {type(parsed)}")
            return False

        if expected_type == dict and not isinstance(parsed, dict):
            log.warning(f"⚠️ {signature_name}: Expected dict, got {type(parsed)}")
            return False

        # Additional validation based on signature type
        if signature_name == "PlanSignature" or signature_name == "ValidateSignature":
            if expected_type == list:
                valid_tasks = {"CheckVendor", "CheckTotals", "AnalyzePatterns"}
                if not all(task in valid_tasks for task in parsed):
                    log.warning(f"⚠️ {signature_name}: Invalid tasks found")
                    return False

        return True

    except Exception as e:
        log.warning(f"⚠️ {signature_name}: Validation error - {e}")
        return False


def get_signature_status() -> Dict[str, Any]:
    """Get status of signature definitions and modules"""
    return {
        "signatures_defined": 6,
        "signature_list": [
            "PlanSignature",
            "ValidateSignature",
            "SummarizeSignature",
            "ReviewSignature",
            "ErrorAnalysisSignature",
            "FallbackPlanSignature",
            "ConfidenceSignature",
        ],
        "module_manager": module_manager.get_status(),
        "available": module_manager.is_available(),
    }


# Testing and Debugging Functions
def test_signature_response_parsing():
    """Test response parsing with various formats"""
    test_cases = [
        # Valid JSON responses
        ('["CheckVendor", "CheckTotals"]', list, True),
        ('{"fraud_risk": "HIGH", "conclusion": "Test"}', dict, True),
        ('{"status": "pass"}', dict, True),
        # JSON in code blocks
        ('```json\n["CheckVendor"]\n```', list, True),
        ('```\n{"fraud_risk": "LOW"}\n```', dict, True),
        # Invalid formats
        ("Not JSON at all", dict, False),
        ('["InvalidTask"]', list, False),  # Invalid task name
        ("", dict, False),
    ]

    results = []
    for response, expected_type, should_pass in test_cases:
        try:
            if expected_type == list:
                parsed = extract_tasks_from_response(
                    type("Response", (), {"tasks": response})()
                )
                success = bool(parsed) == should_pass
            else:
                success = (
                    validate_signature_response("Test", response, expected_type)
                    == should_pass
                )

            results.append(
                {
                    "input": response[:50],
                    "expected_type": expected_type.__name__,
                    "should_pass": should_pass,
                    "actual_pass": success,
                    "test_passed": success == should_pass,
                }
            )
        except Exception as e:
            results.append(
                {
                    "input": response[:50],
                    "expected_type": expected_type.__name__,
                    "should_pass": should_pass,
                    "error": str(e),
                    "test_passed": False,
                }
            )

    return results


def create_sample_responses():
    """Create sample responses for testing"""
    return {
        "plan_response": '["CheckVendor", "CheckTotals", "AnalyzePatterns"]',
        "validate_response": '["CheckVendor", "CheckTotals"]',
        "summary_response": """{
            "fraud_risk": "HIGH",
            "conclusion": "Multiple risk factors detected including unknown vendor and total mismatch",
            "risk_factors": ["unknown_vendor", "total_mismatch", "suspicious_patterns"]
        }""",
        "review_pass": '{"status": "pass"}',
        "review_fail": '{"status": "fail", "error": "inconsistent_risk"}',
        "confidence_response": """{
            "confidence_score": 0.85,
            "confidence_level": "HIGH", 
            "factors": ["clear_indicators", "agent_agreement"],
            "concerns": [],
            "recommendation": "Accept results"
        }""",
        "error_analysis": """{
            "error_type": "data_validation",
            "root_cause": "Missing required invoice fields",
            "impact": "Cannot perform complete fraud analysis",
            "suggestion": "Request complete invoice data",
            "retry_tasks": ["CheckVendor", "CheckTotals"],
            "priority": "HIGH"
        }""",
    }


# Advanced Response Processing
def process_llm_response_with_fallback(
    response: Any, signature_type: str, fallback_value: Any = None
) -> Any:
    """Process LLM response with intelligent fallback handling"""
    try:
        if (
            signature_type == "plan"
            or signature_type == "validate"
            or signature_type == "fallback"
        ):
            return extract_tasks_from_response(response)

        elif signature_type == "summary":
            return extract_summary_from_response(response)

        elif signature_type == "review":
            return extract_review_from_response(response)

        elif signature_type == "confidence":
            return extract_confidence_from_response(response)

        elif signature_type == "error_analysis":
            # Extract error analysis
            text = getattr(response, "analysis", str(response))
            parsed = JSONParser.safe_json_parse(text, {})

            # Provide structured fallback
            if not isinstance(parsed, dict):
                return {
                    "error_type": "unknown",
                    "root_cause": "Could not analyze error",
                    "suggestion": "Manual review required",
                    "retry_tasks": ["CheckVendor", "CheckTotals"],
                    "priority": "MEDIUM",
                }

            return parsed

        else:
            log.warning(f"Unknown signature type: {signature_type}")
            return fallback_value

    except Exception as e:
        log.error(f"Error processing {signature_type} response: {e}")

        # Return appropriate fallback based on signature type
        fallbacks = {
            "plan": ["CheckVendor", "CheckTotals"],
            "validate": ["CheckVendor", "CheckTotals"],
            "fallback": ["CheckVendor", "CheckTotals", "AnalyzePatterns"],
            "summary": {
                "fraud_risk": "MEDIUM",
                "conclusion": "Error in analysis processing",
                "risk_factors": ["processing_error"],
            },
            "review": {"status": "pass"},
            "confidence": {
                "confidence_score": 0.5,
                "confidence_level": "LOW",
                "factors": ["processing_error"],
                "concerns": ["llm_response_error"],
                "recommendation": "Manual review required",
            },
            "error_analysis": {
                "error_type": "processing_error",
                "root_cause": "LLM response processing failed",
                "suggestion": "Use fallback analysis methods",
                "retry_tasks": ["CheckVendor", "CheckTotals"],
                "priority": "HIGH",
            },
        }

        return fallbacks.get(signature_type, fallback_value)


def validate_fraud_risk_consistency(
    summary: Dict[str, Any], agent_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate that fraud risk assessment is consistent with agent findings"""
    try:
        fraud_risk = summary.get("fraud_risk", "MEDIUM")
        risk_factors = summary.get("risk_factors", [])

        # Check vendor results
        vendor_result = agent_results.get("vendor", {})
        vendor_valid = vendor_result.get("vendor_valid", True)
        vendor_risk = vendor_result.get("risk_factor", "LOW")

        # Check totals results
        totals_result = agent_results.get("totals", {})
        totals_match = totals_result.get("totals_match", True)
        totals_risk = totals_result.get("risk_factor", "LOW")

        # Check patterns results
        patterns_result = agent_results.get("patterns", {})
        anomalies_found = patterns_result.get("anomalies_found", 0)
        patterns_risk = patterns_result.get("risk_factor", "LOW")

        # Consistency checks
        issues = []

        # High risk should have supporting evidence
        if fraud_risk == "HIGH":
            high_risk_indicators = [
                vendor_risk == "HIGH",
                totals_risk == "HIGH",
                patterns_risk == "HIGH",
                not vendor_valid,
                not totals_match,
                anomalies_found >= 3,
            ]

            if not any(high_risk_indicators):
                issues.append("HIGH fraud risk not supported by agent findings")

        # Low risk shouldn't have high-risk indicators
        if fraud_risk == "LOW":
            low_risk_conflicts = [
                vendor_risk == "HIGH",
                totals_risk == "HIGH",
                patterns_risk == "HIGH",
            ]

            if any(low_risk_conflicts):
                issues.append("LOW fraud risk conflicts with high-risk agent findings")

        # Risk factors should match findings
        expected_factors = []
        if not vendor_valid:
            expected_factors.append("unknown_vendor")
        if not totals_match:
            expected_factors.append("total_mismatch")
        if anomalies_found > 0:
            expected_factors.append("suspicious_patterns")

        missing_factors = [f for f in expected_factors if f not in risk_factors]
        if missing_factors:
            issues.append(f"Missing risk factors: {missing_factors}")

        return {
            "consistent": len(issues) == 0,
            "issues": issues,
            "recommendations": [
                (
                    "Adjust fraud risk level to match findings"
                    if issues
                    else "Risk assessment is consistent"
                ),
                (
                    "Update risk factors to reflect agent results"
                    if missing_factors
                    else "Risk factors are complete"
                ),
            ],
        }

    except Exception as e:
        log.error(f"Error validating fraud risk consistency: {e}")
        return {
            "consistent": False,
            "issues": [f"Validation error: {e}"],
            "recommendations": ["Manual review required due to validation error"],
        }


# Export main classes and functions
__all__ = [
    "PlanSignature",
    "ValidateSignature",
    "SummarizeSignature",
    "ReviewSignature",
    "ErrorAnalysisSignature",
    "FallbackPlanSignature",
    "ConfidenceSignature",
    "DSPyModuleManager",
    "module_manager",
    "create_dspy_modules",
    "get_dspy_module",
    "reinitialize_modules",
    "extract_tasks_from_response",
    "extract_summary_from_response",
    "extract_review_from_response",
    "extract_confidence_from_response",
    "validate_signature_response",
    "process_llm_response_with_fallback",
    "validate_fraud_risk_consistency",
    "get_signature_status",
    "test_signature_response_parsing",
]
