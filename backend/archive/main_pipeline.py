#!/usr/bin/env python3
"""
Main fraud detection pipeline with multi-agent feedback loop
Enhanced with multi-key API support and robust error handling
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional

import dspy

from config import api_key_manager
from llm import get_configured_lm
from data_models import (
    Invoice, FraudDetectionResult, FraudDetectionSummary, 
    DataValidator, JSONParser
)
from agents import AgentCoordinator
from dspy_signatures import (
    create_dspy_modules, extract_tasks_from_response,
    extract_summary_from_response, extract_review_from_response
)

log = logging.getLogger("fraud_detection_pipeline")

class FraudDetectionPipeline:
    """Main pipeline for fraud detection with multi-agent coordination"""
    
    def __init__(self):
        self.agent_coordinator = AgentCoordinator()
        self.llm_working = False
        self.dspy_modules = None
        self.initialization_attempted = False
        
        log.info("Fraud Detection Pipeline initialized")
    
    def _initialize_llm(self) -> bool:
        """Initialize LLM and DSPy modules"""
        if self.initialization_attempted:
            return self.llm_working
        
        self.initialization_attempted = True
        
        try:
            # Initialize LLM
            lm = get_configured_lm()
            if lm is None:
                log.warning("Could not initialize LLM")
                return False
            
            # Configure DSPy
            dspy.configure(lm=lm)
            
            # Create DSPy modules
            self.dspy_modules = create_dspy_modules()
            if self.dspy_modules is None:
                log.warning("Could not create DSPy modules")
                return False
            
            self.llm_working = True
            log.info("LLM and DSPy modules initialized successfully")
            return True
            
        except Exception as e:
            log.warning(f"LLM initialization failed: {e}")
            self.llm_working = False
            return False
    
    def _fallback_plan_tasks(self, invoice: Invoice, known_error: str = "") -> List[str]:
        """Rule-based task planning when LLM fails"""
        tasks = ["CheckVendor", "CheckTotals"]  # Always check these
        
        # Add pattern analysis for complex invoices
        if len(invoice.items) > 1 or invoice.total > 500:
            tasks.append("AnalyzePatterns")
        
        # Adjust based on known errors
        if "vendor" in known_error.lower():
            if "CheckVendor" not in tasks:
                tasks.insert(0, "CheckVendor")
        elif "total" in known_error.lower():
            if "CheckTotals" not in tasks:
                tasks.insert(0, "CheckTotals")
        elif "pattern" in known_error.lower():
            if "AnalyzePatterns" not in tasks:
                tasks.append("AnalyzePatterns")
        
        log.info(f"Fallback planning: {tasks}")
        return tasks
    
    def _fallback_summarize(self, vendor_result: Dict, totals_result: Dict, 
                           patterns_result: Dict) -> FraudDetectionSummary:
        """Rule-based summarization when LLM fails"""
        
        risk_factors = []
        
        # Analyze vendor results
        if not vendor_result.get("vendor_valid", True):
            risk_factors.append("unknown vendor")
        
        # Analyze totals results
        if not totals_result.get("totals_match", True):
            difference = totals_result.get("difference", 0)
            if difference > 10:
                risk_factors.append("significant total mismatch")
            else:
                risk_factors.append("minor total mismatch")
        
        # Analyze pattern results
        anomalies = patterns_result.get("anomalies_found", 0)
        if anomalies > 0:
            if anomalies >= 3:
                risk_factors.append("multiple suspicious patterns")
            else:
                risk_factors.append("suspicious patterns detected")
        
        # Determine overall risk
        if len(risk_factors) >= 2 or "significant total mismatch" in risk_factors:
            fraud_risk = "HIGH"
        elif len(risk_factors) == 1:
            fraud_risk = "MEDIUM"
        else:
            fraud_risk = "LOW"
        
        # Create conclusion
        if risk_factors:
            conclusion = f"Risk factors detected: {', '.join(risk_factors)}"
        else:
            conclusion = "No significant risk factors detected"
        
        # Calculate confidence score
        confidence_score = 0.8 if not risk_factors else (0.9 if len(risk_factors) >= 2 else 0.7)
        
        return FraudDetectionSummary(
            fraud_risk=fraud_risk,
            conclusion=conclusion,
            risk_factors=risk_factors,
            vendor_valid=vendor_result.get("vendor_valid", True),
            totals_match=totals_result.get("totals_match", True),
            anomalies_found=anomalies,
            confidence_score=confidence_score
        )
    
    def _llm_plan_tasks(self, invoice: Invoice, known_error: str = "") -> Optional[List[str]]:
        """Use LLM to plan tasks"""
        if not self.llm_working or not self.dspy_modules:
            return None
        
        try:
            invoice_json = json.dumps(invoice.to_dict(), indent=2)
            
            # Try main planner
            plan_response = self.dspy_modules['planner'](
                invoice_json=invoice_json, 
                known_error=known_error
            )
            
            tasks = extract_tasks_from_response(plan_response)
            if tasks:
                log.info(f"LLM planning successful: {tasks}")
                return tasks
            
            # Try fallback planner
            log.info("Trying fallback planner...")
            fallback_response = self.dspy_modules['fallback_planner'](
                invoice_json=invoice_json,
                error_context=f"Main planner failed, known_error: {known_error}"
            )
            
            tasks = extract_tasks_from_response(fallback_response)
            if tasks:
                log.info(f"LLM fallback planning successful: {tasks}")
                return tasks
            
        except Exception as e:
            log.warning(f"LLM planning failed: {e}")
            # Mark current key as potentially problematic
            current_key = api_key_manager.get_current_key()
            if "429" in str(e) or "quota" in str(e).lower():
                api_key_manager.mark_key_failed(current_key, str(e))
        
        return None
    
    def _llm_validate_tasks(self, invoice: Invoice, tasks: List[str]) -> List[str]:
        """Use LLM to validate planned tasks"""
        if not self.llm_working or not self.dspy_modules:
            return tasks
        
        try:
            invoice_json = json.dumps(invoice.to_dict(), indent=2)
            
            validate_response = self.dspy_modules['validator'](
                invoice_json=invoice_json,
                proposed_tasks=json.dumps(tasks)
            )
            
            validated_tasks = extract_tasks_from_response(validate_response)
            if validated_tasks:
                log.info(f"LLM validation successful: {validated_tasks}")
                return validated_tasks
            
        except Exception as e:
            log.warning(f"LLM validation failed: {e}")
        
        return tasks
    
    def _llm_summarize(self, results: Dict[str, Any]) -> Optional[FraudDetectionSummary]:
        """Use LLM to summarize results"""
        if not self.llm_working or not self.dspy_modules:
            return None
        
        try:
            summary_response = self.dspy_modules['summarizer'](
                vendor_result=json.dumps(results["vendor"]),
                totals_result=json.dumps(results["totals"]),
                patterns_result=json.dumps(results["patterns"])
            )
            
            summary_dict = extract_summary_from_response(summary_response)
            
            if summary_dict and "fraud_risk" in summary_dict:
                # Convert to FraudDetectionSummary object
                summary = FraudDetectionSummary(
                    fraud_risk=summary_dict.get("fraud_risk", "LOW"),
                    conclusion=summary_dict.get("conclusion", "Analysis completed"),
                    risk_factors=summary_dict.get("risk_factors", []),
                    vendor_valid=results["vendor"].get("vendor_valid", True),
                    totals_match=results["totals"].get("totals_match", True),
                    anomalies_found=results["patterns"].get("anomalies_found", 0),
                    confidence_score=0.9  # High confidence for LLM analysis
                )
                
                log.info(f"LLM summarization successful: {summary.fraud_risk} risk")
                return summary
            
        except Exception as e:
            log.warning(f"LLM summarization failed: {e}")
        
        return None
    
    def _llm_review(self, invoice: Invoice, summary: FraudDetectionSummary) -> Dict[str, Any]:
        """Use LLM to review results"""
        if not self.llm_working or not self.dspy_modules:
            return {"status": "pass"}
        
        try:
            invoice_json = json.dumps(invoice.to_dict(), indent=2)
            summary_json = json.dumps(summary.to_dict(), indent=2)
            
            review_response = self.dspy_modules['reviewer'](
                invoice_json=invoice_json,
                summary=summary_json
            )
            
            review = extract_review_from_response(review_response)
            log.info(f"LLM review: {review}")
            return review
            
        except Exception as e:
            log.warning(f"LLM review failed: {e}")
        
        return {"status": "pass"}
    
    def run_detection(self, invoice_data: Dict[str, Any], max_iterations: int = 3) -> FraudDetectionResult:
        """Run the complete fraud detection pipeline"""
        
        start_time = time.time()
        
        # Validate and normalize input
        invoice = DataValidator.validate_invoice(invoice_data)
        if not invoice:
            return FraudDetectionResult(
                status="error",
                iterations=0,
                llm_working=False,
                results={},
                summary=FraudDetectionSummary(
                    fraud_risk="HIGH",
                    conclusion="Invalid invoice data",
                    risk_factors=["invalid_data"],
                    vendor_valid=False,
                    totals_match=False,
                    anomalies_found=1
                ),
                processing_time=time.time() - start_time
            )
        
        log.info(f"Starting fraud detection for invoice: {invoice.invoice_id}")
        
        # Initialize LLM if not already done
        self._initialize_llm()
        
        known_error = ""
        
        for iteration in range(1, max_iterations + 1):
            log.info(f"=== Iteration {iteration} ===")
            
            try:
                # Step 1: Plan tasks
                log.info("Step 1: Planning tasks...")
                planned_tasks = self._llm_plan_tasks(invoice, known_error)
                
                if not planned_tasks:
                    planned_tasks = self._fallback_plan_tasks(invoice, known_error)
                
                log.info(f"Planned tasks: {planned_tasks}")
                
                # Step 2: Validate tasks (if LLM available)
                log.info("Step 2: Validating tasks...")
                validated_tasks = self._llm_validate_tasks(invoice, planned_tasks)
                log.info(f"Validated tasks: {validated_tasks}")
                
                # Step 3: Execute specialist agents
                log.info("Step 3: Executing specialist agents...")
                results = self.agent_coordinator.execute_tasks(invoice, validated_tasks)
                
                # Step 4: Summarize results
                log.info("Step 4: Summarizing results...")
                summary = self._llm_summarize(results)
                
                if not summary:
                    summary = self._fallback_summarize(
                        results["vendor"], 
                        results["totals"], 
                        results["patterns"]
                    )
                
                log.info(f"Summary: {summary.fraud_risk} risk - {summary.conclusion}")
                
                # Step 5: Review results
                log.info("Step 5: Reviewing results...")
                review = self._llm_review(invoice, summary)
                
                # Check if review passed
                if review.get("status") == "pass":
                    processing_time = time.time() - start_time
                    
                    log.info(f"Analysis completed successfully in {processing_time:.2f}s!")
                    
                    return FraudDetectionResult(
                        status="completed",
                        iterations=iteration,
                        llm_working=self.llm_working,
                        results=results,
                        summary=summary,
                        review=review,
                        processing_time=processing_time
                    )
                else:
                    known_error = review.get("error", f"review_failed_{iteration}")
                    log.warning(f"Review failed: {known_error}. Retrying...")
                    
            except Exception as e:
                log.error(f"Error in iteration {iteration}: {e}")
                known_error = f"processing_error_{iteration}"
                
                # If this is due to API issues, try to reset
                if "429" in str(e) or "quota" in str(e).lower():
                    api_key_manager.reset_failures()
                    self.llm_working = False
                    self.initialization_attempted = False
        
        # If we reach here, all iterations failed
        log.warning("Max iterations reached, returning conservative analysis")
        
        # Generate conservative results using fallback methods
        conservative_tasks = self._fallback_plan_tasks(invoice)
        conservative_results = self.agent_coordinator.execute_tasks(invoice, conservative_tasks)
        conservative_summary = self._fallback_summarize(
            conservative_results["vendor"],
            conservative_results["totals"], 
            conservative_results["patterns"]
        )
        
        processing_time = time.time() - start_time
        
        return FraudDetectionResult(
            status="max_iterations_reached",
            iterations=max_iterations,
            llm_working=self.llm_working,
            results=conservative_results,
            summary=conservative_summary,
            processing_time=processing_time
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "pipeline_ready": True,
            "llm_working": self.llm_working,
            "available_api_keys": api_key_manager.get_available_count(),
            "total_api_keys": len(api_key_manager.api_keys),
            "current_key_index": api_key_manager.current_key_index + 1,
            "failed_keys": len(api_key_manager.failed_keys),
            "agent_status": self.agent_coordinator.get_agent_status(),
            "dspy_modules": self.dspy_modules is not None
        }

# Convenience functions for external use
_pipeline_instance = None

def get_pipeline() -> FraudDetectionPipeline:
    """Get global pipeline instance (singleton pattern)"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FraudDetectionPipeline()
    return _pipeline_instance

def run_fraud_detection(invoice_data: Dict[str, Any], max_iterations: int = 3) -> Dict[str, Any]:
    """Main entry point for fraud detection"""
    pipeline = get_pipeline()
    result = pipeline.run_detection(invoice_data, max_iterations)
    return result.to_dict()

def get_system_status() -> Dict[str, Any]:
    """Get system status"""
    pipeline = get_pipeline()
    return pipeline.get_system_status()

def reset_api_keys():
    """Reset API key failure tracking"""
    api_key_manager.reset_failures()
    log.info("API key failures reset")

def rotate_api_key():
    """Manually rotate to next API key"""
    old_key = api_key_manager.current_key_index + 1
    api_key_manager.rotate_key()
    new_key = api_key_manager.current_key_index + 1
    log.info(f"Manually rotated from key #{old_key} to key #{new_key}")

# Health check functions
def health_check() -> Dict[str, Any]:
    """Perform system health check"""
    try:
        pipeline = get_pipeline()
        status = pipeline.get_system_status()
        
        # Test with minimal invoice
        test_invoice = {
            "invoice_id": "HEALTH-CHECK",
            "vendor": "Test Vendor",
            "date": "2024-01-01",
            "items": [{"description": "Test item", "quantity": 1, "unit_price": 10.0}],
            "total": 10.0
        }
        
        # Run quick detection (1 iteration max)
        result = pipeline.run_detection(test_invoice, max_iterations=1)
        
        health_status = {
            "status": "healthy" if result.status in ["completed", "max_iterations_reached"] else "unhealthy",
            "system_info": status,
            "test_result": {
                "completed": result.status == "completed",
                "processing_time": result.processing_time,
                "llm_used": result.llm_working
            }
        }
        
        log.info(f"Health check: {health_status['status']}")
        return health_status
        
    except Exception as e:
        log.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "system_info": get_system_status() if _pipeline_instance else {}
        }

if __name__ == "__main__":
    # Quick test/demo
    print("Fraud Detection Pipeline - Quick Test")
    print("=" * 50)
    
    # Test invoice with issues
    test_invoice = {
        "invoice_id": "INV-2024-001",
        "vendor": "ACME Corp", 
        "date": "2024-09-26",
        "items": [
            {
                "description": "Office supplies",
                "quantity": 10,
                "unit_price": 15.50
            },
            {
                "description": "Gift cards",  # Suspicious
                "quantity": 5,
                "unit_price": 100.00
            },
            {
                "description": "Consulting services",
                "quantity": 20,
                "unit_price": 75.00
            }
        ],
        "total": 2200.00  # Should be 2155.00 (mismatch)
    }
    
    # Run detection
    result = run_fraud_detection(test_invoice)
    
    print("\nRESULTS:")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    
    # System status
    print(f"\nSYSTEM STATUS:")
    print("=" * 50)
    status = get_system_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print(f"\nCONCLUSION:")
    summary = result.get("summary", {})
    print(f"Fraud Risk: {summary.get('fraud_risk', 'UNKNOWN')}")
    print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
    print(f"LLM Working: {result.get('llm_working', False)}")
    print(f"Status: {result.get('status', 'UNKNOWN')}")