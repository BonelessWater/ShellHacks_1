#!/usr/bin/env python3
"""
Simplified Real-time Invoice Fraud Detection with Multi-Agent Feedback Loop
- Google Gemini API (direct integration)
- DSPy.ai for structured prompting
- Feedback loop for error correction
- Simple JSON parsing with fallbacks

Requirements:
  pip install google-generativeai dspy-ai pydantic

Environment:
  export GOOGLE_API_KEY="your_api_key_here"
"""

import os
import json
import logging
import re
import time
import random
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("fraud_detection")

# Gemini setup
import google.generativeai as genai
import dspy

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Please export it before running.")

genai.configure(api_key=GOOGLE_API_KEY)

class SimpleGeminiLM(dspy.LM):
    """Simplified Gemini LM with rate limiting and better error handling"""
    
    def __init__(self, model_name="gemini-1.5-flash"):
        super().__init__(model=model_name)
        self.model_name = model_name
        self.model = None
        self.last_call_time = 0
        self.min_delay = 1.0  # Minimum delay between calls (seconds)
        self.max_retries = 3
        
        # Try models that are more likely to work with free tier
        model_candidates = [
            "gemini-1.5-flash-8b",  # Often available on free tier
            "models/gemini-1.5-flash-8b",
            "gemini-1.5-flash",
            "models/gemini-1.5-flash", 
            "gemini-pro",
            "models/gemini-pro"
        ]
        
        # Don't initialize during __init__ to avoid quota issues
        # We'll initialize on first call instead
        self.model_candidates = model_candidates
        self.initialized = False
        log.info("SimpleGeminiLM created - will initialize on first call")
    
    def _try_initialize_model(self):
        """Try to initialize a working model"""
        if self.initialized:
            return True
            
        last_error = None
        for candidate in self.model_candidates:
            try:
                log.info(f"Trying to initialize model: {candidate}")
                test_model = genai.GenerativeModel(candidate)
                
                # Very simple test with minimal tokens
                test_response = test_model.generate_content(
                    "Hi",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=5
                    )
                )
                
                if test_response and test_response.text:
                    self.model = test_model
                    self.model_name = candidate
                    self.initialized = True
                    log.info(f"✅ Successfully initialized: {candidate}")
                    return True
                    
            except Exception as e:
                error_msg = str(e)
                last_error = e
                
                if "429" in error_msg or "quota" in error_msg.lower():
                    log.warning(f"❌ Quota exceeded for {candidate}")
                    # Extract retry delay if available
                    if "retry in" in error_msg:
                        try:
                            delay_match = re.search(r'retry in (\d+\.?\d*)', error_msg)
                            if delay_match:
                                delay = float(delay_match.group(1))
                                log.info(f"⏳ Waiting {delay:.1f}s before trying next model...")
                                time.sleep(min(delay, 5))  # Cap at 5 seconds
                        except:
                            time.sleep(2)
                elif "404" in error_msg:
                    log.warning(f"❌ Model not found: {candidate}")
                else:
                    log.warning(f"❌ Failed to initialize {candidate}: {error_msg[:100]}")
                
                # Small delay between attempts
                time.sleep(0.5)
                continue
        
        log.error(f"❌ Could not initialize any model. Last error: {last_error}")
        return False
    
    def _rate_limit_delay(self):
        """Add delay to respect rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            log.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        # Try to initialize if not done yet
        if not self.initialized:
            if not self._try_initialize_model():
                log.error("❌ No working Gemini model available")
                return "Error: No working model available"
        
        if messages and not prompt:
            if isinstance(messages, list):
                prompt = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages])
            else:
                prompt = str(messages)
        
        if not prompt:
            return "Error: No prompt provided"
        
        # Rate limiting
        self._rate_limit_delay()
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=500,  # Reduced to save quota
                    )
                )
                
                if response and response.text:
                    result = response.text.strip()
                    log.debug(f"✅ Gemini response ({self.model_name}): {result[:100]}...")
                    return result
                else:
                    log.warning("Empty response from Gemini")
                    return "Error: Empty response"
                    
            except Exception as e:
                error_msg = str(e)
                
                if "429" in error_msg or "quota" in error_msg.lower():
                    # Extract retry delay
                    delay = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff with jitter
                    
                    if "retry in" in error_msg:
                        try:
                            delay_match = re.search(r'retry in (\d+\.?\d*)', error_msg)
                            if delay_match:
                                delay = max(delay, float(delay_match.group(1)))
                        except:
                            pass
                    
                    if attempt < self.max_retries - 1:
                        log.warning(f"⏳ Rate limited. Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        log.error("❌ Max retries reached for rate limiting")
                        return "Error: Rate limit exceeded"
                
                elif "404" in error_msg:
                    log.error(f"❌ Model not found: {self.model_name}")
                    return "Error: Model not available"
                
                else:
                    log.error(f"❌ Gemini API error: {error_msg}")
                    if attempt < self.max_retries - 1:
                        delay = 1 + (attempt * 0.5)
                        log.info(f"⏳ Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    return f"Error: {error_msg[:100]}"
        
        return "Error: Max retries exceeded"

# Configure DSPy with error handling
try:
    dspy.configure(lm=SimpleGeminiLM())
    log.info("✅ DSPy configured successfully")
except Exception as e:
    log.warning(f"⚠️  DSPy configuration deferred due to: {e}")
    # We'll handle this in the pipeline

# DSPy Signatures with cleaner output formatting and explicit JSON instructions
class PlanSignature(dspy.Signature):
    """Plan which fraud detection tasks to run based on the invoice.
    
    You must respond with ONLY a JSON array containing task names.
    Valid tasks: "CheckVendor", "CheckTotals", "AnalyzePatterns"
    
    Example response: ["CheckVendor", "CheckTotals"]
    """
    
    invoice_json: str = dspy.InputField(desc="Invoice data as JSON string")
    known_error: str = dspy.InputField(desc="Previous error to fix (if any)", default="")
    
    tasks: str = dspy.OutputField(desc="Respond with ONLY a JSON array like: [\"CheckVendor\", \"CheckTotals\", \"AnalyzePatterns\"]")

class ValidateSignature(dspy.Signature):
    """Validate and correct the planned tasks.
    
    You must respond with ONLY a JSON array containing valid task names.
    Valid tasks: "CheckVendor", "CheckTotals", "AnalyzePatterns"
    
    Example response: ["CheckVendor", "CheckTotals"]
    """
    
    invoice_json: str = dspy.InputField(desc="Invoice data as JSON string")
    proposed_tasks: str = dspy.InputField(desc="Proposed tasks as JSON array")
    
    validated_tasks: str = dspy.OutputField(desc="Respond with ONLY a corrected JSON array of valid tasks")

class SummarizeSignature(dspy.Signature):
    """Combine agent results into fraud assessment.
    
    You must respond with ONLY a JSON object containing the summary.
    Required fields: fraud_risk (HIGH/LOW), conclusion (string)
    
    Example: {"fraud_risk": "HIGH", "conclusion": "Multiple red flags detected"}
    """
    
    vendor_result: str = dspy.InputField(desc="Vendor check results")
    totals_result: str = dspy.InputField(desc="Totals check results")
    patterns_result: str = dspy.InputField(desc="Pattern analysis results")
    
    summary: str = dspy.OutputField(desc="Respond with ONLY a JSON object containing fraud_risk and conclusion")

class ReviewSignature(dspy.Signature):
    """Review final results for errors.
    
    You must respond with ONLY a JSON object.
    For success: {"status": "pass"}
    For failure: {"status": "fail", "error": "specific_error_code"}
    """
    
    invoice_json: str = dspy.InputField(desc="Original invoice data")
    summary: str = dspy.InputField(desc="Analysis summary")
    
    review: str = dspy.OutputField(desc="Respond with ONLY JSON: {\"status\": \"pass\"} or {\"status\": \"fail\", \"error\": \"error_code\"}")

# Initialize DSPy modules
planner = dspy.Predict(PlanSignature)
validator = dspy.Predict(ValidateSignature)
summarizer = dspy.Predict(SummarizeSignature)
reviewer = dspy.Predict(ReviewSignature)

# Specialist Agents
def check_vendor(invoice: Dict[str, Any]) -> Dict[str, Any]:
    """Check if vendor is in approved list"""
    vendor = invoice.get("vendor", "")
    approved_vendors = {"ACME Corp", "Beta Industries", "Delta LLC", "Gamma Tech"}
    
    is_valid = vendor in approved_vendors
    result = {
        "vendor": vendor,
        "vendor_valid": is_valid,
        "risk_factor": "LOW" if is_valid else "HIGH"
    }
    
    log.info(f"Vendor check: {vendor} -> {'VALID' if is_valid else 'INVALID'}")
    return result

def check_totals(invoice: Dict[str, Any]) -> Dict[str, Any]:
    """Verify invoice total matches line items"""
    reported_total = invoice.get("total", 0)
    items = invoice.get("items", [])
    
    calculated_total = 0
    for item in items:
        qty = item.get("quantity", 0)
        price = item.get("unit_price", 0)
        calculated_total += qty * price
    
    difference = abs(reported_total - calculated_total)
    matches = difference < 0.01  # Allow for rounding
    
    result = {
        "reported_total": reported_total,
        "calculated_total": calculated_total,
        "difference": difference,
        "totals_match": matches,
        "risk_factor": "LOW" if matches else "HIGH"
    }
    
    log.info(f"Totals check: {reported_total} vs {calculated_total} -> {'MATCH' if matches else 'MISMATCH'}")
    return result

def analyze_patterns(invoice: Dict[str, Any]) -> Dict[str, Any]:
    """Look for suspicious patterns in the invoice"""
    items = invoice.get("items", [])
    anomalies = []
    
    for item in items:
        description = str(item.get("description", "")).lower()
        quantity = item.get("quantity", 0)
        unit_price = item.get("unit_price", 0)
        line_total = quantity * unit_price
        
        # Check for suspicious keywords
        suspicious_words = ["gift", "cash", "tip", "bonus", "personal"]
        if any(word in description for word in suspicious_words):
            anomalies.append(f"Suspicious item: {item.get('description')}")
        
        # Check for unusually high amounts
        if line_total > 1000:
            anomalies.append(f"High-value item: {item.get('description')} (${line_total})")
        
        # Check for unusual quantities
        if quantity > 100:
            anomalies.append(f"High quantity: {quantity} of {item.get('description')}")
    
    result = {
        "anomalies_found": len(anomalies),
        "anomaly_details": anomalies,
        "risk_factor": "HIGH" if anomalies else "LOW"
    }
    
    log.info(f"Pattern analysis: {len(anomalies)} anomalies found")
    return result

# Fallback functions for when DSPy fails
def fallback_plan_tasks(invoice: Dict[str, Any], known_error: str = "") -> List[str]:
    """Simple rule-based task planning when LLM fails"""
    tasks = []
    
    # Always check vendor and totals
    tasks.extend(["CheckVendor", "CheckTotals"])
    
    # Add pattern analysis for invoices with multiple items or high values
    items = invoice.get("items", [])
    total = invoice.get("total", 0)
    
    if len(items) > 1 or total > 500:
        tasks.append("AnalyzePatterns")
    
    # If there was a specific error, adjust tasks
    if "vendor" in known_error.lower():
        if "CheckVendor" not in tasks:
            tasks.insert(0, "CheckVendor")
    elif "total" in known_error.lower():
        if "CheckTotals" not in tasks:
            tasks.insert(0, "CheckTotals")
    
    return tasks

def fallback_summarize(vendor_result: Dict, totals_result: Dict, patterns_result: Dict) -> Dict[str, Any]:
    """Simple rule-based summarization when LLM fails"""
    
    # Determine overall risk
    risk_factors = []
    
    if not vendor_result.get("vendor_valid", True):
        risk_factors.append("unknown vendor")
    
    if not totals_result.get("totals_match", True):
        risk_factors.append("total mismatch")
    
    if patterns_result.get("anomalies_found", 0) > 0:
        risk_factors.append("suspicious patterns")
    
    fraud_risk = "HIGH" if risk_factors else "LOW"
    
    if risk_factors:
        conclusion = f"Risk factors detected: {', '.join(risk_factors)}"
    else:
        conclusion = "No significant risk factors detected"
    
    return {
        "fraud_risk": fraud_risk,
        "conclusion": conclusion,
        "risk_factors": risk_factors,
        "vendor_valid": vendor_result.get("vendor_valid", True),
        "totals_match": totals_result.get("totals_match", True),
        "anomalies_found": patterns_result.get("anomalies_found", 0)
    }
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
        r'```json\s*([^`]+)\s*```',  # Code block
        r'```\s*([^`]+)\s*```',     # Generic code block
        r'(\[.*?\])',               # Array anywhere in text
        r'(\{.*?\})',               # Object anywhere in text
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
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if (line.startswith('[') and line.endswith(']')) or (line.startswith('{') and line.endswith('}')):
            try:
                return json.loads(line)
            except:
                continue
    
    return None

def safe_json_parse(text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback"""
    result = extract_json_from_text(text)
    if result is not None:
        return result
    
    log.warning(f"Could not parse JSON from: {text[:100]}...")
    return default

# Main Pipeline
def run_fraud_detection(invoice: Dict[str, Any], max_iterations: int = 3) -> Dict[str, Any]:
    """Run the complete fraud detection pipeline with feedback loop"""
    
    invoice_json = json.dumps(invoice, indent=2)
    known_error = ""
    llm_working = True
    
    log.info(f"🔍 Starting fraud detection for invoice: {invoice.get('invoice_id', 'unknown')}")
    
    # Try to initialize DSPy if not already done
    try:
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            log.info("🔧 Initializing DSPy LM...")
            dspy.configure(lm=SimpleGeminiLM())
    except Exception as e:
        log.warning(f"⚠️  Could not initialize LLM: {e}")
        log.info("🔄 Falling back to rule-based analysis")
        llm_working = False
    
    for iteration in range(1, max_iterations + 1):
        log.info(f"=== Iteration {iteration} ===")
        
        try:
            # Step 1: Plan
            log.info("Step 1: Planning tasks...")
            planned_tasks = None
            
            if llm_working:
                try:
                    plan_response = planner(invoice_json=invoice_json, known_error=known_error)
                    if hasattr(plan_response, 'tasks'):
                        planned_tasks = safe_json_parse(plan_response.tasks, None)
                    else:
                        planned_tasks = safe_json_parse(str(plan_response), None)
                    
                    if not isinstance(planned_tasks, list):
                        planned_tasks = None
                        
                except Exception as e:
                    log.warning(f"⚠️  LLM planning failed: {e}. Using fallback...")
                    llm_working = False
            
            if planned_tasks is None:
                planned_tasks = fallback_plan_tasks(invoice, known_error)
            
            log.info(f"📋 Planned tasks: {planned_tasks}")
            
            # Step 2: Validate (skip if LLM not working)
            validated_tasks = planned_tasks
            if llm_working:
                try:
                    log.info("Step 2: Validating plan...")
                    validate_response = validator(
                        invoice_json=invoice_json, 
                        proposed_tasks=json.dumps(planned_tasks)
                    )
                    
                    if hasattr(validate_response, 'validated_tasks'):
                        validated_tasks = safe_json_parse(validate_response.validated_tasks, planned_tasks)
                    else:
                        validated_tasks = safe_json_parse(str(validate_response), planned_tasks)
                    
                    if not isinstance(validated_tasks, list):
                        validated_tasks = planned_tasks
                        
                except Exception as e:
                    log.warning(f"⚠️  LLM validation failed: {e}. Using planned tasks...")
                    validated_tasks = planned_tasks
            
            log.info(f"✅ Validated tasks: {validated_tasks}")
            
            # Step 3: Execute specialists
            log.info("Step 3: Executing specialist agents...")
            results = {}
            
            if "CheckVendor" in validated_tasks:
                results["vendor"] = check_vendor(invoice)
            else:
                results["vendor"] = {"vendor_valid": True, "risk_factor": "LOW"}
            
            if "CheckTotals" in validated_tasks:
                results["totals"] = check_totals(invoice)
            else:
                results["totals"] = {"totals_match": True, "risk_factor": "LOW"}
            
            if "AnalyzePatterns" in validated_tasks:
                results["patterns"] = analyze_patterns(invoice)
            else:
                results["patterns"] = {"anomalies_found": 0, "risk_factor": "LOW"}
            
            # Step 4: Summarize
            log.info("Step 4: Summarizing results...")
            summary = None
            
            if llm_working:
                try:
                    summary_response = summarizer(
                        vendor_result=json.dumps(results["vendor"]),
                        totals_result=json.dumps(results["totals"]),
                        patterns_result=json.dumps(results["patterns"])
                    )
                    
                    if hasattr(summary_response, 'summary'):
                        summary = safe_json_parse(summary_response.summary, None)
                    else:
                        summary = safe_json_parse(str(summary_response), None)
                    
                    if not isinstance(summary, dict):
                        summary = None
                        
                except Exception as e:
                    log.warning(f"⚠️  LLM summarization failed: {e}. Using fallback...")
                    llm_working = False
            
            if summary is None:
                summary = fallback_summarize(results["vendor"], results["totals"], results["patterns"])
            
            log.info(f"📊 Summary: {summary}")
            
            # Step 5: Review (skip if LLM not working, assume pass)
            review = {"status": "pass"}
            if llm_working:
                try:
                    log.info("Step 5: Reviewing results...")
                    review_response = reviewer(
                        invoice_json=invoice_json,
                        summary=json.dumps(summary)
                    )
                    
                    if hasattr(review_response, 'review'):
                        review = safe_json_parse(review_response.review, {"status": "pass"})
                    else:
                        review = safe_json_parse(str(review_response), {"status": "pass"})
                    
                    if not isinstance(review, dict):
                        review = {"status": "pass"}
                        
                except Exception as e:
                    log.warning(f"⚠️  LLM review failed: {e}. Assuming pass...")
                    review = {"status": "pass"}
            
            status = review.get("status", "pass")
            error = review.get("error", "")
            
            log.info(f"🔍 Review result: {review}")
            
            if status == "pass":
                log.info("✅ Analysis completed successfully!")
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "llm_working": llm_working,
                    "results": results,
                    "summary": summary,
                    "review": review
                }
            else:
                known_error = error
                log.warning(f"❌ Review failed: {error}. Retrying...")
                
        except Exception as e:
            log.error(f"💥 Error in iteration {iteration}: {e}")
            known_error = f"processing_error_{iteration}"
    
    # If we get here, all iterations failed
    log.warning("⚠️  Max iterations reached, returning conservative analysis")
    
    # Return conservative results
    conservative_results = {
        "vendor": check_vendor(invoice),
        "totals": check_totals(invoice),
        "patterns": analyze_patterns(invoice)
    }
    
    conservative_summary = fallback_summarize(
        conservative_results["vendor"], 
        conservative_results["totals"], 
        conservative_results["patterns"]
    )
    
    return {
        "status": "max_iterations_reached",
        "iterations": max_iterations,
        "llm_working": llm_working,
        "results": conservative_results,
        "summary": conservative_summary
    }

# Test
if __name__ == "__main__":
    # Test invoice with intentional issues
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
    
    print("🔍 Starting Invoice Fraud Detection")
    print("=" * 50)
    
    result = run_fraud_detection(test_invoice)
    
    print("\n📊 FINAL RESULTS:")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    
    # Summary
    summary = result.get("summary", {})
    print(f"\n🎯 CONCLUSION:")
    print(f"Fraud Risk: {summary.get('fraud_risk', 'UNKNOWN')}")
    print(f"Status: {result.get('status', 'UNKNOWN')}")
    print(f"Iterations: {result.get('iterations', 0)}")