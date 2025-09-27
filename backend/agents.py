#!/usr/bin/env python3
"""
Specialist agents for Invoice Fraud Detection System
Each agent focuses on a specific aspect of fraud detection
"""

import logging
import re
from typing import Dict, Any, List, Set
from datetime import datetime, timedelta
import math

from data_models import (
    Invoice, VendorCheckResult, TotalsCheckResult, PatternAnalysisResult, 
    DataValidator
)

log = logging.getLogger("fraud_detection_agents")

# Configuration constants
APPROVED_VENDORS = {
    "ACME Corp", "Beta Industries", "Delta LLC", "Gamma Tech",
    "Omega Solutions", "Alpha Systems", "Prime Services"
}

SUSPICIOUS_KEYWORDS = [
    "gift", "cash", "tip", "bonus", "personal", "entertainment",
    "vacation", "travel", "party", "alcohol", "gambling", "casino"
]

HIGH_VALUE_THRESHOLD = 1000.0
HIGH_QUANTITY_THRESHOLD = 100
TOTAL_MISMATCH_TOLERANCE = 0.01

class VendorAgent:
    """Agent responsible for vendor validation and risk assessment"""
    
    def __init__(self, approved_vendors: Set[str] = None):
        self.approved_vendors = approved_vendors or APPROVED_VENDORS.copy()
        self.vendor_history = {}  # Track vendor patterns over time
        self.blacklisted_vendors = set()  # Known fraudulent vendors
        
        log.info(f"VendorAgent initialized with {len(self.approved_vendors)} approved vendors")
    
    def add_approved_vendor(self, vendor: str) -> bool:
        """Add a vendor to the approved list"""
        try:
            normalized = DataValidator.normalize_vendor_name(vendor)
            if normalized:
                self.approved_vendors.add(normalized)
                log.info(f"✅ Added approved vendor: {normalized}")
                return True
            return False
        except Exception as e:
            log.error(f"Error adding vendor {vendor}: {e}")
            return False
    
    def remove_approved_vendor(self, vendor: str) -> bool:
        """Remove a vendor from the approved list"""
        try:
            normalized = DataValidator.normalize_vendor_name(vendor)
            if normalized in self.approved_vendors:
                self.approved_vendors.discard(normalized)
                log.info(f"❌ Removed approved vendor: {normalized}")
                return True
            return False
        except Exception as e:
            log.error(f"Error removing vendor {vendor}: {e}")
            return False
    
    def blacklist_vendor(self, vendor: str, reason: str = "") -> bool:
        """Add vendor to blacklist"""
        try:
            normalized = DataValidator.normalize_vendor_name(vendor)
            if normalized:
                self.blacklisted_vendors.add(normalized)
                log.warning(f"🚫 Blacklisted vendor: {normalized} - {reason}")
                return True
            return False
        except Exception as e:
            log.error(f"Error blacklisting vendor {vendor}: {e}")
            return False
    
    def check_vendor(self, invoice: Invoice) -> VendorCheckResult:
        """Check vendor against approved list and assess risk"""
        try:
            normalized_vendor = DataValidator.normalize_vendor_name(invoice.vendor)
            
            # Check blacklist first
            if normalized_vendor in self.blacklisted_vendors:
                return VendorCheckResult(
                    vendor=normalized_vendor,
                    vendor_valid=False,
                    risk_factor="HIGH",
                    confidence=1.0,
                    notes="Vendor is blacklisted"
                )
            
            # Check for empty vendor
            if not normalized_vendor.strip():
                return VendorCheckResult(
                    vendor=normalized_vendor,
                    vendor_valid=False,
                    risk_factor="HIGH",
                    confidence=1.0,
                    notes="Empty vendor name"
                )
            
            # Check exact match first
            is_valid = normalized_vendor in self.approved_vendors
            confidence = 1.0
            notes = "Exact match" if is_valid else "No exact match"
            
            # If not exact match, check for fuzzy matches
            if not is_valid:
                fuzzy_match, similarity = self._find_best_fuzzy_match(normalized_vendor)
                if fuzzy_match:
                    is_valid = True
                    confidence = similarity
                    notes = f"Fuzzy match with '{fuzzy_match}' (similarity: {similarity:.2f})"
                    log.info(f"📋 Fuzzy match: '{normalized_vendor}' → '{fuzzy_match}'")
            
            # Determine risk level
            if is_valid:
                if confidence >= 0.9:
                    risk_factor = "LOW"
                elif confidence >= 0.7:
                    risk_factor = "MEDIUM"
                else:
                    risk_factor = "HIGH"
            else:
                risk_factor = "HIGH"
            
            # Update vendor history
            self._update_vendor_history(normalized_vendor, is_valid)
            
            result = VendorCheckResult(
                vendor=normalized_vendor,
                vendor_valid=is_valid,
                risk_factor=risk_factor,
                confidence=confidence,
                notes=notes
            )
            
            log.info(f"Vendor check: '{normalized_vendor}' → {'VALID' if is_valid else 'INVALID'} ({risk_factor} risk)")
            return result
            
        except Exception as e:
            log.error(f"Error in vendor check: {e}")
            return VendorCheckResult(
                vendor=invoice.vendor,
                vendor_valid=False,
                risk_factor="HIGH",
                confidence=0.0,
                notes=f"Error during check: {str(e)[:100]}"
            )
    
    def _find_best_fuzzy_match(self, vendor: str, threshold: float = 0.7) -> tuple:
        """Find best fuzzy match for vendor name"""
        best_match = None
        best_similarity = 0.0
        
        for approved in self.approved_vendors:
            similarity = self._calculate_similarity(vendor, approved)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = approved
        
        return best_match, best_similarity
    
    def _calculate_similarity(self, vendor1: str, vendor2: str) -> float:
        """Calculate similarity between two vendor names"""
        if not vendor1 or not vendor2:
            return 0.0
        
        # Normalize for comparison
        v1 = vendor1.lower().strip()
        v2 = vendor2.lower().strip()
        
        if v1 == v2:
            return 1.0
        
        # Word-based Jaccard similarity
        v1_words = set(re.findall(r'\w+', v1))
        v2_words = set(re.findall(r'\w+', v2))
        
        if not v1_words or not v2_words:
            return 0.0
        
        intersection = len(v1_words.intersection(v2_words))
        union = len(v1_words.union(v2_words))
        
        jaccard = intersection / union if union > 0 else 0
        
        # Character-based similarity (Levenshtein distance)
        char_similarity = 1 - (self._levenshtein_distance(v1, v2) / max(len(v1), len(v2)))
        
        # Weighted combination
        return 0.7 * jaccard + 0.3 * char_similarity
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _update_vendor_history(self, vendor: str, is_valid: bool):
        """Update vendor history for pattern tracking"""
        if vendor not in self.vendor_history:
            self.vendor_history[vendor] = {
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "valid_count": 0,
                "invalid_count": 0,
                "total_amount": 0.0
            }
        
        history = self.vendor_history[vendor]
        history["last_seen"] = datetime.now()
        
        if is_valid:
            history["valid_count"] += 1
        else:
            history["invalid_count"] += 1
    
    def get_vendor_stats(self) -> Dict[str, Any]:
        """Get vendor statistics"""
        return {
            "approved_vendors": len(self.approved_vendors),
            "blacklisted_vendors": len(self.blacklisted_vendors),
            "tracked_vendors": len(self.vendor_history),
            "approved_list": sorted(list(self.approved_vendors))
        }

class TotalsAgent:
    """Agent responsible for totals validation and mathematical accuracy"""
    
    def __init__(self, tolerance: float = TOTAL_MISMATCH_TOLERANCE):
        self.tolerance = tolerance
        self.calculation_history = []
        
        log.info(f"TotalsAgent initialized with tolerance: ${tolerance}")
    
    def check_totals(self, invoice: Invoice) -> TotalsCheckResult:
        """Verify invoice total matches line items"""
        try:
            reported_total = float(invoice.total)
            calculated_total = float(invoice.calculated_total)
            difference = abs(reported_total - calculated_total)
            
            # Check if totals match within tolerance
            matches = difference <= self.tolerance
            
            # Determine risk level based on difference magnitude
            if matches:
                risk_factor = "LOW"
            elif difference <= 1.0:  # Very small discrepancy
                risk_factor = "LOW"
            elif difference <= 10.0:  # Small discrepancy
                risk_factor = "MEDIUM"
            elif difference <= 100.0:  # Moderate discrepancy
                risk_factor = "HIGH"
            else:  # Large discrepancy
                risk_factor = "HIGH"
            
            # Additional checks for systematic errors
            if not matches:
                # Check for percentage-based errors
                if reported_total > 0:
                    percentage_error = (difference / reported_total) * 100
                    if percentage_error > 50:  # More than 50% error
                        risk_factor = "HIGH"
                
                # Check for common calculation mistakes
                if self._check_common_mistakes(invoice, reported_total, calculated_total):
                    risk_factor = "HIGH"
            
            result = TotalsCheckResult(
                reported_total=reported_total,
                calculated_total=calculated_total,
                difference=difference,
                totals_match=matches,
                risk_factor=risk_factor,
                tolerance_used=self.tolerance
            )
            
            # Track calculation for pattern analysis
            self._track_calculation(invoice, result)
            
            status_msg = "MATCH" if matches else f"MISMATCH (${difference:.2f})"
            log.info(f"Totals check: ${reported_total} vs ${calculated_total} → {status_msg}")
            return result
            
        except Exception as e:
            log.error(f"Error in totals check: {e}")
            return TotalsCheckResult(
                reported_total=invoice.total,
                calculated_total=0.0,
                difference=invoice.total,
                totals_match=False,
                risk_factor="HIGH",
                tolerance_used=self.tolerance
            )
    
    def _check_common_mistakes(self, invoice: Invoice, reported: float, calculated: float) -> bool:
        """Check for common calculation mistakes that might indicate fraud"""
        
        # Check if someone forgot to add tax (common percentage additions)
        tax_rates = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15]  # 5% to 15%
        for rate in tax_rates:
            with_tax = calculated * (1 + rate)
            if abs(reported - with_tax) <= self.tolerance:
                log.info(f"Possible tax calculation: {rate*100}% added")
                return False  # This is actually a valid explanation
        
        # Check for decimal point errors
        if abs(reported - calculated * 10) <= self.tolerance:
            log.warning("Possible decimal point error: 10x multiplier")
            return True
        
        if abs(reported - calculated / 10) <= self.tolerance:
            log.warning("Possible decimal point error: 10x divider")
            return True
        
        # Check for round number substitution (fraud indicator)
        if reported % 100 == 0 and calculated % 100 != 0:
            if abs(reported - calculated) > 50:
                log.warning("Suspicious round number substitution")
                return True
        
        return False
    
    def _track_calculation(self, invoice: Invoice, result: TotalsCheckResult):
        """Track calculation results for pattern analysis"""
        self.calculation_history.append({
            "timestamp": datetime.now(),
            "invoice_id": invoice.invoice_id,
            "vendor": invoice.vendor,
            "item_count": len(invoice.items),
            "difference": result.difference,
            "matches": result.totals_match,
            "risk_factor": result.risk_factor
        })
        
        # Keep only recent history (last 100 calculations)
        if len(self.calculation_history) > 100:
            self.calculation_history = self.calculation_history[-100:]
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """Get calculation statistics"""
        if not self.calculation_history:
            return {"message": "No calculations tracked yet"}
        
        total_calcs = len(self.calculation_history)
        matches = sum(1 for calc in self.calculation_history if calc["matches"])
        avg_difference = sum(calc["difference"] for calc in self.calculation_history) / total_calcs
        
        return {
            "total_calculations": total_calcs,
            "match_rate": matches / total_calcs,
            "average_difference": avg_difference,
            "current_tolerance": self.tolerance
        }

class PatternAgent:
    """Agent responsible for pattern analysis and anomaly detection"""
    
    def __init__(self):
        self.suspicious_keywords = SUSPICIOUS_KEYWORDS.copy()
        self.high_value_threshold = HIGH_VALUE_THRESHOLD
        self.high_quantity_threshold = HIGH_QUANTITY_THRESHOLD
        self.pattern_history = []
        
        log.info(f"PatternAgent initialized with {len(self.suspicious_keywords)} suspicious keywords")
    
    def add_suspicious_keyword(self, keyword: str) -> bool:
        """Add a keyword to the suspicious list"""
        try:
            keyword = keyword.lower().strip()
            if keyword and keyword not in self.suspicious_keywords:
                self.suspicious_keywords.append(keyword)
                log.info(f"✅ Added suspicious keyword: {keyword}")
                return True
            return False
        except Exception as e:
            log.error(f"Error adding keyword {keyword}: {e}")
            return False
    
    def remove_suspicious_keyword(self, keyword: str) -> bool:
        """Remove a keyword from the suspicious list"""
        try:
            keyword = keyword.lower().strip()
            if keyword in self.suspicious_keywords:
                self.suspicious_keywords.remove(keyword)
                log.info(f"❌ Removed suspicious keyword: {keyword}")
                return True
            return False
        except Exception as e:
            log.error(f"Error removing keyword {keyword}: {e}")
            return False
    
    def analyze_patterns(self, invoice: Invoice) -> PatternAnalysisResult:
        """Comprehensive pattern analysis of the invoice"""
        try:
            anomalies = []
            pattern_types = []
            severity_score = 0.0
            
            # Analyze each line item
            for i, item in enumerate(invoice.items):
                item_anomalies, item_severity = self._analyze_item(item, i + 1)
                anomalies.extend(item_anomalies)
                severity_score += item_severity
            
            # Analyze invoice-level patterns
            invoice_anomalies, invoice_severity = self._analyze_invoice_level(invoice)
            anomalies.extend(invoice_anomalies)
            severity_score += invoice_severity
            
            # Analyze temporal patterns
            temporal_anomalies = self._analyze_temporal_patterns(invoice)
            anomalies.extend(temporal_anomalies)
            
            # Determine pattern types
            if any("suspicious" in anomaly.lower() for anomaly in anomalies):
                pattern_types.append("keyword_suspicious")
            if any("high" in anomaly.lower() for anomaly in anomalies):
                pattern_types.append("value_anomaly")
            if any("duplicate" in anomaly.lower() for anomaly in anomalies):
                pattern_types.append("duplicate_items")
            if any("round" in anomaly.lower() for anomaly in anomalies):
                pattern_types.append("round_number_bias")
            
            # Determine overall risk level
            anomaly_count = len(anomalies)
            if anomaly_count == 0 and severity_score == 0:
                risk_factor = "LOW"
            elif anomaly_count <= 2 and severity_score < 0.3:
                risk_factor = "MEDIUM"
            else:
                risk_factor = "HIGH"
            
            result = PatternAnalysisResult(
                anomalies_found=anomaly_count,
                anomaly_details=anomalies,
                risk_factor=risk_factor,
                pattern_types=pattern_types,
                severity_score=min(severity_score, 1.0)  # Cap at 1.0
            )
            
            # Track patterns for learning
            self._track_patterns(invoice, result)
            
            log.info(f"Pattern analysis: {anomaly_count} anomalies, {risk_factor} risk, severity {severity_score:.2f}")
            return result
            
        except Exception as e:
            log.error(f"Error in pattern analysis: {e}")
            return PatternAnalysisResult(
                anomalies_found=1,
                anomaly_details=[f"Analysis error: {str(e)[:100]}"],
                risk_factor="HIGH",
                pattern_types=["analysis_error"],
                severity_score=1.0
            )
    
    def _analyze_item(self, item, item_number: int) -> tuple:
        """Analyze individual line item for anomalies"""
        anomalies = []
        severity = 0.0
        
        description = str(item.description).lower()
        quantity = item.quantity
        unit_price = item.unit_price
        line_total = item.line_total
        
        # Check for suspicious keywords
        for keyword in self.suspicious_keywords:
            if keyword in description:
                anomalies.append(f"Item {item_number}: Suspicious description contains '{keyword}' - {item.description}")
                severity += 0.3
                break
        
        # Check for unusually high amounts
        if line_total > self.high_value_threshold:
            anomalies.append(f"Item {item_number}: High-value item - {item.description} (${line_total:.2f})")
            severity += 0.2
        
        # Check for unusual quantities
        if quantity > self.high_quantity_threshold:
            anomalies.append(f"Item {item_number}: High quantity - {quantity} of {item.description}")
            severity += 0.2
        
        # Check for unusual unit prices
        if unit_price > 1000:
            anomalies.append(f"Item {item_number}: High unit price - {item.description} at ${unit_price:.2f} each")
            severity += 0.2
        
        # Check for zero or negative values
        if quantity <= 0:
            anomalies.append(f"Item {item_number}: Invalid quantity - {item.description} has quantity {quantity}")
            severity += 0.4
        
        if unit_price <= 0:
            anomalies.append(f"Item {item_number}: Invalid price - {item.description} has price ${unit_price:.2f}")
            severity += 0.4
        
        # Check for suspicious price patterns
        if unit_price > 0 and unit_price % 1 == 0 and unit_price >= 100:
            # Very round, high prices might be suspicious
            anomalies.append(f"Item {item_number}: Suspiciously round high price - {item.description} at ${unit_price:.0f}")
            severity += 0.1
        
        return anomalies, severity
    
    def _analyze_invoice_level(self, invoice: Invoice) -> tuple:
        """Analyze invoice-level patterns"""
        anomalies = []
        severity = 0.0
        
        # Check total number of items
        if len(invoice.items) > 50:
            anomalies.append(f"Unusually high number of line items: {len(invoice.items)}")
            severity += 0.2
        
        # Check for duplicate items (exact matches)
        descriptions = [item.description.lower().strip() for item in invoice.items]
        seen = set()
        duplicates = set()
        for desc in descriptions:
            if desc in seen:
                duplicates.add(desc)
            seen.add(desc)
        
        if duplicates:
            anomalies.append(f"Duplicate items found: {', '.join(duplicates)}")
            severity += 0.3
        
        # Check for round number bias
        round_count = sum(1 for item in invoice.items 
                         if item.unit_price % 1 == 0 and item.unit_price >= 10)
        
        if len(invoice.items) > 3 and round_count / len(invoice.items) > 0.8:
            anomalies.append("High proportion of round-number prices (potential fabrication)")
            severity += 0.2
        
        # Check for vendor-item consistency
        vendor_lower = invoice.vendor.lower()
        if "tech" in vendor_lower or "software" in vendor_lower:
            non_tech_items = [
                item for item in invoice.items 
                if not any(word in item.description.lower() 
                          for word in ["software", "license", "computer", "tech", "digital", "hardware"])
            ]
            if len(non_tech_items) > len(invoice.items) / 2:
                anomalies.append("Tech vendor with predominantly non-tech items")
                severity += 0.2
        
        # Check for price distribution anomalies
        if len(invoice.items) > 5:
            prices = [item.unit_price for item in invoice.items if item.unit_price > 0]
            if prices:
                avg_price = sum(prices) / len(prices)
                outliers = [p for p in prices if p > avg_price * 5]  # 5x average
                if outliers:
                    anomalies.append(f"Extreme price outliers detected: {len(outliers)} items significantly above average")
                    severity += 0.1
        
        return anomalies, severity
    
    def _analyze_temporal_patterns(self, invoice: Invoice) -> List[str]:
        """Analyze temporal patterns in invoice"""
        anomalies = []
        
        try:
            # Check if date is in future
            invoice_date = datetime.strptime(invoice.date, "%Y-%m-%d")
            if invoice_date > datetime.now():
                anomalies.append(f"Future invoice date: {invoice.date}")
            
            # Check if date is too old (more than 2 years)
            if invoice_date < datetime.now() - timedelta(days=730):
                anomalies.append(f"Very old invoice date: {invoice.date}")
            
            # Check for weekend dates (might be suspicious for business invoices)
            if invoice_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                anomalies.append(f"Weekend invoice date: {invoice.date}")
                
        except ValueError:
            anomalies.append(f"Invalid date format: {invoice.date}")
        except Exception as e:
            log.warning(f"Error analyzing temporal patterns: {e}")
        
        return anomalies
    
    def _track_patterns(self, invoice: Invoice, result: PatternAnalysisResult):
        """Track pattern analysis for learning"""
        self.pattern_history.append({
            "timestamp": datetime.now(),
            "invoice_id": invoice.invoice_id,
            "vendor": invoice.vendor,
            "anomalies_found": result.anomalies_found,
            "risk_factor": result.risk_factor,
            "pattern_types": result.pattern_types,
            "severity_score": result.severity_score
        })
        
        # Keep only recent history
        if len(self.pattern_history) > 100:
            self.pattern_history = self.pattern_history[-100:]
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get pattern analysis statistics"""
        if not self.pattern_history:
            return {"message": "No patterns tracked yet"}
        
        total_analyses = len(self.pattern_history)
        high_risk = sum(1 for p in self.pattern_history if p["risk_factor"] == "HIGH")
        avg_anomalies = sum(p["anomalies_found"] for p in self.pattern_history) / total_analyses
        
        # Most common pattern types
        all_patterns = []
        for analysis in self.pattern_history:
            all_patterns.extend(analysis["pattern_types"])
        
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return {
            "total_analyses": total_analyses,
            "high_risk_rate": high_risk / total_analyses,
            "average_anomalies": avg_anomalies,
            "suspicious_keywords": len(self.suspicious_keywords),
            "common_patterns": pattern_counts
        }

class AgentCoordinator:
    """Coordinates the execution of specialist agents"""
    
    def __init__(self):
        self.vendor_agent = VendorAgent()
        self.totals_agent = TotalsAgent()
        self.pattern_agent = PatternAgent()
        self.execution_history = []
        
        log.info("✅ Agent coordinator initialized")
    
    def execute_tasks(self, invoice: Invoice, tasks: List[str]) -> Dict[str, Any]:
        """Execute specified tasks and return results"""
        start_time = datetime.now()
        results = {}
        
        log.info(f"🔍 Executing tasks: {tasks} for invoice {invoice.invoice_id}")
        
        try:
            # Execute vendor check
            if "CheckVendor" in tasks:
                try:
                    results["vendor"] = self.vendor_agent.check_vendor(invoice).to_dict()
                except Exception as e:
                    log.error(f"❌ Vendor check failed: {e}")
                    results["vendor"] = {
                        "vendor": invoice.vendor,
                        "vendor_valid": False,
                        "risk_factor": "HIGH",
                        "confidence": 0.0,
                        "notes": f"Error: {str(e)[:100]}"
                    }
            else:
                # Default safe result when not checking vendor
                results["vendor"] = {
                    "vendor": invoice.vendor,
                    "vendor_valid": True,
                    "risk_factor": "LOW",
                    "confidence": 1.0,
                    "notes": "Vendor check skipped"
                }
            
            # Execute totals check
            if "CheckTotals" in tasks:
                try:
                    results["totals"] = self.totals_agent.check_totals(invoice).to_dict()
                except Exception as e:
                    log.error(f"❌ Totals check failed: {e}")
                    results["totals"] = {
                        "reported_total": invoice.total,
                        "calculated_total": invoice.total,
                        "difference": 0.0,
                        "totals_match": False,
                        "risk_factor": "HIGH",
                        "tolerance_used": TOTAL_MISMATCH_TOLERANCE
                    }
            else:
                # Default safe result when not checking totals
                results["totals"] = {
                    "reported_total": invoice.total,
                    "calculated_total": invoice.total,
                    "difference": 0.0,
                    "totals_match": True,
                    "risk_factor": "LOW",
                    "tolerance_used": TOTAL_MISMATCH_TOLERANCE
                }
            
            # Execute pattern analysis
            if "AnalyzePatterns" in tasks:
                try:
                    results["patterns"] = self.pattern_agent.analyze_patterns(invoice).to_dict()
                except Exception as e:
                    log.error(f"❌ Pattern analysis failed: {e}")
                    results["patterns"] = {
                        "anomalies_found": 1,
                        "anomaly_details": [f"Analysis error: {str(e)[:100]}"],
                        "risk_factor": "HIGH",
                        "pattern_types": ["error"],
                        "severity_score": 1.0
                    }
            else:
                # Default safe result when not analyzing patterns
                results["patterns"] = {
                    "anomalies_found": 0,
                    "anomaly_details": [],
                    "risk_factor": "LOW",
                    "pattern_types": [],
                    "severity_score": 0.0
                }
            
            # Track execution
            execution_time = (datetime.now() - start_time).total_seconds()
            self._track_execution(invoice, tasks, results, execution_time)
            
            log.info(f"✅ Agent execution completed in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            log.error(f"💥 Critical error in agent execution: {e}")
            # Return error results
            return {
                "vendor": {"vendor": invoice.vendor, "vendor_valid": False, "risk_factor": "HIGH"},
                "totals": {"reported_total": invoice.total, "totals_match": False, "risk_factor": "HIGH"},
                "patterns": {"anomalies_found": 1, "risk_factor": "HIGH", "anomaly_details": [f"Execution error: {str(e)}"]}
            }
    
    def _track_execution(self, invoice: Invoice, tasks: List[str], results: Dict[str, Any], execution_time: float):
        """Track agent execution for performance monitoring"""
        self.execution_history.append({
            "timestamp": datetime.now(),
            "invoice_id": invoice.invoice_id,
            "vendor": invoice.vendor,
            "tasks_executed": tasks,
            "execution_time": execution_time,
            "vendor_risk": results.get("vendor", {}).get("risk_factor", "UNKNOWN"),
            "totals_risk": results.get("totals", {}).get("risk_factor", "UNKNOWN"),
            "patterns_risk": results.get("patterns", {}).get("risk_factor", "UNKNOWN"),
            "overall_risk": self._calculate_overall_risk(results)
        })
        
        # Keep only recent history
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def _calculate_overall_risk(self, results: Dict[str, Any]) -> str:
        """Calculate overall risk from individual agent results"""
        risk_levels = {
            "HIGH": 3,
            "MEDIUM": 2, 
            "LOW": 1,
            "UNKNOWN": 2
        }
        
        vendor_risk = results.get("vendor", {}).get("risk_factor", "LOW")
        totals_risk = results.get("totals", {}).get("risk_factor", "LOW")
        patterns_risk = results.get("patterns", {}).get("risk_factor", "LOW")
        
        max_risk = max(
            risk_levels.get(vendor_risk, 1),
            risk_levels.get(totals_risk, 1),
            risk_levels.get(patterns_risk, 1)
        )
        
        for level, value in risk_levels.items():
            if value == max_risk:
                return level
        
        return "MEDIUM"
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all agents"""
        try:
            vendor_stats = self.vendor_agent.get_vendor_stats()
            totals_stats = self.totals_agent.get_calculation_stats()
            pattern_stats = self.pattern_agent.get_pattern_stats()
            
            # Execution statistics
            execution_stats = {}
            if self.execution_history:
                total_executions = len(self.execution_history)
                avg_time = sum(e["execution_time"] for e in self.execution_history) / total_executions
                high_risk_count = sum(1 for e in self.execution_history if e["overall_risk"] == "HIGH")
                
                execution_stats = {
                    "total_executions": total_executions,
                    "average_execution_time": avg_time,
                    "high_risk_rate": high_risk_count / total_executions
                }
            else:
                execution_stats = {"message": "No executions tracked yet"}
            
            return {
                "coordinator_status": "ready",
                "vendor_agent": {
                    "status": "ready",
                    "stats": vendor_stats
                },
                "totals_agent": {
                    "status": "ready", 
                    "stats": totals_stats
                },
                "pattern_agent": {
                    "status": "ready",
                    "stats": pattern_stats
                },
                "execution_stats": execution_stats
            }
            
        except Exception as e:
            log.error(f"Error getting agent status: {e}")
            return {
                "coordinator_status": "error",
                "error": str(e),
                "vendor_agent": {"status": "unknown"},
                "totals_agent": {"status": "unknown"},
                "pattern_agent": {"status": "unknown"}
            }
    
    def configure_agents(self, config: Dict[str, Any]) -> bool:
        """Configure agents with custom settings"""
        try:
            success = True
            
            # Configure vendor agent
            if "vendor" in config:
                vendor_config = config["vendor"]
                
                if "approved_vendors" in vendor_config:
                    for vendor in vendor_config["approved_vendors"]:
                        self.vendor_agent.add_approved_vendor(vendor)
                
                if "blacklisted_vendors" in vendor_config:
                    for vendor, reason in vendor_config["blacklisted_vendors"].items():
                        self.vendor_agent.blacklist_vendor(vendor, reason)
            
            # Configure totals agent
            if "totals" in config:
                totals_config = config["totals"]
                
                if "tolerance" in totals_config:
                    self.totals_agent.tolerance = float(totals_config["tolerance"])
                    log.info(f"Updated totals tolerance to: {self.totals_agent.tolerance}")
            
            # Configure pattern agent
            if "patterns" in config:
                patterns_config = config["patterns"]
                
                if "suspicious_keywords" in patterns_config:
                    for keyword in patterns_config["suspicious_keywords"]:
                        self.pattern_agent.add_suspicious_keyword(keyword)
                
                if "high_value_threshold" in patterns_config:
                    self.pattern_agent.high_value_threshold = float(patterns_config["high_value_threshold"])
                    log.info(f"Updated high value threshold to: {self.pattern_agent.high_value_threshold}")
                
                if "high_quantity_threshold" in patterns_config:
                    self.pattern_agent.high_quantity_threshold = float(patterns_config["high_quantity_threshold"])
                    log.info(f"Updated high quantity threshold to: {self.pattern_agent.high_quantity_threshold}")
            
            log.info("✅ Agent configuration completed successfully")
            return success
            
        except Exception as e:
            log.error(f"❌ Error configuring agents: {e}")
            return False
    
    def reset_agents(self) -> bool:
        """Reset all agent states and histories"""
        try:
            # Clear histories
            self.execution_history.clear()
            self.vendor_agent.vendor_history.clear()
            self.totals_agent.calculation_history.clear()
            self.pattern_agent.pattern_history.clear()
            
            log.info("✅ All agent states reset")
            return True
            
        except Exception as e:
            log.error(f"❌ Error resetting agents: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        try:
            executions = self.execution_history
            
            # Time-based metrics
            execution_times = [e["execution_time"] for e in executions]
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            # Risk distribution
            risk_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for execution in executions:
                risk = execution.get("overall_risk", "MEDIUM")
                risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
            
            # Task execution frequency
            task_frequency = {}
            for execution in executions:
                for task in execution.get("tasks_executed", []):
                    task_frequency[task] = task_frequency.get(task, 0) + 1
            
            # Recent performance (last 20 executions)
            recent_executions = executions[-20:] if len(executions) >= 20 else executions
            recent_avg_time = sum(e["execution_time"] for e in recent_executions) / len(recent_executions)
            recent_high_risk = sum(1 for e in recent_executions if e["overall_risk"] == "HIGH")
            
            return {
                "total_executions": len(executions),
                "time_metrics": {
                    "average_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "recent_average": recent_avg_time
                },
                "risk_distribution": risk_distribution,
                "task_frequency": task_frequency,
                "recent_performance": {
                    "sample_size": len(recent_executions),
                    "high_risk_count": recent_high_risk,
                    "high_risk_rate": recent_high_risk / len(recent_executions)
                }
            }
            
        except Exception as e:
            log.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}

# Utility functions for external use
def create_agent_coordinator() -> AgentCoordinator:
    """Create a new agent coordinator instance"""
    return AgentCoordinator()

def validate_agent_config(config: Dict[str, Any]) -> tuple:
    """Validate agent configuration format"""
    errors = []
    warnings = []
    
    try:
        # Validate vendor config
        if "vendor" in config:
            vendor_config = config["vendor"]
            if "approved_vendors" in vendor_config:
                if not isinstance(vendor_config["approved_vendors"], list):
                    errors.append("vendor.approved_vendors must be a list")
            
            if "blacklisted_vendors" in vendor_config:
                if not isinstance(vendor_config["blacklisted_vendors"], dict):
                    errors.append("vendor.blacklisted_vendors must be a dictionary")
        
        # Validate totals config
        if "totals" in config:
            totals_config = config["totals"]
            if "tolerance" in totals_config:
                try:
                    tolerance = float(totals_config["tolerance"])
                    if tolerance < 0:
                        errors.append("totals.tolerance must be non-negative")
                    if tolerance > 100:
                        warnings.append("totals.tolerance is very high (>$100)")
                except (ValueError, TypeError):
                    errors.append("totals.tolerance must be a number")
        
        # Validate patterns config
        if "patterns" in config:
            patterns_config = config["patterns"]
            
            if "suspicious_keywords" in patterns_config:
                if not isinstance(patterns_config["suspicious_keywords"], list):
                    errors.append("patterns.suspicious_keywords must be a list")
            
            for threshold_key in ["high_value_threshold", "high_quantity_threshold"]:
                if threshold_key in patterns_config:
                    try:
                        value = float(patterns_config[threshold_key])
                        if value < 0:
                            errors.append(f"patterns.{threshold_key} must be non-negative")
                    except (ValueError, TypeError):
                        errors.append(f"patterns.{threshold_key} must be a number")
        
        return errors, warnings
        
    except Exception as e:
        return [f"Configuration validation error: {e}"], []

# Export main classes and functions
__all__ = [
    'VendorAgent', 'TotalsAgent', 'PatternAgent', 'AgentCoordinator',
    'create_agent_coordinator', 'validate_agent_config',
    'APPROVED_VENDORS', 'SUSPICIOUS_KEYWORDS'
]