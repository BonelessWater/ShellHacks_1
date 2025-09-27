#!/usr/bin/env python3
"""
Hardcoded tools for invoice fraud detection that don't require LLM calls.
These tools provide fast, deterministic analysis for calculations and pattern detection.
"""

import math
import statistics
import re
import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json

class ToolType(Enum):
    HARDCODED = "hardcoded"
    LLM = "llm"

@dataclass
class ToolResult:
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 1.0

class HardcodedTools:
    """Non-LLM tools for fast fraud detection calculations and analysis"""
    
    @staticmethod
    def amount_validator(amounts: List[float], invoice_total: float = None) -> ToolResult:
        """Validate invoice amounts for fraud patterns"""
        import time
        start_time = time.time()
        
        try:
            if not amounts:
                return ToolResult(False, None, "No amounts provided")
            
            analysis = {
                'total_items': len(amounts),
                'sum_amounts': sum(amounts),
                'avg_amount': statistics.mean(amounts),
                'median_amount': statistics.median(amounts),
                'max_amount': max(amounts),
                'min_amount': min(amounts),
                'fraud_indicators': []
            }
            
            # Check for round number patterns (fraud indicator)
            round_numbers = [amt for amt in amounts if amt % 10 == 0 or amt % 100 == 0]
            if len(round_numbers) / len(amounts) > 0.7:
                analysis['fraud_indicators'].append({
                    'type': 'excessive_round_numbers',
                    'severity': 'medium',
                    'count': len(round_numbers),
                    'percentage': (len(round_numbers) / len(amounts)) * 100
                })
            
            # Check for duplicate amounts
            unique_amounts = set(amounts)
            if len(unique_amounts) < len(amounts) * 0.8:
                analysis['fraud_indicators'].append({
                    'type': 'duplicate_amounts',
                    'severity': 'high',
                    'unique_count': len(unique_amounts),
                    'total_count': len(amounts)
                })
            
            # Check total mismatch if provided
            if invoice_total is not None:
                calculated_total = sum(amounts)
                mismatch = abs(calculated_total - invoice_total)
                if mismatch > 0.01:  # More than 1 cent difference
                    analysis['fraud_indicators'].append({
                        'type': 'total_mismatch',
                        'severity': 'high',
                        'calculated_total': calculated_total,
                        'stated_total': invoice_total,
                        'difference': mismatch
                    })
            
            # Check for amounts just under common approval thresholds
            thresholds = [100, 500, 1000, 5000, 10000]
            for threshold in thresholds:
                near_threshold = [amt for amt in amounts if threshold - 50 <= amt < threshold]
                if near_threshold:
                    analysis['fraud_indicators'].append({
                        'type': 'threshold_avoidance',
                        'severity': 'medium',
                        'threshold': threshold,
                        'suspicious_amounts': near_threshold
                    })
            
            # Calculate risk score
            risk_score = len(analysis['fraud_indicators']) * 2
            if risk_score > 10:
                risk_score = 10
                
            analysis['risk_score'] = risk_score
            analysis['recommendation'] = 'REJECT' if risk_score >= 7 else 'REVIEW' if risk_score >= 4 else 'APPROVE'
            
            execution_time = time.time() - start_time
            return ToolResult(True, analysis, execution_time=execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(False, None, str(e), execution_time)
    
    @staticmethod
    def tax_calculator(subtotal: float, tax_rate: float, stated_tax: float) -> ToolResult:
        """Validate tax calculations"""
        import time
        start_time = time.time()
        
        try:
            calculated_tax = subtotal * (tax_rate / 100)
            difference = abs(calculated_tax - stated_tax)
            percentage_error = (difference / calculated_tax) * 100 if calculated_tax > 0 else 0
            
            analysis = {
                'subtotal': subtotal,
                'tax_rate': tax_rate,
                'calculated_tax': round(calculated_tax, 2),
                'stated_tax': stated_tax,
                'difference': round(difference, 2),
                'percentage_error': round(percentage_error, 2),
                'is_accurate': difference <= 0.01,  # Within 1 cent
                'fraud_indicators': []
            }
            
            # Check for significant tax calculation errors
            if percentage_error > 5:  # More than 5% error
                analysis['fraud_indicators'].append({
                    'type': 'tax_calculation_error',
                    'severity': 'high',
                    'error_percentage': percentage_error
                })
            elif percentage_error > 1:  # More than 1% error
                analysis['fraud_indicators'].append({
                    'type': 'tax_calculation_error',
                    'severity': 'medium',
                    'error_percentage': percentage_error
                })
            
            # Check for unusual tax rates
            if tax_rate < 0 or tax_rate > 20:  # Unusual tax rate
                analysis['fraud_indicators'].append({
                    'type': 'unusual_tax_rate',
                    'severity': 'medium',
                    'tax_rate': tax_rate
                })
            
            risk_score = len(analysis['fraud_indicators']) * 3
            if risk_score > 10:
                risk_score = 10
                
            analysis['risk_score'] = risk_score
            analysis['recommendation'] = 'REJECT' if risk_score >= 7 else 'REVIEW' if risk_score >= 4 else 'APPROVE'
            
            execution_time = time.time() - start_time
            return ToolResult(True, analysis, execution_time=execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(False, None, str(e), execution_time)
    
    @staticmethod
    def date_analyzer(invoice_date: str, due_date: str = None, service_dates: List[str] = None) -> ToolResult:
        """Analyze invoice dates for fraud patterns"""
        import time
        start_time = time.time()
        
        try:
            # Parse dates
            def parse_date(date_str):
                formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']
                for fmt in formats:
                    try:
                        return datetime.datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Unable to parse date: {date_str}")
            
            invoice_dt = parse_date(invoice_date)
            current_date = datetime.datetime.now()
            
            analysis = {
                'invoice_date': invoice_date,
                'parsed_invoice_date': invoice_dt.isoformat(),
                'fraud_indicators': []
            }
            
            # Check for future dates
            if invoice_dt > current_date:
                analysis['fraud_indicators'].append({
                    'type': 'future_invoice_date',
                    'severity': 'high',
                    'days_in_future': (invoice_dt - current_date).days
                })
            
            # Check for very old invoices
            days_old = (current_date - invoice_dt).days
            if days_old > 365:  # More than a year old
                analysis['fraud_indicators'].append({
                    'type': 'very_old_invoice',
                    'severity': 'medium',
                    'days_old': days_old
                })
            
            # Check for weekend dates (unusual for business invoices)
            if invoice_dt.weekday() >= 5:  # Saturday or Sunday
                analysis['fraud_indicators'].append({
                    'type': 'weekend_invoice_date',
                    'severity': 'low',
                    'day_of_week': invoice_dt.strftime('%A')
                })
            
            # Analyze due date if provided
            if due_date:
                due_dt = parse_date(due_date)
                payment_days = (due_dt - invoice_dt).days
                
                analysis['due_date'] = due_date
                analysis['payment_days'] = payment_days
                
                if payment_days < 0:  # Due date before invoice date
                    analysis['fraud_indicators'].append({
                        'type': 'invalid_due_date',
                        'severity': 'high',
                        'payment_days': payment_days
                    })
                elif payment_days > 180:  # More than 6 months
                    analysis['fraud_indicators'].append({
                        'type': 'excessive_payment_terms',
                        'severity': 'medium',
                        'payment_days': payment_days
                    })
            
            # Analyze service dates if provided
            if service_dates:
                analysis['service_date_issues'] = []
                for service_date in service_dates:
                    service_dt = parse_date(service_date)
                    if service_dt > invoice_dt:
                        analysis['service_date_issues'].append({
                            'service_date': service_date,
                            'issue': 'service_after_invoice',
                            'severity': 'medium'
                        })
            
            risk_score = len(analysis['fraud_indicators']) * 2
            if risk_score > 10:
                risk_score = 10
                
            analysis['risk_score'] = risk_score
            analysis['recommendation'] = 'REJECT' if risk_score >= 7 else 'REVIEW' if risk_score >= 4 else 'APPROVE'
            
            execution_time = time.time() - start_time
            return ToolResult(True, analysis, execution_time=execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(False, None, str(e), execution_time)
    
    @staticmethod
    def vendor_authenticator(vendor_name: str, vendor_email: str = None, vendor_address: str = None, 
                           approved_vendors: List[str] = None) -> ToolResult:
        """Authenticate vendor information"""
        import time
        start_time = time.time()
        
        try:
            analysis = {
                'vendor_name': vendor_name,
                'vendor_email': vendor_email,
                'vendor_address': vendor_address,
                'fraud_indicators': []
            }
            
            # Check against approved vendor list
            if approved_vendors:
                vendor_lower = vendor_name.lower().strip()
                approved_lower = [v.lower().strip() for v in approved_vendors]
                
                if vendor_lower not in approved_lower:
                    # Check for close matches (potential typosquatting)
                    close_matches = []
                    for approved in approved_vendors:
                        # Simple fuzzy matching
                        if abs(len(vendor_name) - len(approved)) <= 2:
                            similarity = sum(c1 == c2 for c1, c2 in zip(vendor_name.lower(), approved.lower()))
                            if similarity / max(len(vendor_name), len(approved)) > 0.8:
                                close_matches.append(approved)
                    
                    if close_matches:
                        analysis['fraud_indicators'].append({
                            'type': 'potential_typosquatting',
                            'severity': 'high',
                            'close_matches': close_matches
                        })
                    else:
                        analysis['fraud_indicators'].append({
                            'type': 'unknown_vendor',
                            'severity': 'medium',
                            'vendor_name': vendor_name
                        })
            
            # Validate email if provided
            if vendor_email:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, vendor_email):
                    analysis['fraud_indicators'].append({
                        'type': 'invalid_email_format',
                        'severity': 'medium',
                        'email': vendor_email
                    })
                
                # Check for suspicious email domains
                suspicious_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
                domain = vendor_email.split('@')[1].lower() if '@' in vendor_email else ''
                if domain in suspicious_domains:
                    analysis['fraud_indicators'].append({
                        'type': 'personal_email_domain',
                        'severity': 'low',
                        'domain': domain
                    })
            
            # Check vendor name for suspicious patterns
            suspicious_patterns = [
                r'inc\.?$',  # Ending with "inc" without proper formatting
                r'llc\.?$',  # Ending with "llc" without proper formatting
                r'\d{4,}',   # Long sequences of numbers
                r'^[a-z]{1,3}$',  # Very short names
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, vendor_name.lower()):
                    analysis['fraud_indicators'].append({
                        'type': 'suspicious_vendor_name_pattern',
                        'severity': 'low',
                        'pattern': pattern
                    })
            
            risk_score = len(analysis['fraud_indicators']) * 2
            if risk_score > 10:
                risk_score = 10
                
            analysis['risk_score'] = risk_score
            analysis['recommendation'] = 'REJECT' if risk_score >= 7 else 'REVIEW' if risk_score >= 4 else 'APPROVE'
            
            execution_time = time.time() - start_time
            return ToolResult(True, analysis, execution_time=execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(False, None, str(e), execution_time)
    
    @staticmethod
    def format_inspector(invoice_text: str) -> ToolResult:
        """Inspect invoice format for professionalism and completeness"""
        import time
        start_time = time.time()
        
        try:
            analysis = {
                'character_count': len(invoice_text),
                'word_count': len(invoice_text.split()),
                'fraud_indicators': []
            }
            
            # Check for required fields
            required_fields = ['invoice', 'amount', 'date', 'vendor', 'total']
            missing_fields = []
            
            for field in required_fields:
                if field.lower() not in invoice_text.lower():
                    missing_fields.append(field)
            
            if missing_fields:
                analysis['fraud_indicators'].append({
                    'type': 'missing_required_fields',
                    'severity': 'high',
                    'missing_fields': missing_fields
                })
            
            # Check for excessive typos/errors
            words = invoice_text.split()
            suspicious_chars = sum(1 for char in invoice_text if not char.isalnum() and char not in ' .,;:!?-_()[]{}/@#$%^&*+=|\\~`"\'')
            if len(words) > 0 and suspicious_chars / len(invoice_text) > 0.1:
                analysis['fraud_indicators'].append({
                    'type': 'excessive_special_characters',
                    'severity': 'medium',
                    'percentage': (suspicious_chars / len(invoice_text)) * 100
                })
            
            # Check for very short invoice (likely incomplete)
            if len(invoice_text) < 100:
                analysis['fraud_indicators'].append({
                    'type': 'very_short_invoice',
                    'severity': 'high',
                    'character_count': len(invoice_text)
                })
            
            # Check for all caps text (unprofessional)
            if invoice_text.isupper() and len(invoice_text) > 50:
                analysis['fraud_indicators'].append({
                    'type': 'all_caps_text',
                    'severity': 'low'
                })
            
            risk_score = len(analysis['fraud_indicators']) * 2
            if risk_score > 10:
                risk_score = 10
                
            analysis['risk_score'] = risk_score
            analysis['recommendation'] = 'REJECT' if risk_score >= 7 else 'REVIEW' if risk_score >= 4 else 'APPROVE'
            
            execution_time = time.time() - start_time
            return ToolResult(True, analysis, execution_time=execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(False, None, str(e), execution_time)
    
    @staticmethod
    def statistics_calc(numbers: List[float], operation: str) -> ToolResult:
        """Calculate statistics on list of numbers"""
        import time
        start_time = time.time()
        
        try:
            if not numbers:
                raise ValueError("Empty list provided")
            
            operations = {
                'mean': statistics.mean,
                'median': statistics.median,
                'mode': statistics.mode,
                'stdev': statistics.stdev if len(numbers) > 1 else lambda x: 0,
                'variance': statistics.variance if len(numbers) > 1 else lambda x: 0,
                'min': min,
                'max': max,
                'sum': sum,
                'count': len
            }
            
            if operation not in operations:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = operations[operation](numbers)
            execution_time = time.time() - start_time
            return ToolResult(True, result, execution_time=execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(False, None, str(e), execution_time)
    
    @staticmethod
    def calculator(expression: str) -> ToolResult:
        """Safe calculator for mathematical expressions"""
        import time
        start_time = time.time()
        
        try:
            # Only allow safe operations
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "pow": pow
            })
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            execution_time = time.time() - start_time
            return ToolResult(True, result, execution_time=execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(False, None, str(e), execution_time)

# Tool registry for easy access
HARDCODED_TOOL_REGISTRY = {
    'amount_validator': {
        'type': ToolType.HARDCODED,
        'function': HardcodedTools.amount_validator,
        'description': 'Validate invoice amounts for fraud patterns',
        'params': ['amounts', 'invoice_total']
    },
    'tax_calculator': {
        'type': ToolType.HARDCODED,
        'function': HardcodedTools.tax_calculator,
        'description': 'Validate tax calculations',
        'params': ['subtotal', 'tax_rate', 'stated_tax']
    },
    'date_analyzer': {
        'type': ToolType.HARDCODED,
        'function': HardcodedTools.date_analyzer,
        'description': 'Analyze invoice dates for fraud patterns',
        'params': ['invoice_date', 'due_date', 'service_dates']
    },
    'vendor_authenticator': {
        'type': ToolType.HARDCODED,
        'function': HardcodedTools.vendor_authenticator,
        'description': 'Authenticate vendor information',
        'params': ['vendor_name', 'vendor_email', 'vendor_address', 'approved_vendors']
    },
    'format_inspector': {
        'type': ToolType.HARDCODED,
        'function': HardcodedTools.format_inspector,
        'description': 'Inspect invoice format for professionalism',
        'params': ['invoice_text']
    },
    'statistics': {
        'type': ToolType.HARDCODED,
        'function': HardcodedTools.statistics_calc,
        'description': 'Calculate statistics on numbers',
        'params': ['numbers', 'operation']
    },
    'calculator': {
        'type': ToolType.HARDCODED,
        'function': HardcodedTools.calculator,
        'description': 'Safe mathematical calculations',
        'params': ['expression']
    }
}

def get_tool_descriptions() -> str:
    """Get formatted description of available hardcoded tools"""
    descriptions = []
    for name, tool in HARDCODED_TOOL_REGISTRY.items():
        descriptions.append(f"{name}: {tool['description']} (params: {tool['params']})")
    return "\n".join(descriptions)