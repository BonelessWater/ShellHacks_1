#!/usr/bin/env python3
"""
Error validation and recovery system for invoice fraud detection.
"""

import re
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from agent_definitions import (
    ErrorContext, AgentResponse, ErrorValidatorSignature,
    validate_agent_response
)

log = logging.getLogger("error_validator")

class ErrorType(Enum):
    PARSING_ERROR = "parsing_error"
    VALIDATION_ERROR = "validation_error" 
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    FORMAT_ERROR = "format_error"
    MODEL_ERROR = "model_error"
    JSON_ERROR = "json_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(Enum):
    LOW = 1      # Can continue with warnings
    MEDIUM = 2   # Should retry with fixes
    HIGH = 3     # Must restart current step
    CRITICAL = 4 # Must restart from beginning

@dataclass
class ErrorAnalysis:
    error_type: ErrorType
    severity: ErrorSeverity
    is_recoverable: bool
    suggested_fix: str
    should_restart: bool
    retry_strategy: str
    context_to_add: str = ""

class ErrorValidator:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.error_patterns = self._initialize_error_patterns()
        
    def _initialize_error_patterns(self) -> Dict[str, ErrorType]:
        """Initialize common error patterns for quick classification"""
        return {
            r"invalid literal for int\(\)": ErrorType.PARSING_ERROR,
            r"JSON.*decode.*error": ErrorType.JSON_ERROR,
            r"list index out of range": ErrorType.PARSING_ERROR,
            r"KeyError": ErrorType.VALIDATION_ERROR,
            r"AttributeError": ErrorType.VALIDATION_ERROR,
            r"timeout": ErrorType.TIMEOUT_ERROR,
            r"rate.*limit": ErrorType.API_ERROR,
            r"quota.*exceeded": ErrorType.API_ERROR,
            r"model.*not.*found": ErrorType.MODEL_ERROR,
            r"permission.*denied": ErrorType.API_ERROR,
            r"connection.*error": ErrorType.API_ERROR,
        }
    
    def classify_error(self, error_message: str, context: str = "") -> ErrorType:
        """Classify error type based on message patterns"""
        error_lower = error_message.lower()
        
        for pattern, error_type in self.error_patterns.items():
            if re.search(pattern, error_lower, re.IGNORECASE):
                return error_type
        
        return ErrorType.UNKNOWN_ERROR
    
    def analyze_error(self, error: Exception, context: str, previous_attempts: List[str] = None) -> ErrorAnalysis:
        """Analyze error and determine recovery strategy"""
        if previous_attempts is None:
            previous_attempts = []
            
        error_msg = str(error)
        error_type = self.classify_error(error_msg, context)
        
        # Determine severity and recovery strategy
        if "parsing" in context.lower() and "int()" in error_msg:
            return self._handle_parsing_error(error_msg, context, previous_attempts)
        elif "json" in error_msg.lower():
            return self._handle_json_error(error_msg, context, previous_attempts)
        elif "api" in error_msg.lower() or "rate" in error_msg.lower():
            return self._handle_api_error(error_msg, context, previous_attempts)
        elif "model" in error_msg.lower():
            return self._handle_model_error(error_msg, context, previous_attempts)
        else:
            return self._handle_generic_error(error_msg, context, previous_attempts)
    
    def _handle_parsing_error(self, error_msg: str, context: str, previous_attempts: List[str]) -> ErrorAnalysis:
        """Handle parsing-related errors"""
        if "int()" in error_msg:
            return ErrorAnalysis(
                error_type=ErrorType.PARSING_ERROR,
                severity=ErrorSeverity.MEDIUM,
                is_recoverable=True,
                suggested_fix="Add robust parsing with regex extraction and validation",
                should_restart=False,
                retry_strategy="improve_parsing",
                context_to_add=f"Previous parsing failed: {error_msg}. Use more robust number extraction."
            )
        
        return ErrorAnalysis(
            error_type=ErrorType.PARSING_ERROR,
            severity=ErrorSeverity.HIGH,
            is_recoverable=True,
            suggested_fix="Implement comprehensive input validation",
            should_restart=True,
            retry_strategy="restart_with_validation"
        )
    
    def _handle_json_error(self, error_msg: str, context: str, previous_attempts: List[str]) -> ErrorAnalysis:
        """Handle JSON parsing errors"""
        return ErrorAnalysis(
            error_type=ErrorType.JSON_ERROR,
            severity=ErrorSeverity.MEDIUM,
            is_recoverable=True,
            suggested_fix="Use regex to extract JSON from response, add fallback parsing",
            should_restart=False,
            retry_strategy="improve_json_extraction",
            context_to_add="Previous JSON parsing failed. Provide cleaner JSON format in response."
        )
    
    def _handle_api_error(self, error_msg: str, context: str, previous_attempts: List[str]) -> ErrorAnalysis:
        """Handle API-related errors"""
        if "rate" in error_msg.lower() or "quota" in error_msg.lower():
            return ErrorAnalysis(
                error_type=ErrorType.API_ERROR,
                severity=ErrorSeverity.HIGH,
                is_recoverable=True,
                suggested_fix="Implement exponential backoff and rate limiting",
                should_restart=False,
                retry_strategy="backoff_retry"
            )
        
        return ErrorAnalysis(
            error_type=ErrorType.API_ERROR,
            severity=ErrorSeverity.CRITICAL,
            is_recoverable=len(previous_attempts) < 2,
            suggested_fix="Check API key and model availability",
            should_restart=True,
            retry_strategy="restart_from_beginning"
        )
    
    def _handle_model_error(self, error_msg: str, context: str, previous_attempts: List[str]) -> ErrorAnalysis:
        """Handle model-related errors"""
        return ErrorAnalysis(
            error_type=ErrorType.MODEL_ERROR,
            severity=ErrorSeverity.HIGH,
            is_recoverable=True,
            suggested_fix="Switch to fallback model",
            should_restart=False,
            retry_strategy="try_fallback_model"
        )
    
    def _handle_generic_error(self, error_msg: str, context: str, previous_attempts: List[str]) -> ErrorAnalysis:
        """Handle unknown/generic errors"""
        retry_count = len(previous_attempts)
        
        if retry_count >= self.max_retries:
            return ErrorAnalysis(
                error_type=ErrorType.UNKNOWN_ERROR,
                severity=ErrorSeverity.CRITICAL,
                is_recoverable=False,
                suggested_fix="Manual intervention required",
                should_restart=False,
                retry_strategy="abort"
            )
        
        return ErrorAnalysis(
            error_type=ErrorType.UNKNOWN_ERROR,
            severity=ErrorSeverity.MEDIUM,
            is_recoverable=True,
            suggested_fix="Generic retry with additional context",
            should_restart=retry_count > 1,
            retry_strategy="retry_with_context",
            context_to_add=f"Previous attempt failed with: {error_msg}. Please be more careful with output format."
        )

def robust_parse_integer(text: str, field_name: str, default: int = 5) -> Tuple[int, List[str]]:
    """Robustly parse integer from text with multiple strategies"""
    errors = []
    
    # Strategy 1: Direct integer parsing
    try:
        return int(text), errors
    except ValueError:
        pass
    
    # Strategy 2: Extract first number from text
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        try:
            value = int(numbers[0])
            if 1 <= value <= 10:
                return value, errors
            else:
                errors.append(f"{field_name} value {value} out of range 1-10")
        except ValueError:
            pass
    
    # Strategy 3: Extract from "X/10" format
    fraction_match = re.search(r'(\d+)/10', text)
    if fraction_match:
        try:
            value = int(fraction_match.group(1))
            if 1 <= value <= 10:
                return value, errors
            else:
                errors.append(f"{field_name} fraction value {value} out of range 1-10")
        except ValueError:
            pass
    
    # Strategy 4: Look for scale words
    if any(word in text.lower() for word in ['low', 'minimal', 'slight']):
        return 3, errors
    elif any(word in text.lower() for word in ['medium', 'moderate', 'average']):
        return 5, errors
    elif any(word in text.lower() for word in ['high', 'significant', 'major']):
        return 7, errors
    elif any(word in text.lower() for word in ['critical', 'severe', 'maximum']):
        return 9, errors
    
    errors.append(f"Could not parse {field_name} from: {text}")
    return default, errors

def robust_extract_field(text: str, field: str, default: str = "Unknown") -> str:
    """Robustly extract field from agent response"""
    try:
        # Strategy 1: Standard field: value format
        lines = text.split('\n')
        for line in lines:
            if line.strip().upper().startswith(f"{field.upper()}:"):
                return line.split(":", 1)[1].strip()
        
        # Strategy 2: Look for field anywhere in text (case insensitive)
        pattern = rf"{re.escape(field)}\s*:?\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # Strategy 3: Field without colon
        pattern = rf"{re.escape(field)}\s+(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
            
        return default
    except Exception:
        return default

def robust_parse_red_flags(text: str) -> List[str]:
    """Robustly parse red flags from text"""
    try:
        if not text or text.lower().strip() in ['none', 'no flags', 'no red flags', 'n/a']:
            return []
        
        # Split by common separators
        flags = []
        for separator in [',', ';', '\n', '|']:
            if separator in text:
                flags = [flag.strip() for flag in text.split(separator) if flag.strip()]
                break
        
        if not flags:
            # Single flag or bullet points
            if text.startswith('- '):
                flags = [text[2:].strip()]
            else:
                flags = [text.strip()]
        
        # Clean up flags
        cleaned_flags = []
        for flag in flags:
            flag = flag.strip('- •*')
            if flag and flag.lower() not in ['none', 'no flags']:
                cleaned_flags.append(flag)
        
        return cleaned_flags
    except Exception:
        return [text] if text else []

def validate_and_fix_agent_response(raw_response: str, agent_type: str) -> Tuple[AgentResponse, List[str]]:
    """Validate and fix agent response with robust parsing"""
    errors = []
    
    try:
        # Extract fields with robust parsing
        analysis = robust_extract_field(raw_response, "ANALYSIS", "Analysis unavailable")
        
        risk_text = robust_extract_field(raw_response, "RISK_SCORE", "5")
        risk_score, risk_errors = robust_parse_integer(risk_text, "risk_score", 5)
        errors.extend(risk_errors)
        
        confidence_text = robust_extract_field(raw_response, "CONFIDENCE", "5")
        confidence, conf_errors = robust_parse_integer(confidence_text, "confidence", 5)
        errors.extend(conf_errors)
        
        red_flags_text = robust_extract_field(raw_response, "RED_FLAGS", "None")
        red_flags = robust_parse_red_flags(red_flags_text)
        
        # Create response object
        response = AgentResponse(
            agent_type=agent_type,
            analysis=analysis,
            risk_score=max(1, min(10, risk_score)),
            confidence=max(1, min(10, confidence)),
            red_flags=red_flags
        )
        
        # Additional validation
        validation_errors = validate_agent_response(response)
        errors.extend(validation_errors)
        
        return response, errors
        
    except Exception as e:
        errors.append(f"Critical parsing error: {str(e)}")
        # Return minimal valid response
        return AgentResponse(
            agent_type=agent_type,
            analysis=f"Parsing failed: {str(e)}",
            risk_score=5,
            confidence=1,
            red_flags=["Parsing error"]
        ), errors

def create_error_context(errors: List[str], step: str, retry_count: int = 0) -> str:
    """Create error context string for LLM"""
    if not errors:
        return ""
    
    context = f"\n\nERROR CONTEXT (Step: {step}, Retry: {retry_count}):\n"
    context += "Previous attempt failed with the following issues:\n"
    for i, error in enumerate(errors, 1):
        context += f"{i}. {error}\n"
    
    context += "\nPlease provide your response in the exact format requested, ensuring all fields are properly formatted.\n"
    return context