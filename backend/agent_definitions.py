#!/usr/bin/env python3
"""
Agent definitions and DSPy signatures for invoice fraud detection system.
"""

import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    LOW_MEDIUM = 3
    MEDIUM = 4
    MEDIUM_HIGH = 5
    HIGH = 6
    HIGH_MEDIUM = 7
    VERY_HIGH = 8
    CRITICAL = 9
    MAXIMUM = 10

class Recommendation(Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"

@dataclass
class AgentResponse:
    agent_type: str
    analysis: str
    risk_score: int  # 1-10 scale
    confidence: int  # 1-10 scale
    red_flags: List[str]
    processing_time: float = 0.0
    error_count: int = 0

@dataclass
class ErrorContext:
    error_type: str
    error_message: str
    failed_agent: Optional[str] = None
    retry_count: int = 0
    suggested_fix: Optional[str] = None

# DSPy Signatures for structured LLM interactions

class AgentSelectorSignature(dspy.Signature):
    """Core LLM signature for determining which agents to summon"""
    invoice_data: str = dspy.InputField(desc="Raw invoice data to analyze")
    available_agents: str = dspy.InputField(desc="List of available specialist agents and their capabilities")
    error_context: str = dspy.InputField(desc="Previous errors encountered (if any)", default="")
    
    selected_agents: str = dspy.OutputField(desc="JSON array of agent names to summon, e.g., ['agent1', 'agent2']")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why these agents were selected")

class SpecialistAgentSignature(dspy.Signature):
    """Signature for specialist agent analysis"""
    invoice_data: str = dspy.InputField(desc="Invoice data to analyze")
    agent_role: str = dspy.InputField(desc="Specific role and expertise of this agent")
    focus_area: str = dspy.InputField(desc="Specific aspect to focus analysis on")
    
    analysis: str = dspy.OutputField(desc="Detailed analysis findings (2-3 sentences)")
    risk_score: int = dspy.OutputField(desc="Fraud risk score from 1-10")
    confidence: int = dspy.OutputField(desc="Confidence in assessment from 1-10") 
    red_flags: str = dspy.OutputField(desc="Comma-separated list of red flags found, or 'None'")

class ResultCompilerSignature(dspy.Signature):
    """Signature for compiling final fraud assessment"""
    invoice_data: str = dspy.InputField(desc="Original invoice data")
    agent_analyses: str = dspy.InputField(desc="Combined analyses from all specialist agents")
    statistics: str = dspy.InputField(desc="Statistical summary of agent responses")
    
    overall_risk: int = dspy.OutputField(desc="Final fraud risk score from 1-10")
    recommendation: str = dspy.OutputField(desc="Final recommendation: APPROVE, REVIEW, or REJECT")
    summary: str = dspy.OutputField(desc="2-3 sentence summary of key findings")
    top_concerns: str = dspy.OutputField(desc="Top 3 critical issues found, comma-separated, or 'None'")
    next_steps: str = dspy.OutputField(desc="Recommended next actions")

class ErrorValidatorSignature(dspy.Signature):
    """Signature for validating and fixing errors"""
    error_message: str = dspy.InputField(desc="Error message encountered")
    context: str = dspy.InputField(desc="Context where error occurred")
    previous_attempts: str = dspy.InputField(desc="Previous fix attempts", default="")
    
    error_type: str = dspy.OutputField(desc="Classification of error type")
    is_recoverable: str = dspy.OutputField(desc="Whether error is recoverable (YES/NO)")
    suggested_fix: str = dspy.OutputField(desc="Specific fix recommendation")
    should_restart: str = dspy.OutputField(desc="Whether to restart from core LLM (YES/NO)")

# Agent Definitions

FRAUD_DETECTION_AGENTS = {
    "amount_validator": {
        "name": "Amount Validator",
        "description": "Analyzes invoice amounts for unusual patterns, duplicates, or suspicious values",
        "focus_areas": [
            "Amount reasonableness vs typical ranges",
            "Duplicate amount detection",
            "Round number patterns (potential manual entry)",
            "Currency format and decimal consistency",
            "Mathematical relationships between line items"
        ],
        "risk_indicators": [
            "Unusually round numbers",
            "Amounts just under approval thresholds", 
            "Duplicated amounts across invoices",
            "Inconsistent currency formatting"
        ]
    },
    
    "vendor_authenticator": {
        "name": "Vendor Authenticator", 
        "description": "Validates vendor information and checks for shell companies or suspicious entities",
        "focus_areas": [
            "Vendor name legitimacy and formatting",
            "Address format and geographic consistency",
            "Contact information completeness",
            "Business registration indicators",
            "Vendor-client relationship logic"
        ],
        "risk_indicators": [
            "Generic or suspicious company names",
            "Incomplete address information",
            "Missing or invalid contact details",
            "Vendor name too similar to known companies"
        ]
    },
    
    "date_analyzer": {
        "name": "Date Analyzer",
        "description": "Examines dates for logical consistency and timeline issues", 
        "focus_areas": [
            "Invoice date vs service/delivery dates",
            "Due date calculation accuracy",
            "Date format consistency",
            "Weekend/holiday date patterns",
            "Sequential date logic"
        ],
        "risk_indicators": [
            "Future-dated invoices for past services",
            "Inconsistent date formats",
            "Impossible date sequences",
            "Dates on weekends/holidays"
        ]
    },
    
    "tax_calculator": {
        "name": "Tax Calculator",
        "description": "Verifies tax calculations and rates for accuracy",
        "focus_areas": [
            "Tax rate accuracy for jurisdiction",
            "Tax calculation mathematical accuracy", 
            "Tax exemption appropriateness",
            "Multiple tax handling",
            "Tax ID format validation"
        ],
        "risk_indicators": [
            "Incorrect tax calculations",
            "Unusual tax rates for location",
            "Missing tax when required",
            "Invalid tax ID formats"
        ]
    },
    
    "format_inspector": {
        "name": "Format Inspector",
        "description": "Checks invoice format, structure, and professional appearance",
        "focus_areas": [
            "Professional formatting and layout",
            "Required field completeness",
            "Logo and branding consistency", 
            "Template structure analysis",
            "Language and grammar quality"
        ],
        "risk_indicators": [
            "Poor formatting or layout",
            "Missing required invoice fields",
            "Inconsistent fonts or styling",
            "Grammar/spelling errors"
        ]
    },
    
    "payment_terms_checker": {
        "name": "Payment Terms Checker",
        "description": "Analyzes payment terms and conditions for red flags",
        "focus_areas": [
            "Payment terms reasonableness",
            "Banking information validity",
            "Payment method appropriateness",
            "Terms and conditions completeness",
            "Discount terms logic"
        ],
        "risk_indicators": [
            "Unusual payment terms",
            "Suspicious banking details",
            "Pressure for immediate payment",
            "Unusual payment methods requested"
        ]
    },
    
    "line_item_validator": {
        "name": "Line Item Validator", 
        "description": "Reviews individual line items for reasonableness and pricing",
        "focus_areas": [
            "Item description clarity and detail",
            "Quantity reasonableness",
            "Unit price market consistency",
            "Service/product categorization",
            "Billing period alignment"
        ],
        "risk_indicators": [
            "Vague item descriptions",
            "Unusual quantities or pricing",
            "Services billed outside normal periods",
            "Items inconsistent with vendor type"
        ]
    }
}

def get_agent_prompt_template(agent_type: str) -> str:
    """Get specialized prompt template for each agent type"""
    agent_config = FRAUD_DETECTION_AGENTS.get(agent_type)
    if not agent_config:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    focus_areas_text = "\n".join([f"- {area}" for area in agent_config["focus_areas"]])
    risk_indicators_text = "\n".join([f"- {indicator}" for indicator in agent_config["risk_indicators"]])
    
    return f"""
You are a {agent_config['name']} specialist for invoice fraud detection.

EXPERTISE: {agent_config['description']}

YOUR FOCUS AREAS:
{focus_areas_text}

KEY RISK INDICATORS TO WATCH FOR:
{risk_indicators_text}

SCORING GUIDELINES:
- Risk Score 1-3: Low risk, minor concerns only
- Risk Score 4-6: Medium risk, some red flags present  
- Risk Score 7-9: High risk, multiple concerning factors
- Risk Score 10: Critical risk, clear fraud indicators

- Confidence 1-3: Low confidence, insufficient data
- Confidence 4-6: Medium confidence, adequate analysis possible
- Confidence 7-10: High confidence, clear assessment possible

Analyze ONLY your specific domain. Be precise and factual.
"""

def validate_agent_response(response: AgentResponse) -> List[str]:
    """Validate agent response for common errors"""
    errors = []
    
    if not (1 <= response.risk_score <= 10):
        errors.append(f"Invalid risk_score: {response.risk_score}. Must be 1-10")
    
    if not (1 <= response.confidence <= 10):
        errors.append(f"Invalid confidence: {response.confidence}. Must be 1-10")
    
    if not response.analysis or len(response.analysis.strip()) < 20:
        errors.append("Analysis too short or empty")
    
    if not response.agent_type:
        errors.append("Missing agent_type")
    
    return errors

def get_default_agent_selection() -> List[str]:
    """Get default agents when auto-selection fails"""
    return ["amount_validator", "vendor_authenticator", "format_inspector"]

def get_all_agent_types() -> List[str]:
    """Get list of all available agent types"""
    return list(FRAUD_DETECTION_AGENTS.keys())