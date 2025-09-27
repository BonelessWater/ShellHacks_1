#!/usr/bin/env python3
"""
Invoice processing engine with robust error handling and recovery loops.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import dspy

from agent_definitions import (
    AgentResponse, ErrorContext, FRAUD_DETECTION_AGENTS,
    AgentSelectorSignature, SpecialistAgentSignature, ResultCompilerSignature,
    get_agent_prompt_template, get_default_agent_selection, get_all_agent_types
)
from error_validation import (
    ErrorValidator, ErrorAnalysis, ErrorType, ErrorSeverity,
    validate_and_fix_agent_response, create_error_context,
    robust_parse_integer, robust_extract_field, robust_parse_red_flags
)

log = logging.getLogger("invoice_processor")

@dataclass
class ProcessingResult:
    success: bool
    overall_risk: int
    recommendation: str
    summary: str
    top_concerns: List[str]
    next_steps: str
    agent_details: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    processing_time: float
    error_count: int
    retry_count: int

class InvoiceProcessor:
    def __init__(self, max_retries: int = 3, backoff_delay: float = 1.0):
        self.max_retries = max_retries
        self.backoff_delay = backoff_delay
        self.error_validator = ErrorValidator(max_retries)
        
        # Initialize DSPy and Google AI
        self.api_key = self._get_api_key()
        self.genai = self._setup_genai()
        self.lm = self._setup_dspy()
        
        # Processing state
        self.error_contexts: List[ErrorContext] = []
        self.retry_count = 0
        self.total_errors = 0
        
    def _get_api_key(self) -> str:
        """Get Google API key from environment"""
        load_dotenv()
        
        # Try main key first
        key = os.getenv("GOOGLE_API_KEY")
        if key and key.strip():
            return key.strip()
            
        # Try numbered keys
        for i in range(10):
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if key and key.strip():
                return key.strip()
                
        raise ValueError("No valid Google API key found. Set GOOGLE_API_KEY in .env file")

    def _setup_genai(self):
        """Initialize Google Generative AI"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            return genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    def _setup_dspy(self):
        """Initialize DSPy with Google AI"""
        try:
            # Configure DSPy to use Google AI
            lm = dspy.GoogleGenAI(model="gemini-2.5-flash", api_key=self.api_key)
            dspy.settings.configure(lm=lm)
            return lm
        except Exception as e:
            log.warning(f"DSPy setup failed: {e}. Falling back to direct API calls.")
            return None

    def _get_model(self):
        """Get preferred Gemini model"""
        try:
            return self.genai.GenerativeModel("models/gemini-2.5-flash")
        except:
            try:
                return self.genai.GenerativeModel("models/gemini-2.5-pro")
            except:
                # Fallback to any available model
                models = list(self.genai.list_models())
                if models:
                    return self.genai.GenerativeModel(models[0].name)
                raise RuntimeError("No available Gemini models found")

    def process_invoice(self, invoice_data: str) -> ProcessingResult:
        """Main processing method with error recovery loops"""
        start_time = time.time()
        self.retry_count = 0
        self.total_errors = 0
        self.error_contexts = []
        
        log.info("🔍 Starting invoice fraud analysis with error recovery...")
        
        while self.retry_count <= self.max_retries:
            try:
                # Step 1: Determine agents to summon
                agents_to_summon = self._determine_agents_with_recovery(invoice_data)
                if not agents_to_summon:
                    agents_to_summon = get_default_agent_selection()
                    log.warning("Using default agent selection due to failures")
                
                log.info(f"📋 Summoning {len(agents_to_summon)} agents: {', '.join(agents_to_summon)}")
                
                # Step 2: Run specialist agents with error recovery
                agent_responses = self._run_agents_with_recovery(agents_to_summon, invoice_data)
                
                # Step 3: Compile results with error recovery
                final_result = self._compile_results_with_recovery(invoice_data, agent_responses)
                
                # Success - return results
                processing_time = time.time() - start_time
                return ProcessingResult(
                    success=True,
                    overall_risk=final_result["overall_risk"],
                    recommendation=final_result["recommendation"],
                    summary=final_result["summary"],
                    top_concerns=final_result["top_concerns"],
                    next_steps=final_result["next_steps"],
                    agent_details=final_result["agent_details"],
                    statistics=final_result["statistics"],
                    processing_time=processing_time,
                    error_count=self.total_errors,
                    retry_count=self.retry_count
                )
                
            except Exception as e:
                self.total_errors += 1
                error_analysis = self.error_validator.analyze_error(e, "main_processing", [])
                
                if error_analysis.should_restart and self.retry_count < self.max_retries:
                    self.retry_count += 1
                    log.warning(f"⚠️ Processing failed, restarting attempt {self.retry_count}: {str(e)}")
                    
                    # Add error context for next attempt
                    self.error_contexts.append(ErrorContext(
                        error_type=str(error_analysis.error_type),
                        error_message=str(e),
                        retry_count=self.retry_count,
                        suggested_fix=error_analysis.suggested_fix
                    ))
                    
                    # Exponential backoff
                    time.sleep(self.backoff_delay * (2 ** self.retry_count))
                    continue
                else:
                    # Final failure
                    processing_time = time.time() - start_time
                    log.error(f"❌ Processing failed after {self.retry_count} retries: {str(e)}")
                    
                    return ProcessingResult(
                        success=False,
                        overall_risk=8,  # High risk due to processing failure
                        recommendation="REVIEW",
                        summary=f"Processing failed after {self.retry_count} attempts: {str(e)}",
                        top_concerns=["System processing error"],
                        next_steps="Manual review required due to system failure",
                        agent_details=[],
                        statistics={
                            "agents_consulted": 0,
                            "average_risk": 0,
                            "average_confidence": 0,
                            "total_red_flags": 0,
                            "processing_errors": self.total_errors
                        },
                        processing_time=processing_time,
                        error_count=self.total_errors,
                        retry_count=self.retry_count
                    )
        
        # Should never reach here, but handle gracefully
        processing_time = time.time() - start_time
        return ProcessingResult(
            success=False,
            overall_risk=8,
            recommendation="REVIEW",
            summary="Maximum retry attempts exceeded",
            top_concerns=["Processing timeout"],
            next_steps="Manual review required",
            agent_details=[],
            statistics={"processing_errors": self.total_errors},
            processing_time=processing_time,
            error_count=self.total_errors,
            retry_count=self.retry_count
        )

    def _determine_agents_with_recovery(self, invoice_data: str) -> List[str]:
        """Determine agents to summon with error recovery"""
        error_context = self._build_error_context_string("agent_selection")
        
        for attempt in range(self.max_retries):
            try:
                if self.lm:  # Use DSPy if available
                    return self._determine_agents_dspy(invoice_data, error_context)
                else:
                    return self._determine_agents_direct(invoice_data, error_context)
                    
            except Exception as e:
                self.total_errors += 1
                log.warning(f"Agent selection attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # Add error context for next attempt
                    error_context += f"\nPrevious attempt {attempt + 1} failed: {str(e)}"
                    time.sleep(self.backoff_delay)
                    continue
                else:
                    log.error("All agent selection attempts failed, using defaults")
                    return get_default_agent_selection()
        
        return get_default_agent_selection()

    def _determine_agents_dspy(self, invoice_data: str, error_context: str) -> List[str]:
        """Use DSPy signature for agent selection"""
        selector = dspy.Predict(AgentSelectorSignature)
        
        available_agents = json.dumps({
            name: config["description"] 
            for name, config in FRAUD_DETECTION_AGENTS.items()
        }, indent=2)
        
        result = selector(
            invoice_data=invoice_data,
            available_agents=available_agents,
            error_context=error_context
        )
        
        # Parse the selected agents
        try:
            if "[" in result.selected_agents and "]" in result.selected_agents:
                start = result.selected_agents.find("[")
                end = result.selected_agents.rfind("]") + 1
                agents_json = result.selected_agents[start:end]
                selected_agents = json.loads(agents_json)
                
                # Validate agents exist
                valid_agents = [a for a in selected_agents if a in FRAUD_DETECTION_AGENTS]
                if valid_agents:
                    log.info(f"DSPy agent selection reasoning: {result.reasoning}")
                    return valid_agents
            
            # Fallback parsing
            return self._fallback_parse_agents(result.selected_agents)
            
        except Exception as e:
            log.warning(f"DSPy agent parsing failed: {e}")
            return get_default_agent_selection()

    def _determine_agents_direct(self, invoice_data: str, error_context: str) -> List[str]:
        """Direct API call for agent selection"""
        model = self._get_model()
        
        agents_list = "\n".join([
            f"- {k}: {v['description']}" 
            for k, v in FRAUD_DETECTION_AGENTS.items()
        ])
        
        prompt = f"""
You are a fraud detection coordinator. Analyze this invoice data and determine which specialist agents should examine it.

INVOICE DATA:
{invoice_data}

AVAILABLE SPECIALIST AGENTS:
{agents_list}

{error_context}

Based on the invoice content, select 3-5 most relevant agents. Respond with ONLY a JSON array like: ["agent1", "agent2", "agent3"]

Consider what aspects seem most important to verify and what red flags might exist.
"""

        response = model.generate_content(prompt)
        agents_text = response.text.strip()
        
        # Parse JSON response
        if "[" in agents_text and "]" in agents_text:
            start = agents_text.find("[")
            end = agents_text.rfind("]") + 1
            agents_json = agents_text[start:end]
            selected_agents = json.loads(agents_json)
            
            # Validate agents exist
            valid_agents = [a for a in selected_agents if a in FRAUD_DETECTION_AGENTS]
            return valid_agents if valid_agents else get_default_agent_selection()
        
        return self._fallback_parse_agents(agents_text)

    def _fallback_parse_agents(self, text: str) -> List[str]:
        """Fallback parsing for agent selection"""
        all_agents = get_all_agent_types()
        found_agents = []
        
        text_lower = text.lower()
        for agent in all_agents:
            if agent in text_lower:
                found_agents.append(agent)
        
        return found_agents[:5] if found_agents else get_default_agent_selection()

    def _run_agents_with_recovery(self, agents: List[str], invoice_data: str) -> List[AgentResponse]:
        """Run specialist agents with individual error recovery"""
        agent_responses = []
        
        for agent_type in agents:
            for attempt in range(self.max_retries):
                try:
                    response = self._run_single_agent_with_recovery(agent_type, invoice_data, attempt)
                    agent_responses.append(response)
                    log.info(f"✅ {agent_type}: Risk {response.risk_score}/10, Confidence {response.confidence}/10")
                    break
                    
                except Exception as e:
                    self.total_errors += 1
                    log.warning(f"Agent {agent_type} attempt {attempt + 1} failed: {str(e)}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_delay)
                        continue
                    else:
                        # Create fallback response
                        fallback_response = AgentResponse(
                            agent_type=agent_type,
                            analysis=f"Agent analysis failed after {self.max_retries} attempts: {str(e)}",
                            risk_score=6,  # Medium-high risk due to uncertainty
                            confidence=2,   # Low confidence
                            red_flags=[f"Agent processing error: {str(e)}"],
                            error_count=self.max_retries
                        )
                        agent_responses.append(fallback_response)
                        log.error(f"❌ {agent_type} failed permanently, using fallback response")
        
        return agent_responses

    def _run_single_agent_with_recovery(self, agent_type: str, invoice_data: str, attempt: int) -> AgentResponse:
        """Run single agent with error recovery"""
        error_context = create_error_context(
            [ctx.error_message for ctx in self.error_contexts if ctx.failed_agent == agent_type],
            f"agent_{agent_type}",
            attempt
        )
        
        if self.lm:
            return self._run_agent_dspy(agent_type, invoice_data, error_context)
        else:
            return self._run_agent_direct(agent_type, invoice_data, error_context)

    def _run_agent_dspy(self, agent_type: str, invoice_data: str, error_context: str) -> AgentResponse:
        """Run agent using DSPy signature"""
        agent = dspy.Predict(SpecialistAgentSignature)
        
        agent_config = FRAUD_DETECTION_AGENTS[agent_type]
        prompt_template = get_agent_prompt_template(agent_type)
        
        result = agent(
            invoice_data=invoice_data,
            agent_role=f"{agent_config['name']}: {agent_config['description']}",
            focus_area=prompt_template + error_context
        )
        
        # Validate and create response
        response_text = f"ANALYSIS: {result.analysis}\nRISK_SCORE: {result.risk_score}\nCONFIDENCE: {result.confidence}\nRED_FLAGS: {result.red_flags}"
        
        agent_response, errors = validate_and_fix_agent_response(response_text, agent_type)
        
        if errors:
            log.warning(f"DSPy agent {agent_type} had validation errors: {errors}")
        
        return agent_response

    def _run_agent_direct(self, agent_type: str, invoice_data: str, error_context: str) -> AgentResponse:
        """Run agent using direct API call"""
        model = self._get_model()
        agent_config = FRAUD_DETECTION_AGENTS[agent_type]
        prompt_template = get_agent_prompt_template(agent_type)
        
        prompt = f"""
{prompt_template}

INVOICE DATA TO ANALYZE:
{invoice_data}

{error_context}

Analyze this invoice data focusing ONLY on your area of expertise. Provide your response in this EXACT format:

ANALYSIS: [your 2-3 sentence analysis]
RISK_SCORE: [single number 1-10]
CONFIDENCE: [single number 1-10]
RED_FLAGS: [comma-separated list of flags, or "None"]

Be precise with the format - use exactly these field names and provide clean numeric values.
"""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Validate and fix response
        agent_response, errors = validate_and_fix_agent_response(response_text, agent_type)
        
        if errors:
            log.warning(f"Direct agent {agent_type} had validation errors: {errors}")
        
        return agent_response

    def _compile_results_with_recovery(self, invoice_data: str, agent_responses: List[AgentResponse]) -> Dict[str, Any]:
        """Compile results with error recovery"""
        error_context = self._build_error_context_string("compilation")
        
        for attempt in range(self.max_retries):
            try:
                if self.lm:
                    return self._compile_results_dspy(invoice_data, agent_responses, error_context)
                else:
                    return self._compile_results_direct(invoice_data, agent_responses, error_context)
                    
            except Exception as e:
                self.total_errors += 1
                log.warning(f"Results compilation attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    error_context += f"\nPrevious compilation attempt {attempt + 1} failed: {str(e)}"
                    time.sleep(self.backoff_delay)
                    continue
                else:
                    # Fallback compilation
                    return self._create_fallback_compilation(agent_responses, str(e))
        
        return self._create_fallback_compilation(agent_responses, "Max retries exceeded")

    def _compile_results_dspy(self, invoice_data: str, agent_responses: List[AgentResponse], error_context: str) -> Dict[str, Any]:
        """Compile results using DSPy signature"""
        compiler = dspy.Predict(ResultCompilerSignature)
        
        # Prepare agent analyses summary
        agent_summary = self._format_agent_summary(agent_responses)
        statistics = self._calculate_statistics(agent_responses)
        stats_text = json.dumps(statistics, indent=2)
        
        result = compiler(
            invoice_data=invoice_data,
            agent_analyses=agent_summary + error_context,
            statistics=stats_text
        )
        
        # Parse and validate results
        top_concerns = robust_parse_red_flags(result.top_concerns)
        
        return {
            "overall_risk": max(1, min(10, int(result.overall_risk))),
            "recommendation": result.recommendation.upper(),
            "summary": result.summary,
            "top_concerns": top_concerns,
            "next_steps": result.next_steps,
            "agent_details": [self._format_agent_detail(r) for r in agent_responses],
            "statistics": statistics
        }

    def _compile_results_direct(self, invoice_data: str, agent_responses: List[AgentResponse], error_context: str) -> Dict[str, Any]:
        """Compile results using direct API call"""
        model = self._get_model()
        
        agent_summary = self._format_agent_summary(agent_responses)
        statistics = self._calculate_statistics(agent_responses)
        
        prompt = f"""
You are the final fraud assessment compiler. Synthesize the specialist agent analyses into a comprehensive fraud assessment.

ORIGINAL INVOICE DATA:
{invoice_data}

SPECIALIST AGENT ANALYSES:
{agent_summary}

STATISTICAL SUMMARY:
{json.dumps(statistics, indent=2)}

{error_context}

Provide a final assessment in this EXACT format:

OVERALL_RISK: [single number 1-10]
RECOMMENDATION: [APPROVE, REVIEW, or REJECT]
SUMMARY: [2-3 sentence summary]
TOP_CONCERNS: [comma-separated list of top 3 concerns, or "None"]
NEXT_STEPS: [recommended actions]

Be precise with the format and provide clean, parseable values.
"""

        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Parse response with robust extraction
        overall_risk_text = robust_extract_field(text, "OVERALL_RISK", "5")
        overall_risk, _ = robust_parse_integer(overall_risk_text, "overall_risk", 5)
        
        recommendation = robust_extract_field(text, "RECOMMENDATION", "REVIEW").upper()
        if recommendation not in ["APPROVE", "REVIEW", "REJECT"]:
            recommendation = "REVIEW"
        
        summary = robust_extract_field(text, "SUMMARY", "Analysis completed")
        top_concerns_text = robust_extract_field(text, "TOP_CONCERNS", "None")
        top_concerns = robust_parse_red_flags(top_concerns_text)
        next_steps = robust_extract_field(text, "NEXT_STEPS", "Manual review recommended")
        
        return {
            "overall_risk": overall_risk,
            "recommendation": recommendation,
            "summary": summary,
            "top_concerns": top_concerns,
            "next_steps": next_steps,
            "agent_details": [self._format_agent_detail(r) for r in agent_responses],
            "statistics": statistics
        }

    def _format_agent_summary(self, agent_responses: List[AgentResponse]) -> str:
        """Format agent responses for compilation"""
        summary = ""
        for response in agent_responses:
            summary += f"\n{response.agent_type.upper()}:\n"
            summary += f"  Analysis: {response.analysis}\n"
            summary += f"  Risk Score: {response.risk_score}/10\n"
            summary += f"  Confidence: {response.confidence}/10\n"
            summary += f"  Red Flags: {', '.join(response.red_flags) if response.red_flags else 'None'}\n"
        return summary

    def _calculate_statistics(self, agent_responses: List[AgentResponse]) -> Dict[str, Any]:
        """Calculate statistical summary"""
        if not agent_responses:
            return {
                "agents_consulted": 0,
                "average_risk": 0,
                "average_confidence": 0,
                "total_red_flags": 0
            }
        
        total_risk = sum(r.risk_score for r in agent_responses)
        total_confidence = sum(r.confidence for r in agent_responses)
        all_red_flags = [flag for r in agent_responses for flag in r.red_flags]
        
        return {
            "agents_consulted": len(agent_responses),
            "average_risk": round(total_risk / len(agent_responses), 1),
            "average_confidence": round(total_confidence / len(agent_responses), 1),
            "total_red_flags": len(all_red_flags)
        }

    def _format_agent_detail(self, response: AgentResponse) -> Dict[str, Any]:
        """Format agent response for output"""
        return {
            "agent": response.agent_type,
            "risk_score": response.risk_score,
            "confidence": response.confidence,
            "analysis": response.analysis,
            "red_flags": response.red_flags,
            "error_count": getattr(response, 'error_count', 0)
        }

    def _create_fallback_compilation(self, agent_responses: List[AgentResponse], error_msg: str) -> Dict[str, Any]:
        """Create fallback compilation when all attempts fail"""
        statistics = self._calculate_statistics(agent_responses)
        
        return {
            "overall_risk": int(statistics.get("average_risk", 6)),
            "recommendation": "REVIEW",
            "summary": f"Compilation failed: {error_msg}. Based on {len(agent_responses)} agent responses.",
            "top_concerns": ["System compilation error"],
            "next_steps": "Manual review required due to compilation failure",
            "agent_details": [self._format_agent_detail(r) for r in agent_responses],
            "statistics": statistics
        }

    def _build_error_context_string(self, step: str) -> str:
        """Build error context string for current step"""
        relevant_errors = [e for e in self.error_contexts if step in e.error_type.lower()]
        if not relevant_errors:
            return ""
        
        context = f"\nERROR CONTEXT for {step}:\n"
        for error in relevant_errors[-3:]:  # Last 3 errors only
            context += f"- Previous attempt failed: {error.error_message}\n"
            if error.suggested_fix:
                context += f"  Suggested fix: {error.suggested_fix}\n"
        
        context += "Please ensure proper formatting and avoid these issues.\n"
        return context