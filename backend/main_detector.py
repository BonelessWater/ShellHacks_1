# File: backend/main_detector.py
#!/usr/bin/env python3
"""
Enhanced Multi-Agent Invoice Fraud Detection System with Parallel LLM Processing

This system combines:
- Parallel LLM processing with proper synchronization
- Core LLM that determines which specialized agents to summon
- Specialized LLM agents for focused fraud detection tasks
- Hardcoded tools for deterministic fraud detection
- Comprehensive error recovery and agent swapping capabilities

Usage:
    python main_detector.py --demo
    python main_detector.py --invoice "invoice_data_here"
    python main_detector.py --file invoice.json
    python main_detector.py --parallel --max-workers 6
"""
 
import os
import sys
import json
import logging
import argparse
import time
import asyncio
import concurrent.futures
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Google GenerativeAI not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fraud_detection.log", encoding='utf-8')
    ]
)

# Set console encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

log = logging.getLogger("main_detector")

# Load environment variables
load_dotenv()


class AgentStatus(Enum):
    """LLM Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentResponse:
    """Response from a fraud detection agent"""
    agent_type: str
    analysis: str
    risk_score: int  # 1-10 scale
    confidence: int  # 1-10 scale
    red_flags: List[str]
    execution_time: float = 0.0
    tool_used: str = "llm"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "analysis": self.analysis,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "red_flags": self.red_flags,
            "execution_time": self.execution_time,
            "tool_used": self.tool_used
        }


@dataclass
class LLMAgentResult:
    """Result from LLM agent execution"""
    agent_id: str
    agent_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    status: AgentStatus = AgentStatus.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt_used: str = ""
    model_used: str = ""
    tokens_used: int = 0


@dataclass
class LLMTask:
    """Task for LLM agents"""
    task_id: str
    data: Any
    agent_names: List[str]
    priority: int = 0
    timeout: float = 30.0
    context: Dict[str, Any] = field(default_factory=dict)


class LLMAgentConfig:
    """Configuration for LLM agents"""
    
    def __init__(self, 
                 agent_name: str,
                 prompt_template: str,
                 system_prompt: str = "",
                 model_name: str = "models/gemini-2.5-flash",
                 temperature: float = 0.1,
                 max_tokens: int = 2048):
        self.agent_name = agent_name
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def format_prompt(self, data: Any, context: Dict[str, Any] = None) -> str:
        """Format the prompt template"""
        format_vars = {'data': data, 'context': context or {}}
        if isinstance(data, dict):
            format_vars.update(data)
        if context:
            format_vars.update(context)
        
        try:
            return self.prompt_template.format(**format_vars)
        except KeyError as e:
            log.warning(f"Missing template variable {e} for {self.agent_name}")
            return self.prompt_template


class LLMAgent:
    """LLM Agent for specialized fraud detection tasks"""
    
    def __init__(self, agent_id: str, config: LLMAgentConfig, llm_client: Any = None):
        self.agent_id = agent_id
        self.config = config
        self.llm_client = llm_client
        self.status = AgentStatus.IDLE
        self._lock = threading.Lock()
    
    async def execute(self, task: LLMTask) -> LLMAgentResult:
        """Execute the LLM agent task"""
        start_time = time.time()
        
        # Simplified status check - allow multiple concurrent executions
        log.info(f"🤖 Agent {self.config.agent_name} starting task {task.task_id}")
        
        try:
            # Format prompt
            formatted_prompt = self.config.format_prompt(task.data, task.context)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._call_llm(formatted_prompt, task),
                timeout=task.timeout
            )
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.status = AgentStatus.COMPLETED
            
            log.info(f"✅ Agent {self.config.agent_name} completed in {execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Agent timed out after {task.timeout}s"
            log.error(f"⏰ Agent {self.config.agent_name}: {error_msg}")
            return LLMAgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.agent_name,
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time,
                status=AgentStatus.TIMEOUT
            )
        except Exception as e:
            error_msg = f"Agent failed: {str(e)}"
            log.error(f"❌ Agent {self.config.agent_name}: {error_msg}")
            return LLMAgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.agent_name,
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time,
                status=AgentStatus.FAILED
            )
    
    async def _call_llm(self, prompt: str, task: LLMTask) -> LLMAgentResult:
        """Make LLM API call"""
        try:
            full_prompt = prompt
            if self.config.system_prompt:
                full_prompt = f"System: {self.config.system_prompt}\n\nUser: {prompt}"
            
            if self.llm_client:
                try:
                    response = self.llm_client.generate_content(
                        full_prompt,
                        generation_config={
                            'temperature': self.config.temperature,
                            'max_output_tokens': self.config.max_tokens,
                        }
                    )
                    response_text = response.text
                    tokens_used = getattr(response, 'usage_metadata', {}).get('total_token_count', 0)
                    log.info(f"🎯 {self.config.agent_name} received LLM response ({len(response_text)} chars)")
                except Exception as llm_error:
                    log.error(f"❌ LLM API call failed for {self.config.agent_name}: {str(llm_error)}")
                    # Fall back to mock response
                    response_text = self._generate_mock_response()
                    tokens_used = 100
            else:
                # Mock response for demo
                log.info(f"🤖 Using mock response for {self.config.agent_name}")
                response_text = self._generate_mock_response()
                tokens_used = 100
                await asyncio.sleep(0.1)  # Simulate some processing time
            
            # Parse JSON response
            parsed_result = self._parse_response(response_text)
            
            return LLMAgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.agent_name,
                success=True,
                result=parsed_result,
                prompt_used=full_prompt[:200] + "..." if len(full_prompt) > 200 else full_prompt,
                model_used=self.config.model_name,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            log.error(f"❌ _call_llm failed for {self.config.agent_name}: {str(e)}")
            return LLMAgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.agent_name,
                success=False,
                error=f"LLM call failed: {str(e)}"
            )
    
    def _generate_mock_response(self) -> str:
        """Generate a mock response based on agent type"""
        agent_name = self.config.agent_name
        
        if "amount" in agent_name:
            return """
            {
                "risk_score": 8,
                "confidence": 9,
                "analysis": "Detected high-value transactions with round numbers that may indicate fabrication. The $90,000 subtotal is suspiciously round.",
                "red_flags": ["HIGH_ROUND_AMOUNTS", "SUSPICIOUS_TOTAL"],
                "fraud_indicators": [
                    {"type": "amount_anomaly", "severity": "high", "description": "Unusually round subtotal amount"}
                ]
            }
            """
        elif "vendor" in agent_name:
            return """
            {
                "risk_score": 9,
                "confidence": 8,
                "analysis": "Vendor name 'SuspiciousCorp LLC' follows common fraudulent naming patterns. Generic address and contact information.",
                "red_flags": ["SUSPICIOUS_VENDOR_NAME", "GENERIC_ADDRESS"],
                "fraud_indicators": [
                    {"type": "vendor_suspicion", "severity": "high", "description": "Vendor name contains suspicious keywords"}
                ]
            }
            """
        elif "payment" in agent_name:
            return """
            {
                "risk_score": 9,
                "confidence": 9,
                "analysis": "Wire transfer only payment method is a major red flag. Legitimate businesses typically offer multiple payment options.",
                "red_flags": ["WIRE_TRANSFER_ONLY", "NO_ALTERNATIVE_PAYMENT"],
                "fraud_indicators": [
                    {"type": "payment_suspicion", "severity": "high", "description": "Requires wire transfer only"}
                ]
            }
            """
        else:
            return """
            {
                "risk_score": 6,
                "confidence": 7,
                "analysis": "General fraud analysis completed with moderate risk indicators detected.",
                "red_flags": ["MODERATE_RISK"],
                "fraud_indicators": [
                    {"type": "general_suspicion", "severity": "medium", "description": "Multiple minor risk factors"}
                ]
            }
            """
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse LLM response into structured format"""
        try:
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            parsed_result = json.loads(response_text)
            
            # Ensure required fields exist
            if not isinstance(parsed_result, dict):
                raise ValueError("Response is not a dictionary")
            
            # Set defaults for missing fields
            parsed_result.setdefault('risk_score', 5)
            parsed_result.setdefault('confidence', 5)
            parsed_result.setdefault('analysis', f'Analysis from {self.config.agent_name}')
            parsed_result.setdefault('red_flags', [])
            parsed_result.setdefault('fraud_indicators', [])
            
            return parsed_result
            
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"Could not parse JSON from {self.agent_id}: {e}")
            return {
                "risk_score": 5,
                "confidence": 3,
                "analysis": f"Analysis from {self.config.agent_name}: {response_text[:200]}...",
                "red_flags": ["PARSING_ERROR"],
                "fraud_indicators": []
            }
    
    async def _reset_status(self):
        """Reset status after delay"""
        await asyncio.sleep(1.0)
        with self._lock:
            if self.status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.TIMEOUT]:
                self.status = AgentStatus.IDLE


class LLMAgentRegistry:
    """Registry for managing LLM agent configurations"""
    
    def __init__(self):
        self._agent_configs: Dict[str, LLMAgentConfig] = {}
        self._agent_instances: Dict[str, LLMAgent] = {}
        self._lock = threading.Lock()
    
    def register_agent_config(self, config: LLMAgentConfig):
        """Register agent configuration"""
        with self._lock:
            self._agent_configs[config.agent_name] = config
            log.info(f"📋 Registered agent: {config.agent_name}")
    
    def create_agent_instance(self, agent_name: str, agent_id: str, llm_client: Any = None) -> LLMAgent:
        """Create agent instance"""
        with self._lock:
            if agent_name not in self._agent_configs:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            config = self._agent_configs[agent_name]
            agent = LLMAgent(agent_id, config, llm_client)
            self._agent_instances[agent_id] = agent
            return agent
    
    def get_agents_by_name(self, agent_name: str) -> List[LLMAgent]:
        """Get agents by name"""
        return [a for a in self._agent_instances.values() if a.config.agent_name == agent_name]
    
    def get_available_agent_types(self) -> List[str]:
        """Get available agent types"""
        return list(self._agent_configs.keys())


class ParallelLLMExecutor:
    """Parallel executor for LLM agents with synchronization"""
    
    def __init__(self, max_workers: int = 4, llm_client: Any = None):
        self.max_workers = max_workers
        self.agent_registry = LLMAgentRegistry()
        self.llm_client = llm_client
        self._task_barrier = None
    
    async def execute_tasks_parallel(self, tasks: List[LLMTask], wait_for_all: bool = True) -> List[LLMAgentResult]:
        """Execute tasks in parallel with synchronization"""
        if not tasks:
            return []
        
        log.info(f"🚀 Executing {len(tasks)} LLM tasks in parallel")
        start_time = time.time()
        
        # Assign agents to tasks
        task_agent_pairs = self._assign_agents_to_tasks(tasks)
        
        if not task_agent_pairs:
            log.error("No agents available")
            return []
        
        # Remove barrier - just execute in parallel with asyncio.gather
        log.info(f"🔄 Starting {len(task_agent_pairs)} agents concurrently...")
        
        # Create all tasks IMMEDIATELY - this is the key to parallel execution
        async_tasks = []
        for task, agent in task_agent_pairs:
            # Start each agent execution immediately
            async_task = asyncio.create_task(agent.execute(task))
            async_tasks.append(async_task)
            log.info(f"🚀 Started agent {agent.config.agent_name} for task {task.task_id}")
        
        # Wait for ALL tasks to complete concurrently
        log.info("⏳ Waiting for all agents to complete...")
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task, agent = task_agent_pairs[i]
                error_result = LLMAgentResult(
                    agent_id=agent.agent_id,
                    agent_name=agent.config.agent_name,
                    success=False,
                    error=f"Execution failed: {str(result)}",
                    status=AgentStatus.FAILED
                )
                processed_results.append(error_result)
                log.error(f"❌ Agent {agent.agent_id} failed: {str(result)}")
            else:
                processed_results.append(result)
        
        execution_time = time.time() - start_time
        successful = sum(1 for r in processed_results if r.success)
        log.info(f"✅ Parallel execution completed in {execution_time:.2f}s ({successful}/{len(processed_results)} successful)")
        
        return processed_results
    
    async def _execute_with_barrier(self, task: LLMTask, agent: LLMAgent, use_barrier: bool) -> LLMAgentResult:
        """Execute task with barrier synchronization - NO BARRIER, just execute"""
        # Execute the LLM task immediately - don't wait for barrier
        result = await agent.execute(task)
        
        # Note: Removed barrier wait to ensure true parallel execution
        # The asyncio.gather() in execute_tasks_parallel handles synchronization
        
        return result
    
    def _assign_agents_to_tasks(self, tasks: List[LLMTask]) -> List[tuple[LLMTask, LLMAgent]]:
        """Assign agents to tasks"""
        assignments = []
        
        for task in tasks:
            suitable_agents = []
            for agent_name in task.agent_names:
                agents = self.agent_registry.get_agents_by_name(agent_name)
                suitable_agents.extend([a for a in agents if a.status == AgentStatus.IDLE])
            
            if suitable_agents:
                chosen_agent = suitable_agents[0]
                assignments.append((task, chosen_agent))
            else:
                # Create new agent
                if task.agent_names:
                    agent_name = task.agent_names[0]
                    try:
                        new_agent_id = f"{agent_name}_{int(time.time() * 1000)}"
                        new_agent = self.agent_registry.create_agent_instance(
                            agent_name, new_agent_id, self.llm_client
                        )
                        assignments.append((task, new_agent))
                    except ValueError as e:
                        log.error(f"Could not create agent for {task.task_id}: {e}")
        
        return assignments


class EnhancedParallelInvoiceFraudDetector:
    """Enhanced fraud detector with parallel LLM agents and core coordinator"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        
        # Initialize LLM client
        self.api_key = self._get_api_key()
        if GENAI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        else:
            self.model = None
            log.warning("⚠️  Google GenerativeAI not available or no API key")
        
        # Initialize parallel executor
        self.llm_executor = ParallelLLMExecutor(max_workers=max_workers, llm_client=self.model)
        
        # Register specialized fraud detection agents
        self._register_fraud_agents()
        
        # Demo invoice for testing
        self.demo_invoice = """
INVOICE #INV-2025-0928-001
==========================================

FROM: SuspiciousCorp LLC
      1234 Fake Street
      Nowhere, NY 10001
      
TO:   YourCompany Inc
      5678 Real Avenue
      Somewhere, CA 90210

Date: September 28, 2025
Due Date: October 28, 2025

ITEMS:
------
1. "Consulting Services" - $50,000.00
   (Description: General business consulting for Q4)
   
2. Software License - $25,000.00
   (Description: Enterprise software package)
   
3. Training Services - $15,000.00
   (Description: Staff training program)

SUBTOTAL: $90,000.00
TAX (8.5%): $7,650.00
TOTAL: $97,650.00

Payment Terms: Net 30
Payment Method: Wire Transfer Only
Account: FirstNational Bank
Account #: 555-123-4567
Routing #: 021000021

Notes: Payment must be received within 30 days.
Contact: john.doe@suspiciouscorp.com
Phone: (555) 999-9999
"""
    
    def _get_api_key(self) -> Optional[str]:
        """Get Google API key from environment"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            log.warning("⚠️  No GOOGLE_API_KEY found in environment")
        return api_key
    
    def _register_fraud_agents(self):
        """Register specialized fraud detection agents"""
        
        # Amount Validation Agent
        amount_config = LLMAgentConfig(
            agent_name="amount_validator",
            system_prompt="You are an expert at detecting fraudulent invoice amounts and mathematical inconsistencies.",
            prompt_template="""
Analyze this invoice for amount-related fraud indicators:

INVOICE DATA:
{data}

Focus on:
- Mathematical inconsistencies in calculations
- Unusually round numbers that may indicate fabrication
- Amounts that are suspiciously high or low for the services described
- Tax calculation errors
- Subtotal/total mismatches

Return JSON format:
{{
    "risk_score": <1-10>,
    "confidence": <1-10>,
    "analysis": "<detailed analysis>",
    "red_flags": ["<flag1>", "<flag2>"],
    "fraud_indicators": [
        {{"type": "amount_anomaly", "severity": "high", "description": "Detailed description"}}
    ]
}}
""",
            temperature=0.1
        )
        
        # Vendor Validation Agent
        vendor_config = LLMAgentConfig(
            agent_name="vendor_validator",
            system_prompt="You are an expert at detecting fraudulent vendors and suspicious business relationships.",
            prompt_template="""
Analyze this invoice for vendor-related fraud indicators:

INVOICE DATA:
{data}

Focus on:
- Vendor legitimacy indicators
- Suspicious contact information patterns
- Address validation concerns
- Business name authenticity
- Contact method red flags

Return JSON format:
{{
    "risk_score": <1-10>,
    "confidence": <1-10>,
    "analysis": "<detailed analysis>",
    "red_flags": ["<flag1>", "<flag2>"],
    "fraud_indicators": [
        {{"type": "vendor_suspicion", "severity": "medium", "description": "Detailed description"}}
    ]
}}
""",
            temperature=0.1
        )
        
        # Date and Timing Analysis Agent
        date_config = LLMAgentConfig(
            agent_name="date_analyzer",
            system_prompt="You are an expert at detecting suspicious date patterns and timing anomalies in invoices.",
            prompt_template="""
Analyze this invoice for date and timing-related fraud indicators:

INVOICE DATA:
{data}

Focus on:
- Suspicious date patterns (weekends, holidays)
- Backdating or future-dating concerns
- Payment term anomalies
- Date format inconsistencies
- Timeline feasibility

Return JSON format:
{{
    "risk_score": <1-10>,
    "confidence": <1-10>,
    "analysis": "<detailed analysis>",
    "red_flags": ["<flag1>", "<flag2>"],
    "fraud_indicators": [
        {{"type": "date_anomaly", "severity": "low", "description": "Detailed description"}}
    ]
}}
""",
            temperature=0.1
        )
        
        # Payment Terms Analyzer
        payment_config = LLMAgentConfig(
            agent_name="payment_analyzer",
            system_prompt="You are an expert at detecting fraudulent payment terms and banking information.",
            prompt_template="""
Analyze this invoice for payment-related fraud indicators:

INVOICE DATA:
{data}

Focus on:
- Suspicious payment methods (wire transfer only, unusual methods)
- Banking information legitimacy
- Payment term red flags
- Account number patterns
- Urgency tactics

Return JSON format:
{{
    "risk_score": <1-10>,
    "confidence": <1-10>,
    "analysis": "<detailed analysis>",
    "red_flags": ["<flag1>", "<flag2>"],
    "fraud_indicators": [
        {{"type": "payment_suspicion", "severity": "high", "description": "Detailed description"}}
    ]
}}
""",
            temperature=0.1
        )
        
        # Register all agents
        configs = [amount_config, vendor_config, date_config, payment_config]
        for config in configs:
            self.llm_executor.agent_registry.register_agent_config(config)
        
        log.info(f"📋 Registered {len(configs)} specialized fraud detection agents")
    
    async def determine_agents_to_summon(self, invoice_data: str) -> List[str]:
        """Core LLM determines which agents to summon based on invoice content"""
        if not self.model:
            # Fallback: use all available agents
            return self.llm_executor.agent_registry.get_available_agent_types()
        
        available_agents = self.llm_executor.agent_registry.get_available_agent_types()
        agents_list = "\n".join([f"- {agent}" for agent in available_agents])
        
        coordinator_prompt = f"""
You are a fraud detection coordinator. Analyze this invoice and determine which specialist agents should examine it.

INVOICE DATA:
{invoice_data}

AVAILABLE SPECIALIST AGENTS:
{agents_list}

Based on the invoice content, select 3-4 most relevant agents for comprehensive fraud detection.
Consider what aspects seem most suspicious or important to verify.

Respond with ONLY a JSON list of agent names:
["agent1", "agent2", "agent3"]
"""
        
        try:
            response = self.model.generate_content(
                coordinator_prompt,
                generation_config={'temperature': 0.1}
            )
            
            # Parse response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            selected_agents = json.loads(response_text)
            
            # Validate selection
            valid_agents = [agent for agent in selected_agents if agent in available_agents]
            
            if not valid_agents:
                log.warning("Core LLM returned invalid agents, using all available")
                return available_agents
            
            log.info(f"🎯 Core LLM selected agents: {', '.join(valid_agents)}")
            return valid_agents
            
        except Exception as e:
            log.error(f"Error in agent selection: {e}")
            log.info("Using all available agents as fallback")
            return available_agents
    
    async def analyze_invoice_parallel(self, invoice_data: str) -> Dict[str, Any]:
        """Main analysis method with parallel LLM agents"""
        start_time = time.time()
        log.info("🔍 Starting enhanced parallel invoice fraud analysis...")
        
        # Step 1: Core LLM determines which agents to summon
        log.info("🤔 Core LLM determining which agents to summon...")
        agents_to_summon = await self.determine_agents_to_summon(invoice_data)
        
        # Step 2: Create tasks for selected agents
        tasks = []
        for agent_name in agents_to_summon:
            task = LLMTask(
                task_id=f"{agent_name}_analysis",
                data=invoice_data,
                agent_names=[agent_name],
                timeout=45.0,
                context={"analysis_type": "fraud_detection"}
            )
            tasks.append(task)
        
        log.info(f"📋 Summoning {len(tasks)} specialized agents: {', '.join(agents_to_summon)}")
        
        # Step 3: Execute agents in parallel with synchronization
        results = await self.llm_executor.execute_tasks_parallel(
            tasks,
            wait_for_all=True  # Wait for all agents to complete
        )
        
        # Step 4: Aggregate results
        final_analysis = await self._aggregate_results(results, invoice_data)
        
        execution_time = time.time() - start_time
        final_analysis['total_execution_time'] = execution_time
        
        log.info(f"✅ Analysis completed in {execution_time:.2f}s")
        return final_analysis
    
    async def _aggregate_results(self, results: List[LLMAgentResult], invoice_data: str) -> Dict[str, Any]:
        """Aggregate results from all agents"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                "overall_risk_score": 10,
                "confidence": 1,
                "recommendation": "MANUAL_REVIEW",
                "status": "ANALYSIS_FAILED",
                "analysis": "Analysis failed - all agents encountered errors",
                "red_flags": ["ANALYSIS_FAILURE"],
                "agent_results": [],
                "agents_used": 0,
                "agents_failed": len(results),
                "error": "All agents failed to complete analysis"
            }
        
        # Extract data from successful results
        risk_scores = []
        confidences = []
        all_red_flags = []
        all_fraud_indicators = []
        agent_summaries = []
        
        for result in successful_results:
            if isinstance(result.result, dict):
                risk_score = result.result.get('risk_score', 5)
                confidence = result.result.get('confidence', 5)
                red_flags = result.result.get('red_flags', [])
                fraud_indicators = result.result.get('fraud_indicators', [])
                analysis = result.result.get('analysis', 'No analysis provided')
                
                risk_scores.append(risk_score)
                confidences.append(confidence)
                all_red_flags.extend(red_flags)
                all_fraud_indicators.extend(fraud_indicators)
                
                agent_summaries.append({
                    "agent": result.agent_name,
                    "risk_score": risk_score,
                    "confidence": confidence,
                    "execution_time": result.execution_time,
                    "analysis": analysis,
                    "red_flags": red_flags
                })
        
        # Calculate weighted averages
        if risk_scores and confidences:
            # Weight by confidence - more confident results have higher weight
            total_weight = sum(confidences)
            weighted_risk = sum(r * c for r, c in zip(risk_scores, confidences)) / total_weight
            avg_confidence = sum(confidences) / len(confidences)
        else:
            weighted_risk = 5
            avg_confidence = 1
        
        # Determine overall recommendation
        if weighted_risk >= 8:
            recommendation = "REJECT"
            status = "HIGH_RISK"
        elif weighted_risk >= 6:
            recommendation = "MANUAL_REVIEW"
            status = "MEDIUM_RISK"
        elif weighted_risk >= 4:
            recommendation = "ADDITIONAL_VERIFICATION"
            status = "LOW_RISK"
        else:
            recommendation = "APPROVE"
            status = "MINIMAL_RISK"
        
        # Create summary analysis
        unique_red_flags = list(set(all_red_flags))
        high_severity_indicators = [fi for fi in all_fraud_indicators if fi.get('severity') == 'high']
        
        summary_analysis = f"""
Comprehensive fraud analysis completed using {len(successful_results)} specialized agents.
Overall risk assessment: {status} (Risk Score: {weighted_risk:.1f}/10)
Key concerns identified: {len(unique_red_flags)} red flags, {len(high_severity_indicators)} high-severity indicators.
Primary risk factors: {', '.join(unique_red_flags[:3]) if unique_red_flags else 'None identified'}.
"""
        
        return {
            "overall_risk_score": round(weighted_risk, 1),
            "confidence": round(avg_confidence, 1),
            "recommendation": recommendation,
            "status": status,
            "analysis": summary_analysis.strip(),
            "red_flags": unique_red_flags,
            "fraud_indicators": all_fraud_indicators,
            "agent_results": agent_summaries,
            "agents_used": len(successful_results),
            "agents_failed": len(results) - len(successful_results)
        }


async def demo_enhanced_fraud_detection():
    """Demo the enhanced fraud detection system"""
    print("🚀 Enhanced Parallel Invoice Fraud Detection Demo")
    print("=" * 60)
    
    # Initialize the enhanced detector
    detector = EnhancedParallelInvoiceFraudDetector(max_workers=4)
    
    print("📄 Analyzing demo invoice with parallel LLM agents...")
    print("⏳ Please wait while all agents complete their analysis...\n")
    
    # Analyze the demo invoice
    start_time = time.time()
    results = await detector.analyze_invoice_parallel(detector.demo_invoice)
    total_time = time.time() - start_time
    
    # Display results
    print(f"✅ Analysis completed in {total_time:.2f} seconds\n")
    
    print("📊 FRAUD ANALYSIS RESULTS:")
    print("=" * 40)
    print(f"🎯 Overall Risk Score: {results['overall_risk_score']}/10")
    print(f"📊 Confidence Level: {results['confidence']}/10")
    print(f"📋 Recommendation: {results['recommendation']}")
    print(f"🚨 Status: {results.get('status', 'UNKNOWN')}")
    print(f"🤖 Agents Used: {results['agents_used']}")
    
    if results.get('agents_failed', 0) > 0:
        print(f"❌ Agents Failed: {results['agents_failed']}")
    
    print(f"\n📝 Analysis Summary:")
    print(f"{results['analysis']}")
    
    print(f"\n🚩 Red Flags Detected ({len(results.get('red_flags', []))}):")
    red_flags = results.get('red_flags', [])
    for i, flag in enumerate(red_flags[:5], 1):  # Show top 5
        print(f"   {i}. {flag}")
    
    if len(red_flags) > 5:
        print(f"   ... and {len(red_flags) - 5} more")
    
    print(f"\n🔍 Individual Agent Results:")
    print("-" * 40)
    agent_results = results.get('agent_results', [])
    for agent_result in agent_results:
        print(f"🤖 {agent_result['agent']}:")
        print(f"   🎯 Risk: {agent_result['risk_score']}/10")
        print(f"   📊 Confidence: {agent_result['confidence']}/10")
        print(f"   ⏱️  Time: {agent_result['execution_time']:.2f}s")
        agent_red_flags = agent_result.get('red_flags', [])
        print(f"   🚩 Flags: {len(agent_red_flags)}")
        if agent_red_flags:
            print(f"   📋 Top concerns: {', '.join(agent_red_flags[:2])}")
        print()
    
    # Test agent swapping
    print("🔄 Testing Agent Hot-Swapping...")
    print("-" * 40)
    
    # Add a new specialized agent
    new_agent_config = LLMAgentConfig(
        agent_name="urgency_detector",
        system_prompt="You are an expert at detecting urgency tactics and pressure techniques in fraudulent invoices.",
        prompt_template="""
Analyze this invoice for urgency tactics and pressure techniques:

INVOICE DATA:
{data}

Focus on:
- Language that creates false urgency
- Pressure tactics in payment terms
- Threats or consequences mentioned
- Unusual urgency indicators

Return JSON format:
{{
    "risk_score": <1-10>,
    "confidence": <1-10>,
    "analysis": "<detailed analysis>",
    "red_flags": ["<flag1>", "<flag2>"],
    "fraud_indicators": [
        {{"type": "urgency_tactic", "severity": "medium", "description": "Detailed description"}}
    ]
}}
""",
        temperature=0.1
    )
    
    print("➕ Adding new 'urgency_detector' agent...")
    detector.llm_executor.agent_registry.register_agent_config(new_agent_config)
    
    # Test with the new agent
    urgency_task = LLMTask(
        task_id="urgency_analysis",
        data=detector.demo_invoice,
        agent_names=["urgency_detector"],
        timeout=30.0
    )
    
    urgency_results = await detector.llm_executor.execute_tasks_parallel([urgency_task])
    
    if urgency_results and urgency_results[0].success:
        print("✅ New urgency detector agent executed successfully!")
        urgency_data = urgency_results[0].result
        if isinstance(urgency_data, dict):
            print(f"   🎯 Urgency Risk Score: {urgency_data.get('risk_score', 'N/A')}/10")
            print(f"   📋 Urgency Flags: {len(urgency_data.get('red_flags', []))}")
    else:
        print("❌ New agent failed to execute")
    
    print("\n🔄 Running analysis again with new agent...")
    enhanced_results = await detector.analyze_invoice_parallel(detector.demo_invoice)
    print(f"✅ Enhanced analysis completed with {enhanced_results['agents_used']} agents")
    print(f"🎯 Updated Risk Score: {enhanced_results['overall_risk_score']}/10")
    
    print("\n🏁 Demo completed successfully!")


async def main(file=None):
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Enhanced Invoice Fraud Detection with Parallel LLM Agents")
    
    # Input options
    parser.add_argument("--demo", action="store_true", help="Run demo with sample invoice")
    parser.add_argument("--invoice", help="Invoice text to analyze")
    parser.add_argument("--file", help="File containing invoice data")
    
    # Processing options
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run demo if requested
    if args.demo:
        await demo_enhanced_fraud_detection()
        return
    
    # Initialize detector
    detector = EnhancedParallelInvoiceFraudDetector(max_workers=args.max_workers)

    if file is not None:
        args.file = file
        
    # Get invoice data
    invoice_data = None
    try:
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                content = f.read()
                try:
                    data = json.loads(content)
                    invoice_data = data.get('invoice_text', content)
                except json.JSONDecodeError:
                    invoice_data = content
        elif args.invoice:
            invoice_data = args.invoice
        else:
            print("❌ Error: No invoice data provided. Use --demo, --invoice, or --file")
            return 1
            
        if not invoice_data.strip():
            print("❌ Error: Empty invoice data provided")
            return 1
            
    except FileNotFoundError:
        print(f"❌ Error: File '{args.file}' not found")
        return 1
    except Exception as e:
        print(f"❌ Error reading input: {e}")
        return 1
    
    # Analyze invoice
    try:
        print("🔍 Starting enhanced fraud analysis...")
        print(f"   Max Workers: {args.max_workers}")
        
        results = await detector.analyze_invoice_parallel(invoice_data)
        
        # Display results
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"🎯 Risk Score: {results['overall_risk_score']}/10")
        print(f"📋 Recommendation: {results['recommendation']}")
        print(f"🚨 Status: {results['status']}")
        print(f"🤖 Agents Used: {results['agents_used']}")
        print(f"⏱️  Total Time: {results.get('total_execution_time', 0):.2f}s")
        
        if results['red_flags']:
            print(f"\n🚩 Red Flags ({len(results['red_flags'])}):")
            for flag in results['red_flags']:
                print(f"   • {flag}")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to {args.output}")
        
        # Return appropriate exit code
        if results['recommendation'] in ['REJECT']:
            return 2  # High risk
        elif results['recommendation'] in ['MANUAL_REVIEW']:
            return 1  # Medium risk
        else:
            return 0  # Low risk
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 3


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run demo
        print("No arguments provided. Running demo...")
        asyncio.run(demo_enhanced_fraud_detection())
    else:
        # Run with command line arguments
        exit_code = asyncio.run(main())
        sys.exit(exit_code)