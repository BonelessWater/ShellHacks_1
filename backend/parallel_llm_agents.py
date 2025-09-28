# File: backend/parallel_llm_agents.py
#!/usr/bin/env python3
"""
Enhanced Parallel LLM Agent System with Synchronization and Swappable Agents

Features:
- True parallelization of LLM agent calls with proper synchronization
- Agent swapping with registry pattern for different LLM prompts/tasks
- Barrier synchronization to ensure all agents finish before aggregating results
- Comprehensive error handling and recovery for LLM calls
- Performance monitoring and logging
- Plugin-like LLM agent architecture with prompt templates
"""

import asyncio
import concurrent.futures
import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import json
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """LLM Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

 
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "status": self.status.value,
            "metadata": self.metadata,
            "prompt_used": self.prompt_used,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used
        }


@dataclass
class LLMTask:
    """Task to be executed by LLM agents"""
    task_id: str
    data: Any
    agent_names: List[str]  # Which LLM agents can handle this task
    priority: int = 0
    timeout: float = 30.0
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMAgentConfig:
    """Configuration for an LLM agent"""
    
    def __init__(self, 
                 agent_name: str,
                 prompt_template: str,
                 system_prompt: str = "",
                 model_name: str = "gemini-2.5-flash",
                 temperature: float = 0.1,
                 max_tokens: int = 2048,
                 response_format: str = "json"):
        self.agent_name = agent_name
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.last_updated = time.time()
    
    def format_prompt(self, data: Any, context: Dict[str, Any] = None) -> str:
        """Format the prompt template with data and context"""
        format_vars = {
            'data': data,
            'context': context or {},
        }
        
        # If data is a dict, add its keys as top-level variables
        if isinstance(data, dict):
            format_vars.update(data)
            
        # If context provided, add its keys as top-level variables
        if context:
            format_vars.update(context)
            
        try:
            return self.prompt_template.format(**format_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable {e} in prompt for {self.agent_name}")
            return self.prompt_template


class LLMAgent:
    """LLM Agent that executes specific tasks using language models"""
    
    def __init__(self, 
                 agent_id: str, 
                 config: LLMAgentConfig,
                 llm_client: Any = None):
        self.agent_id = agent_id
        self.config = config
        self.llm_client = llm_client
        self.status = AgentStatus.IDLE
        self._lock = threading.Lock()
        self.execution_count = 0
        self.total_execution_time = 0.0
        
    async def execute(self, task: LLMTask) -> LLMAgentResult:
        """Execute a task using the LLM"""
        start_time = time.time()
        
        with self._lock:
            if self.status != AgentStatus.IDLE:
                return LLMAgentResult(
                    agent_id=self.agent_id,
                    agent_name=self.config.agent_name,
                    success=False,
                    error=f"Agent busy with status: {self.status.value}",
                    execution_time=0.0,
                    status=self.status
                )
            self.status = AgentStatus.RUNNING
        
        try:
            logger.info(f"LLM Agent {self.agent_id} ({self.config.agent_name}) starting task {task.task_id}")
            
            # Format the prompt
            formatted_prompt = self.config.format_prompt(task.data, task.context)
            
            # Execute LLM call with timeout
            result = await asyncio.wait_for(
                self._call_llm(formatted_prompt, task),
                timeout=task.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Update statistics
            with self._lock:
                self.execution_count += 1
                self.total_execution_time += execution_time
                self.status = AgentStatus.COMPLETED
            
            result.execution_time = execution_time
            result.status = AgentStatus.COMPLETED
            
            logger.info(f"LLM Agent {self.agent_id} completed task {task.task_id} in {execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"LLM call timed out after {task.timeout}s"
            logger.error(f"LLM Agent {self.agent_id}: {error_msg}")
            
            with self._lock:
                self.status = AgentStatus.TIMEOUT
                
            return LLMAgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.agent_name,
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time,
                status=AgentStatus.TIMEOUT
            )
            
        except Exception as e:
            error_msg = f"LLM call failed: {str(e)}"
            logger.error(f"LLM Agent {self.agent_id}: {error_msg}")
            logger.debug(traceback.format_exc())
            
            with self._lock:
                self.status = AgentStatus.FAILED
                
            return LLMAgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.agent_name,
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time,
                status=AgentStatus.FAILED
            )
        
        finally:
            # Reset status to idle after a short delay
            asyncio.create_task(self._reset_status_after_delay())
    
    async def _call_llm(self, prompt: str, task: LLMTask) -> LLMAgentResult:
        """Make the actual LLM API call"""
        try:
            # Prepare the full prompt with system message
            full_prompt = prompt
            if self.config.system_prompt:
                full_prompt = f"System: {self.config.system_prompt}\n\nUser: {prompt}"
            
            # Mock LLM call - replace with your actual LLM client
            if self.llm_client and hasattr(self.llm_client, 'generate_content'):
                # For Google Gemini
                response = self.llm_client.generate_content(
                    full_prompt,
                    generation_config={
                        'temperature': self.config.temperature,
                        'max_output_tokens': self.config.max_tokens,
                    }
                )
                response_text = response.text
                tokens_used = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                
            elif self.llm_client and hasattr(self.llm_client, 'chat'):
                # For OpenAI-style clients
                response = await self.llm_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                response_text = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
                
            else:
                # Fallback mock response for demo
                await asyncio.sleep(0.5)  # Simulate API call delay
                response_text = f"Mock response from {self.config.agent_name} for task {task.task_id}"
                tokens_used = 100
            
            # Parse response if JSON format expected
            parsed_result = response_text
            if self.config.response_format == "json":
                try:
                    # Try to extract JSON from response
                    response_text = response_text.strip()
                    if response_text.startswith('```json'):
                        response_text = response_text[7:-3]
                    elif response_text.startswith('```'):
                        response_text = response_text[3:-3]
                    
                    parsed_result = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON response from {self.agent_id}, using raw text")
                    parsed_result = response_text
            
            return LLMAgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.agent_name,
                success=True,
                result=parsed_result,
                prompt_used=full_prompt,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                metadata={
                    'temperature': self.config.temperature,
                    'max_tokens': self.config.max_tokens,
                    'response_format': self.config.response_format
                }
            )
            
        except Exception as e:
            return LLMAgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.agent_name,
                success=False,
                error=f"LLM API call failed: {str(e)}",
                prompt_used=full_prompt,
                model_used=self.config.model_name
            )
    
    async def _reset_status_after_delay(self, delay: float = 1.0):
        """Reset agent status to idle after delay"""
        await asyncio.sleep(delay)
        with self._lock:
            if self.status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.TIMEOUT]:
                self.status = AgentStatus.IDLE
    
    def can_handle(self, task: LLMTask) -> bool:
        """Check if this agent can handle the given task"""
        return self.config.agent_name in task.agent_names
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        with self._lock:
            avg_time = self.total_execution_time / self.execution_count if self.execution_count > 0 else 0
            return {
                'agent_id': self.agent_id,
                'agent_name': self.config.agent_name,
                'status': self.status.value,
                'execution_count': self.execution_count,
                'total_execution_time': self.total_execution_time,
                'average_execution_time': avg_time,
                'model_name': self.config.model_name
            }


class LLMAgentRegistry:
    """Registry for managing LLM agent configurations and instances"""
    
    def __init__(self):
        self._agent_configs: Dict[str, LLMAgentConfig] = {}
        self._agent_instances: Dict[str, LLMAgent] = {}
        self._lock = threading.Lock()
    
    def register_agent_config(self, config: LLMAgentConfig):
        """Register an LLM agent configuration"""
        with self._lock:
            self._agent_configs[config.agent_name] = config
            logger.info(f"Registered LLM agent config: {config.agent_name}")
    
    def unregister_agent_config(self, agent_name: str):
        """Unregister an LLM agent configuration"""
        with self._lock:
            if agent_name in self._agent_configs:
                del self._agent_configs[agent_name]
                # Also remove any instances of this agent type
                instances_to_remove = [
                    aid for aid, agent in self._agent_instances.items()
                    if agent.config.agent_name == agent_name
                ]
                for aid in instances_to_remove:
                    del self._agent_instances[aid]
                logger.info(f"Unregistered LLM agent config: {agent_name}")
    
    def update_agent_config(self, config: LLMAgentConfig):
        """Update an existing agent configuration (hot-swap)"""
        with self._lock:
            if config.agent_name in self._agent_configs:
                self._agent_configs[config.agent_name] = config
                # Update existing instances
                for agent in self._agent_instances.values():
                    if agent.config.agent_name == config.agent_name:
                        agent.config = config
                logger.info(f"Updated LLM agent config: {config.agent_name}")
            else:
                self.register_agent_config(config)
    
    def create_agent_instance(self, 
                            agent_name: str, 
                            agent_id: str, 
                            llm_client: Any = None) -> LLMAgent:
        """Create an LLM agent instance"""
        with self._lock:
            if agent_name not in self._agent_configs:
                raise ValueError(f"Unknown agent type: {agent_name}")
            
            config = self._agent_configs[agent_name]
            agent = LLMAgent(agent_id, config, llm_client)
            self._agent_instances[agent_id] = agent
            
            logger.info(f"Created LLM agent instance {agent_id} of type {agent_name}")
            return agent
    
    def get_agent_instance(self, agent_id: str) -> Optional[LLMAgent]:
        """Get an agent instance by ID"""
        return self._agent_instances.get(agent_id)
    
    def remove_agent_instance(self, agent_id: str):
        """Remove an agent instance"""
        with self._lock:
            if agent_id in self._agent_instances:
                del self._agent_instances[agent_id]
                logger.info(f"Removed LLM agent instance {agent_id}")
    
    def get_agents_by_name(self, agent_name: str) -> List[LLMAgent]:
        """Get all agent instances of a specific type"""
        return [
            agent for agent in self._agent_instances.values()
            if agent.config.agent_name == agent_name
        ]
    
    def get_available_agent_types(self) -> List[str]:
        """Get list of available agent types"""
        return list(self._agent_configs.keys())
    
    def get_all_agent_instances(self) -> List[LLMAgent]:
        """Get all agent instances"""
        return list(self._agent_instances.values())


class ParallelLLMExecutor:
    """Main executor for parallel LLM agent processing with synchronization"""
    
    def __init__(self, 
                 max_workers: int = 4, 
                 agent_registry: Optional[LLMAgentRegistry] = None,
                 llm_client: Any = None):
        self.max_workers = max_workers
        self.agent_registry = agent_registry or LLMAgentRegistry()
        self.llm_client = llm_client
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._shutdown = False
        
        # Synchronization primitives
        self._task_barrier = None
        self._results_lock = threading.Lock()
        
    async def execute_tasks_parallel(self, 
                                   tasks: List[LLMTask], 
                                   wait_for_all: bool = True,
                                   aggregate_results: bool = True) -> List[LLMAgentResult]:
        """
        Execute multiple LLM tasks in parallel with proper synchronization
        
        Args:
            tasks: List of LLM tasks to execute
            wait_for_all: If True, wait for all tasks to complete before returning
            aggregate_results: If True, aggregate results after all tasks complete
        """
        if not tasks:
            return []
        
        logger.info(f"Starting parallel execution of {len(tasks)} LLM tasks")
        start_time = time.time()
        
        # Assign agents to tasks
        task_agent_pairs = self._assign_agents_to_tasks(tasks)
        
        if not task_agent_pairs:
            logger.error("No LLM agents available to handle tasks")
            return []
        
        # Create barrier for synchronization if needed
        if wait_for_all:
            self._task_barrier = asyncio.Barrier(len(task_agent_pairs))
        
        # Execute tasks
        async_tasks = []
        for task, agent in task_agent_pairs:
            async_task = asyncio.create_task(
                self._execute_task_with_barrier(task, agent, wait_for_all)
            )
            async_tasks.append(async_task)
        
        # Wait for completion
        if wait_for_all:
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
        else:
            # Fire and forget - just start the tasks
            return []
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = task_agent_pairs[i][0]
                agent = task_agent_pairs[i][1]
                
                error_result = LLMAgentResult(
                    agent_id=agent.agent_id,
                    agent_name=agent.config.agent_name,
                    success=False,
                    error=f"Task execution failed: {str(result)}",
                    execution_time=0.0,
                    status=AgentStatus.FAILED
                )
                processed_results.append(error_result)
                logger.error(f"LLM Task {task.task_id} failed with exception: {result}")
            else:
                processed_results.append(result)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in processed_results if r.success)
        
        logger.info(f"Parallel LLM execution completed in {total_time:.2f}s")
        logger.info(f"Success rate: {successful}/{len(processed_results)} tasks")
        
        # Aggregate results if requested
        if aggregate_results and processed_results:
            aggregated = await self._aggregate_results(processed_results, tasks)
            logger.info("Results aggregated successfully")
            return aggregated
        
        return processed_results
    
    async def _execute_task_with_barrier(self, 
                                       task: LLMTask, 
                                       agent: LLMAgent, 
                                       use_barrier: bool) -> LLMAgentResult:
        """Execute LLM task and wait at barrier if needed"""
        try:
            # Execute the LLM task
            result = await agent.execute(task)
            
            # Wait at barrier if synchronization is required
            if use_barrier and self._task_barrier:
                logger.debug(f"LLM Agent {agent.agent_id} waiting at barrier")
                await self._task_barrier.wait()
                logger.debug(f"LLM Agent {agent.agent_id} passed barrier")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM task execution with barrier: {e}")
            return LLMAgentResult(
                agent_id=agent.agent_id,
                agent_name=agent.config.agent_name,
                success=False,
                error=str(e),
                status=AgentStatus.FAILED
            )
    
    def _assign_agents_to_tasks(self, tasks: List[LLMTask]) -> List[tuple[LLMTask, LLMAgent]]:
        """Assign available LLM agents to tasks"""
        assignments = []
        
        for task in tasks:
            # Find available agent that can handle this task
            suitable_agents = []
            
            for agent_name in task.agent_names:
                agents = self.agent_registry.get_agents_by_name(agent_name)
                for agent in agents:
                    if (agent.status == AgentStatus.IDLE and 
                        agent.can_handle(task)):
                        suitable_agents.append(agent)
            
            if suitable_agents:
                # Use the first available agent
                chosen_agent = suitable_agents[0]
                assignments.append((task, chosen_agent))
                logger.debug(f"Assigned LLM task {task.task_id} to agent {chosen_agent.agent_id}")
            else:
                # Create a new agent if none available
                if task.agent_names:
                    agent_name = task.agent_names[0]
                    try:
                        new_agent_id = f"{agent_name}_{len(assignments)}_{int(time.time())}"
                        new_agent = self.agent_registry.create_agent_instance(
                            agent_name, new_agent_id, self.llm_client
                        )
                        assignments.append((task, new_agent))
                        logger.info(f"Created new LLM agent {new_agent_id} for task {task.task_id}")
                    except ValueError as e:
                        logger.error(f"Could not create LLM agent for task {task.task_id}: {e}")
                else:
                    logger.error(f"No agent types specified for LLM task {task.task_id}")
        
        return assignments
    
    async def _aggregate_results(self, 
                               results: List[LLMAgentResult], 
                               original_tasks: List[LLMTask]) -> List[LLMAgentResult]:
        """Aggregate results from multiple LLM agents"""
        # This is a basic aggregation - customize based on your needs
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Log aggregation summary
        logger.info(f"Aggregating {len(successful_results)} successful and {len(failed_results)} failed results")
        
        # You can implement custom aggregation logic here
        # For example, combining results, voting, consensus, etc.
        
        return results  # Return original results for now
    
    def add_agent_type(self, config: LLMAgentConfig):
        """Add a new LLM agent type (hot-swappable)"""
        self.agent_registry.register_agent_config(config)
    
    def update_agent_type(self, config: LLMAgentConfig):
        """Update an existing LLM agent type (hot-swappable)"""
        self.agent_registry.update_agent_config(config)
    
    def remove_agent_type(self, agent_name: str):
        """Remove an LLM agent type"""
        self.agent_registry.unregister_agent_config(agent_name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        agents = self.agent_registry.get_all_agent_instances()
        
        status_counts = {}
        for status in AgentStatus:
            status_counts[status.value] = sum(
                1 for agent in agents if agent.status == status
            )
        
        # Get agent statistics
        agent_stats = [agent.get_stats() for agent in agents]
        
        return {
            "total_agents": len(agents),
            "max_workers": self.max_workers,
            "agent_status_counts": status_counts,
            "available_agent_types": self.agent_registry.get_available_agent_types(),
            "agent_statistics": agent_stats,
            "shutdown": self._shutdown
        }
    
    async def shutdown(self):
        """Gracefully shutdown the executor"""
        self._shutdown = True
        logger.info("Shutting down parallel LLM agent executor")
        
        # Wait for running tasks to complete (with timeout)
        agents = self.agent_registry.get_all_agent_instances()
        running_agents = [a for a in agents if a.status == AgentStatus.RUNNING]
        
        if running_agents:
            logger.info(f"Waiting for {len(running_agents)} running LLM agents to complete")
            timeout = 60.0  # Longer timeout for LLM calls
            start_time = time.time()
            
            while running_agents and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.5)
                running_agents = [a for a in agents if a.status == AgentStatus.RUNNING]
        
        self.executor.shutdown(wait=True)
        logger.info("Parallel LLM agent executor shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Example LLM Agent Configurations for Invoice Fraud Detection

def get_fraud_detection_agent_configs() -> List[LLMAgentConfig]:
    """Get predefined LLM agent configurations for fraud detection"""
    
    configs = []
    
    # Amount Validation Agent
    configs.append(LLMAgentConfig(
        agent_name="amount_validator",
        system_prompt="You are an expert at detecting fraudulent invoice amounts and mathematical inconsistencies.",
        prompt_template="""
Analyze the following invoice data for amount-related fraud indicators:

INVOICE DATA:
{data}

CONTEXT:
{context}

Please analyze and provide:
1. Risk score (1-10, where 10 is highest fraud risk)
2. Confidence level (1-10, where 10 is highest confidence)
3. Detailed analysis of amount patterns
4. List of any red flags found
5. Specific fraud indicators with severity levels

Return your response in JSON format:
{{
    "risk_score": <1-10>,
    "confidence": <1-10>,
    "analysis": "<detailed analysis>",
    "red_flags": ["<flag1>", "<flag2>"],
    "fraud_indicators": [
        {{"type": "<indicator_type>", "severity": "<low/medium/high>", "description": "<description>"}}
    ]
}}
""",
        temperature=0.1,
        response_format="json"
    ))
    
    # Vendor Validation Agent
    configs.append(LLMAgentConfig(
        agent_name="vendor_validator", 
        system_prompt="You are an expert at detecting fraudulent vendors and supplier relationships.",
        prompt_template="""
Analyze the following invoice data for vendor-related fraud indicators:

INVOICE DATA:
{data}

CONTEXT:
{context}

Focus on vendor legitimacy, historical patterns, and suspicious vendor characteristics.

Return your response in JSON format:
{{
    "risk_score": <1-10>,
    "confidence": <1-10>,
    "analysis": "<detailed analysis>",
    "red_flags": ["<flag1>", "<flag2>"],
    "fraud_indicators": [
        {{"type": "<indicator_type>", "severity": "<low/medium/high>", "description": "<description>"}}
    ]
}}
""",
        temperature=0.1,
        response_format="json"
    ))
    
    # Date/Timing Analysis Agent
    configs.append(LLMAgentConfig(
        agent_name="date_analyzer",
        system_prompt="You are an expert at detecting fraudulent date patterns and timing anomalies in invoices.",
        prompt_template="""
Analyze the following invoice data for date and timing-related fraud indicators:

INVOICE DATA:
{data}

CONTEXT:
{context}

Look for suspicious date patterns, weekend/holiday submissions, backdating, etc.

Return your response in JSON format:
{{
    "risk_score": <1-10>,
    "confidence": <1-10>,
    "analysis": "<detailed analysis>",
    "red_flags": ["<flag1>", "<flag2>"],
    "fraud_indicators": [
        {{"type": "<indicator_type>", "severity": "<low/medium/high>", "description": "<description>"}}
    ]
}}
""",
        temperature=0.1,
        response_format="json"
    ))
    
    # Pattern Recognition Agent
    configs.append(LLMAgentConfig(
        agent_name="pattern_detector",
        system_prompt="You are an expert at detecting complex fraud patterns and anomalies across invoice data.",
        prompt_template="""
Analyze the following invoice data for complex fraud patterns and anomalies:

INVOICE DATA:
{data}

CONTEXT:
{context}

Look for subtle patterns, correlations, and anomalies that might indicate fraud.

Return your response in JSON format:
{{
    "risk_score": <1-10>,
    "confidence": <1-10>,
    "analysis": "<detailed analysis>",
    "red_flags": ["<flag1>", "<flag2>"],
    "fraud_indicators": [
        {{"type": "<indicator_type>", "severity": "<low/medium/high>", "description": "<description>"}}
    ]
}}
""",
        temperature=0.2,
        response_format="json"
    ))
    
    return configs


# Demo and Testing Functions

async def demo_parallel_llm_execution():
    """Demonstrate the parallel LLM agent system"""
    print("🚀 Parallel LLM Agent System Demo")
    print("=" * 50)
    
    # Create executor and registry
    executor = ParallelLLMExecutor(max_workers=4)
    
    # Register LLM agent types
    for config in get_fraud_detection_agent_configs():
        executor.add_agent_type(config)
    
    # Create sample invoice data
    sample_invoice_data = {
        "invoice_id": "INV-12345",
        "vendor_name": "Acme Corp",
        "total_amount": 15000.00,
        "line_items": [
            {"description": "Software licenses", "amount": 8000.00, "quantity": 1},
            {"description": "Consulting services", "amount": 7000.00, "quantity": 40}
        ],
        "invoice_date": "2025-09-15",
        "due_date": "2025-10-15",
        "vendor_address": "123 Business St, City, State",
        "raw_text": "Invoice #INV-12345 from Acme Corp for $15,000 total..."
    }
    
    # Create sample tasks
    tasks = [
        LLMTask(
            task_id="amount_validation",
            data=sample_invoice_data,
            agent_names=["amount_validator"],
            timeout=30.0,
            context={"analysis_type": "amount_validation"}
        ),
        LLMTask(
            task_id="vendor_validation", 
            data=sample_invoice_data,
            agent_names=["vendor_validator"],
            timeout=30.0,
            context={"analysis_type": "vendor_validation"}
        ),
        LLMTask(
            task_id="date_analysis",
            data=sample_invoice_data,
            agent_names=["date_analyzer"],
            timeout=30.0,
            context={"analysis_type": "date_analysis"}
        ),
        LLMTask(
            task_id="pattern_detection",
            data=sample_invoice_data,
            agent_names=["pattern_detector"],
            timeout=30.0,
            context={"analysis_type": "pattern_detection"}
        )
    ]
    
    print(f"📋 Created {len(tasks)} LLM tasks for fraud detection")
    
    # Execute tasks in parallel with synchronization
    print("\n🔄 Executing LLM agents in parallel...")
    print("⏳ Waiting for all agents to complete before aggregating results...")
    start_time = time.time()
    
    results = await executor.execute_tasks_parallel(
        tasks, 
        wait_for_all=True,  # Ensure all agents finish
        aggregate_results=True
    )
    
    execution_time = time.time() - start_time
    
    # Display results
    print(f"\n✅ All LLM agents completed in {execution_time:.2f} seconds")
    print("\n📊 Individual Agent Results:")
    
    total_risk_score = 0
    total_confidence = 0
    all_red_flags = []
    
    for result in results:
        status_icon = "✅" if result.success else "❌"
        print(f"\n{status_icon} {result.agent_name} ({result.agent_id}):")
        print(f"   ⏱️  Execution time: {result.execution_time:.2f}s")
        print(f"   🔧 Model used: {result.model_used}")
        print(f"   🎯 Tokens used: {result.tokens_used}")
        
        if result.success and isinstance(result.result, dict):
            risk_score = result.result.get('risk_score', 0)
            confidence = result.result.get('confidence', 0)
            analysis = result.result.get('analysis', 'No analysis')
            red_flags = result.result.get('red_flags', [])
            
            total_risk_score += risk_score
            total_confidence += confidence
            all_red_flags.extend(red_flags)
            
            print(f"   🎯 Risk Score: {risk_score}/10")
            print(f"   🔍 Confidence: {confidence}/10")
            print(f"   📝 Analysis: {analysis[:100]}...")
            print(f"   🚩 Red Flags: {len(red_flags)} found")
            
        elif result.error:
            print(f"   ❌ Error: {result.error}")
    
    # Aggregate final results
    if results:
        avg_risk_score = total_risk_score / len([r for r in results if r.success])
        avg_confidence = total_confidence / len([r for r in results if r.success])
        unique_red_flags = list(set(all_red_flags))
        
        print(f"\n🎯 AGGREGATED FRAUD ANALYSIS:")
        print(f"   📊 Average Risk Score: {avg_risk_score:.1f}/10")
        print(f"   📊 Average Confidence: {avg_confidence:.1f}/10")
        print(f"   🚩 Total Unique Red Flags: {len(unique_red_flags)}")
        print(f"   ⚡ Total Processing Time: {execution_time:.2f}s")
        
        if avg_risk_score >= 7:
            print("   🚨 HIGH FRAUD RISK DETECTED")
        elif avg_risk_score >= 4:
            print("   ⚠️  MEDIUM FRAUD RISK")
        else:
            print("   ✅ LOW FRAUD RISK")
    
    # Show system status
    print(f"\n🖥️  System Status:")
    status = executor.get_system_status()
    print(f"   📊 Total Agents: {status['total_agents']}")
    print(f"   📊 Max Workers: {status['max_workers']}")
    print(f"   📊 Available Agent Types: {len(status['available_agent_types'])}")
    
    for agent_type in status['available_agent_types']:
        print(f"      - {agent_type}")
    
    # Demonstrate hot-swapping agents
    print(f"\n🔄 Demonstrating agent hot-swapping...")
    
    # Add a new specialized agent
    new_agent_config = LLMAgentConfig(
        agent_name="tax_validator",
        system_prompt="You are an expert at validating tax calculations and detecting tax-related fraud.",
        prompt_template="""
Analyze the following invoice for tax calculation fraud:

INVOICE DATA:
{data}

Return JSON with risk_score, confidence, analysis, and red_flags.
""",
        temperature=0.1,
        response_format="json"
    )
    
    print("   ➕ Adding new 'tax_validator' agent...")
    executor.add_agent_type(new_agent_config)
    
    # Test the new agent
    tax_task = LLMTask(
        task_id="tax_validation",
        data=sample_invoice_data,
        agent_names=["tax_validator"],
        timeout=30.0
    )
    
    tax_results = await executor.execute_tasks_parallel([tax_task])
    tax_success = tax_results[0].success if tax_results else False
    print(f"   ✅ New tax validator agent executed: {tax_success}")
    
    # Update an existing agent
    print("   🔄 Updating 'amount_validator' agent prompt...")
    updated_config = LLMAgentConfig(
        agent_name="amount_validator",
        system_prompt="You are an ENHANCED expert at detecting fraudulent invoice amounts.",
        prompt_template="""
ENHANCED ANALYSIS - Analyze this invoice for amount fraud with extra scrutiny:

{data}

Return detailed JSON analysis with enhanced risk assessment.
""",
        temperature=0.05,  # Lower temperature for more consistent results
        response_format="json"
    )
    
    executor.update_agent_type(updated_config)
    print("   ✅ Amount validator agent updated with enhanced prompts")
    
    # Remove an agent
    print("   ➖ Removing 'pattern_detector' agent...")
    executor.remove_agent_type("pattern_detector")
    
    updated_status = executor.get_system_status()
    print(f"   📊 Agent types after changes: {len(updated_status['available_agent_types'])}")
    
    # Test execution after agent changes
    print("\n🔄 Testing execution after agent hot-swap...")
    
    modified_tasks = [
        LLMTask(
            task_id="enhanced_amount_check",
            data=sample_invoice_data,
            agent_names=["amount_validator"],  # This should use the updated config
            timeout=30.0
        ),
        LLMTask(
            task_id="tax_check",
            data=sample_invoice_data,
            agent_names=["tax_validator"],  # This should use the new agent
            timeout=30.0
        )
    ]
    
    modified_results = await executor.execute_tasks_parallel(modified_tasks)
    successful_modified = sum(1 for r in modified_results if r.success)
    print(f"   ✅ Post-swap execution: {successful_modified}/{len(modified_results)} agents succeeded")
    
    # Cleanup
    await executor.shutdown()
    print("\n🏁 Demo completed!")


async def demo_integration_with_existing_system():
    """Demo integration with existing invoice fraud detection system"""
    print("\n🔗 Integration with Existing System Demo")
    print("=" * 50)
    
    # This shows how to integrate with your existing ParallelInvoiceFraudDetector
    class EnhancedFraudDetector:
        """Enhanced fraud detector using the new parallel LLM system"""
        
        def __init__(self, llm_client=None):
            self.llm_executor = ParallelLLMExecutor(max_workers=6, llm_client=llm_client)
            
            # Register all fraud detection agents
            for config in get_fraud_detection_agent_configs():
                self.llm_executor.add_agent_type(config)
        
        async def analyze_invoice_parallel(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze invoice using parallel LLM agents with synchronization"""
            
            # Create tasks for all available fraud detection agents
            tasks = []
            agent_types = self.llm_executor.agent_registry.get_available_agent_types()
            
            for agent_type in agent_types:
                task = LLMTask(
                    task_id=f"{agent_type}_{invoice_data.get('invoice_id', 'unknown')}",
                    data=invoice_data,
                    agent_names=[agent_type],
                    timeout=45.0,
                    context={"invoice_id": invoice_data.get('invoice_id')}
                )
                tasks.append(task)
            
            # Execute all agents in parallel and wait for completion
            print(f"🔄 Executing {len(tasks)} fraud detection agents in parallel...")
            results = await self.llm_executor.execute_tasks_parallel(
                tasks,
                wait_for_all=True,  # Critical: wait for all agents to finish
                aggregate_results=True
            )
            
            # Aggregate results
            return self._aggregate_fraud_results(results, invoice_data)
        
        def _aggregate_fraud_results(self, 
                                   results: List[LLMAgentResult], 
                                   invoice_data: Dict[str, Any]) -> Dict[str, Any]:
            """Aggregate results from all fraud detection agents"""
            successful_results = [r for r in results if r.success]
            
            if not successful_results:
                return {
                    "invoice_id": invoice_data.get('invoice_id'),
                    "overall_risk_score": 10,  # High risk due to analysis failure
                    "confidence": 1,
                    "status": "ANALYSIS_FAILED",
                    "agent_results": [],
                    "red_flags": ["ANALYSIS_FAILURE"],
                    "recommendation": "MANUAL_REVIEW"
                }
            
            # Calculate aggregate scores
            risk_scores = []
            confidences = []
            all_red_flags = []
            agent_results = []
            
            for result in successful_results:
                if isinstance(result.result, dict):
                    risk_score = result.result.get('risk_score', 5)
                    confidence = result.result.get('confidence', 5)
                    red_flags = result.result.get('red_flags', [])
                    
                    risk_scores.append(risk_score)
                    confidences.append(confidence)
                    all_red_flags.extend(red_flags)
                    
                    agent_results.append({
                        "agent_name": result.agent_name,
                        "risk_score": risk_score,
                        "confidence": confidence,
                        "execution_time": result.execution_time,
                        "red_flags": red_flags
                    })
            
            # Weighted average (higher confidence results have more weight)
            if risk_scores and confidences:
                weighted_risk = sum(r * c for r, c in zip(risk_scores, confidences)) / sum(confidences)
                avg_confidence = sum(confidences) / len(confidences)
            else:
                weighted_risk = 5
                avg_confidence = 1
            
            # Determine recommendation
            if weighted_risk >= 8:
                recommendation = "REJECT"
            elif weighted_risk >= 6:
                recommendation = "MANUAL_REVIEW"
            elif weighted_risk >= 4:
                recommendation = "ADDITIONAL_VERIFICATION"
            else:
                recommendation = "APPROVE"
            
            return {
                "invoice_id": invoice_data.get('invoice_id'),
                "overall_risk_score": round(weighted_risk, 1),
                "confidence": round(avg_confidence, 1),
                "status": "COMPLETED",
                "agent_results": agent_results,
                "red_flags": list(set(all_red_flags)),
                "recommendation": recommendation,
                "total_agents": len(successful_results),
                "analysis_summary": f"Analyzed by {len(successful_results)} specialized agents"
            }
        
        async def shutdown(self):
            """Shutdown the enhanced detector"""
            await self.llm_executor.shutdown()
    
    # Test the enhanced detector
    detector = EnhancedFraudDetector()
    
    test_invoice = {
        "invoice_id": "INV-DEMO-001",
        "vendor_name": "Test Vendor LLC",
        "total_amount": 25000.00,
        "line_items": [
            {"description": "Consulting", "amount": 25000.00, "quantity": 1}
        ],
        "invoice_date": "2025-09-28",
        "vendor_address": "Unknown"
    }
    
    print(f"📄 Testing with invoice: {test_invoice['invoice_id']}")
    
    analysis_result = await detector.analyze_invoice_parallel(test_invoice)
    
    print(f"\n📊 FINAL ANALYSIS RESULT:")
    print(f"   🎯 Overall Risk Score: {analysis_result['overall_risk_score']}/10")
    print(f"   📊 Confidence: {analysis_result['confidence']}/10")
    print(f"   📋 Recommendation: {analysis_result['recommendation']}")
    print(f"   🚩 Red Flags: {len(analysis_result['red_flags'])}")
    print(f"   🤖 Agents Used: {analysis_result['total_agents']}")
    
    await detector.shutdown()
    print("✅ Integration demo completed!")


if __name__ == "__main__":
    async def main():
        await demo_parallel_llm_execution()
        await demo_integration_with_existing_system()
    
    asyncio.run(main())