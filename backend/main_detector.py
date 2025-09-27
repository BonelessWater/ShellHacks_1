import os
import sys
import json
import logging
import argparse
import time
import asyncio
import concurrent.futures
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from hardcoded_tools import HardcodedTools, ToolResult, ToolType
    from agent_definitions import FRAUD_DETECTION_AGENTS
    from error_validation import ErrorValidator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory:")
    print("- hardcoded_tools.py")
    print("- agent_definitions.py") 
    print("- error_validation.py")
    sys.exit(1)

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("DSPy not available, using direct API calls")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Google GenerativeAI not available")

# Configure DSPy with your preferred LM
# dspy.configure(lm=dspy.OpenAI(model="gpt-3.5-turbo"))

class ToolType(Enum):
    HARDCODED = "hardcoded"
    LLM = "llm"

@dataclass
class ToolResult:
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0

class HardcodedTools:
    """Non-LLM tools for fast calculations and operations"""
    
    @staticmethod
    def calculator(expression: str) -> ToolResult:
        """Safe calculator for mathematical expressions"""
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
    
    @staticmethod
    def statistics_calc(numbers: List[float], operation: str) -> ToolResult:
        """Statistical calculations on list of numbers"""
        start_time = time.time()
        try:
            if not numbers:
                raise ValueError("Empty list provided")
            
            operations = {
                'mean': statistics.mean,
                'median': statistics.median,
                'mode': statistics.mode,
                'stdev': statistics.stdev,
                'variance': statistics.variance,
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
    def string_operations(text: str, operation: str, **kwargs) -> ToolResult:
        """Fast string operations"""
        start_time = time.time()
        try:
            operations = {
                'length': lambda t: len(t),
                'upper': lambda t: t.upper(),
                'lower': lambda t: t.lower(),
                'reverse': lambda t: t[::-1],
                'word_count': lambda t: len(t.split()),
                'char_count': lambda t: len(t.replace(' ', '')),
                'replace': lambda t: t.replace(kwargs.get('old', ''), kwargs.get('new', '')),
                'split': lambda t: t.split(kwargs.get('delimiter', ' ')),
                'join': lambda t: kwargs.get('delimiter', ' ').join(t.split())
            }
            
            if operation not in operations:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = operations[operation](text)
            execution_time = time.time() - start_time
            return ToolResult(True, result, execution_time=execution_time)
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(False, None, str(e), execution_time)
    
    @staticmethod
    def list_operations(data: List[Any], operation: str, **kwargs) -> ToolResult:
        """Fast list operations"""
        start_time = time.time()
        try:
            operations = {
                'length': lambda d: len(d),
                'reverse': lambda d: list(reversed(d)),
                'sort': lambda d: sorted(d, reverse=kwargs.get('reverse', False)),
                'unique': lambda d: list(set(d)),
                'filter_type': lambda d: [x for x in d if isinstance(x, kwargs.get('type', str))],
                'sum': lambda d: sum(x for x in d if isinstance(x, (int, float))),
                'slice': lambda d: d[kwargs.get('start', 0):kwargs.get('end', len(d))]
            }
            
            if operation not in operations:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = operations[operation](data)
            execution_time = time.time() - start_time
            return ToolResult(True, result, execution_time=execution_time)
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(False, None, str(e), execution_time)

class ToolSelector(dspy.Signature):
    """Signature for selecting appropriate tools for a task"""
    query: str = dspy.InputField(desc="The user's query or task")
    available_tools: str = dspy.InputField(desc="List of available tools and their descriptions")
    selected_tools: str = dspy.OutputField(desc="JSON list of tools to use with their parameters")

class TaskDecomposer(dspy.Signature):
    """Signature for decomposing complex tasks into subtasks"""
    task: str = dspy.InputField(desc="Complex task to decompose")
    subtasks: str = dspy.OutputField(desc="JSON list of independent subtasks that can be executed in parallel")

class ResultSynthesizer(dspy.Signature):
    """Signature for combining results from parallel execution"""
    original_query: str = dspy.InputField(desc="Original user query")
    results: str = dspy.InputField(desc="JSON results from parallel execution")
    final_answer: str = dspy.OutputField(desc="Synthesized final answer combining all results")

class ParallelAgent:
    """Main agent class with parallel processing capabilities"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.hardcoded_tools = HardcodedTools()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize DSPy modules
        self.tool_selector = dspy.ChainOfThought(ToolSelector)
        self.task_decomposer = dspy.ChainOfThought(TaskDecomposer)
        self.result_synthesizer = dspy.ChainOfThought(ResultSynthesizer)
        
        # Tool registry
        self.tools = {
            'calculator': {
                'type': ToolType.HARDCODED,
                'function': self.hardcoded_tools.calculator,
                'description': 'Evaluate mathematical expressions safely',
                'params': ['expression']
            },
            'statistics': {
                'type': ToolType.HARDCODED,
                'function': self.hardcoded_tools.statistics_calc,
                'description': 'Calculate statistics on list of numbers',
                'params': ['numbers', 'operation']
            },
            'string_ops': {
                'type': ToolType.HARDCODED,
                'function': self.hardcoded_tools.string_operations,
                'description': 'Perform string operations',
                'params': ['text', 'operation', 'kwargs']
            },
            'list_ops': {
                'type': ToolType.HARDCODED,
                'function': self.hardcoded_tools.list_operations,
                'description': 'Perform list operations',
                'params': ['data', 'operation', 'kwargs']
            }
        }
    
    def get_available_tools_description(self) -> str:
        """Get formatted description of available tools"""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"{name}: {tool['description']} (params: {tool['params']})")
        return "\n".join(descriptions)
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a single tool with given parameters"""
        if tool_name not in self.tools:
            return ToolResult(False, None, f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        try:
            if tool['type'] == ToolType.HARDCODED:
                return tool['function'](**params)
            else:
                # For LLM tools, implement async execution here
                pass
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    async def execute_tools_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute multiple tools in parallel"""
        loop = asyncio.get_event_loop()
        
        # Create futures for each tool call
        futures = []
        for call in tool_calls:
            future = loop.run_in_executor(
                self.executor, 
                self.execute_tool, 
                call['tool'], 
                call['params']
            )
            futures.append(future)
        
        # Wait for all to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolResult(False, None, str(result)))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def decompose_task(self, task: str) -> List[str]:
        """Decompose a complex task into subtasks"""
        try:
            response = self.task_decomposer(task=task)
            subtasks = json.loads(response.subtasks)
            return subtasks if isinstance(subtasks, list) else [task]
        except:
            # If decomposition fails, return original task
            return [task]
    
    def select_tools(self, query: str) -> List[Dict[str, Any]]:
        """Select appropriate tools for a query"""
        try:
            available_tools = self.get_available_tools_description()
            response = self.tool_selector(query=query, available_tools=available_tools)
            tool_calls = json.loads(response.selected_tools)
            return tool_calls if isinstance(tool_calls, list) else []
        except:
            return []
    
    async def process_query(self, query: str) -> str:
        """Main method to process a query with parallel execution"""
        start_time = time.time()
        
        # Step 1: Decompose the task if complex
        subtasks = self.decompose_task(query)
        print(f"Decomposed into {len(subtasks)} subtasks")
        
        # Step 2: For each subtask, select tools
        all_tool_calls = []
        for subtask in subtasks:
            tool_calls = self.select_tools(subtask)
            all_tool_calls.extend(tool_calls)
        
        if not all_tool_calls:
            return "No applicable tools found for this query."
        
        print(f"Selected {len(all_tool_calls)} tool calls for parallel execution")
        
        # Step 3: Execute all tools in parallel
        results = await self.execute_tools_parallel(all_tool_calls)
        
        # Step 4: Synthesize results
        results_json = json.dumps([
            {
                'tool': all_tool_calls[i]['tool'],
                'success': result.success,
                'result': result.result,
                'error': result.error,
                'execution_time': result.execution_time
            }
            for i, result in enumerate(results)
        ])
        
        try:
            final_response = self.result_synthesizer(
                original_query=query,
                results=results_json
            )
            synthesis_result = final_response.final_answer
        except:
            # Fallback synthesis
            successful_results = [r for r in results if r.success]
            synthesis_result = f"Executed {len(successful_results)} tools successfully. Results: {[r.result for r in successful_results]}"
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f}s")
        
        return synthesis_result
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Example usage and testing
async def main():
    """Example usage of the parallel agent"""
    
    # Initialize agent
    agent = ParallelAgent(max_workers=4)
    
    # Example queries
    queries = [
        "Calculate the mean, median, and standard deviation of the numbers [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and also compute 5 + 3 * 2",
        "Find the length of the string 'Hello World', convert it to uppercase, and reverse it",
        "Sort the list [3, 1, 4, 1, 5, 9, 2, 6] in descending order and find its length"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*50}")
        print(f"Query {i}: {query}")
        print('='*50)
        
        result = await agent.process_query(query)
        print(f"Result: {result}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())

# Additional utility functions for advanced parallel processing
class ParallelTaskManager:
    """Advanced task manager for complex parallel workflows"""
    
    def __init__(self, agent: ParallelAgent):
        self.agent = agent
        self.task_queue = asyncio.Queue()
        self.results = {}
    
    async def add_task(self, task_id: str, query: str, priority: int = 0):
        """Add a task to the queue with priority"""
        await self.task_queue.put((priority, task_id, query))
    
    async def process_queue(self, max_concurrent: int = 3):
        """Process all tasks in the queue with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        
        while not self.task_queue.empty():
            priority, task_id, query = await self.task_queue.get()
            
            async def process_task(tid, q):
                async with semaphore:
                    result = await self.agent.process_query(q)
                    self.results[tid] = result
                    return tid, result
            
            task = asyncio.create_task(process_task(task_id, query))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        return self.results

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            print(f"{func.__name__} executed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise
    return wrapper