#!/usr/bin/env python3
"""
Enhanced Multi-Agent Invoice Fraud Detection System with Parallel Processing

This system combines:
- Parallel processing for faster execution
- Hardcoded tools for deterministic fraud detection
- LLM agents for contextual analysis
- Comprehensive error recovery

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
    from hardcoded_tools import HardcodedTools, ToolResult, ToolType, HARDCODED_TOOL_REGISTRY
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure hardcoded_tools.py is in the same directory")
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

# Configure logging with Windows-safe formatting
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

@dataclass
class AgentResponse:
    agent_type: str
    analysis: str
    risk_score: int  # 1-10 scale
    confidence: int  # 1-10 scale
    red_flags: List[str]
    execution_time: float = 0.0
    tool_used: str = "llm"

class ParallelInvoiceFraudDetector:
    """Enhanced fraud detector with parallel processing and hardcoded tools"""
    
    def __init__(self, max_workers: int = 4, enable_parallel: bool = True):
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel
        self.hardcoded_tools = HardcodedTools()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) if enable_parallel else None
        
        # Initialize API
        self.api_key = self._get_api_key()
        if GENAI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        else:
            self.model = None
            log.warning("Google GenerativeAI not available or no API key")
        
        # Available agents - mix of hardcoded and LLM
        self.available_agents = {
            "amount_validator": {
                "description": "Analyzes invoice amounts using hardcoded fraud detection algorithms",
                "type": "hardcoded",
                "tool": "amount_validator"
            },
            "tax_calculator": {
                "description": "Validates tax calculations using mathematical verification",
                "type": "hardcoded", 
                "tool": "tax_calculator"
            },
            "date_analyzer": {
                "description": "Examines dates using pattern recognition algorithms",
                "type": "hardcoded",
                "tool": "date_analyzer"
            },
            "vendor_authenticator": {
                "description": "Validates vendor information using database matching",
                "type": "hardcoded",
                "tool": "vendor_authenticator"
            },
            "format_inspector": {
                "description": "Checks invoice format using text analysis algorithms",
                "type": "hardcoded",
                "tool": "format_inspector"
            },
            "line_item_validator": {
                "description": "Reviews individual line items for reasonableness using LLM analysis",
                "type": "llm"
            },
            "payment_terms_checker": {
                "description": "Analyzes payment terms and conditions using LLM reasoning",
                "type": "llm"
            }
        }
        
        # Demo invoice for testing
        self.demo_invoice = """
        INVOICE #INV-2024-0156
        Date: 2024-03-15
        Due Date: 2024-04-15
        
        From: QuickFix Solutions LLC
        Email: billing@quickfixsolutions.biz
        Address: 123 Business St, City, ST 12345
        
        To: ACME Corporation
        
        Description                    Qty    Unit Price    Total
        Emergency IT Consulting         10        $500.00   $5,000.00
        Weekend Database Maintenance     2      $1,000.00   $2,000.00
        Security Audit Premium           1      $3,000.00   $3,000.00
        
        Subtotal:                                          $10,000.00
        Tax (8.5%):                                           $850.00
        Total:                                             $10,850.00
        
        Payment Terms: Net 30 Days
        Thank you for your business!
        """
    
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
        
        log.warning("No Google API key found - hardcoded tools only")
        return None
    
    def extract_invoice_data(self, invoice_text: str) -> Dict[str, Any]:
        """Extract structured data from invoice text for hardcoded tools"""
        try:
            # Simple extraction - in practice, you might use more sophisticated parsing
            import re
            
            data = {
                'amounts': [],
                'vendor_name': '',
                'vendor_email': '',
                'invoice_date': '',
                'due_date': '',
                'subtotal': 0.0,
                'tax_rate': 0.0,
                'stated_tax': 0.0,
                'total': 0.0
            }
            
            # Extract amounts using regex
            amount_pattern = r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
            amounts = []
            for match in re.finditer(amount_pattern, invoice_text):
                try:
                    amount = float(match.group(1).replace(',', ''))
                    amounts.append(amount)
                except ValueError:
                    continue
            data['amounts'] = amounts
            
            # Extract vendor name (after "From:" or "Vendor:")
            vendor_match = re.search(r'(?:From|Vendor):\s*([^\n\r]+)', invoice_text, re.IGNORECASE)
            if vendor_match:
                data['vendor_name'] = vendor_match.group(1).strip()
            
            # Extract email
            email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', invoice_text)
            if email_match:
                data['vendor_email'] = email_match.group(1)
            
            # Extract dates
            date_pattern = r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})'
            dates = re.findall(date_pattern, invoice_text)
            if dates:
                data['invoice_date'] = dates[0]
                if len(dates) > 1:
                    data['due_date'] = dates[1]
            
            # Extract tax information
            tax_rate_match = re.search(r'tax.*?(\d+\.?\d*)%', invoice_text, re.IGNORECASE)
            if tax_rate_match:
                data['tax_rate'] = float(tax_rate_match.group(1))
            
            # Extract specific amounts
            subtotal_match = re.search(r'subtotal.*?\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', invoice_text, re.IGNORECASE)
            if subtotal_match:
                data['subtotal'] = float(subtotal_match.group(1).replace(',', ''))
            
            tax_match = re.search(r'tax.*?\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', invoice_text, re.IGNORECASE)
            if tax_match:
                data['stated_tax'] = float(tax_match.group(1).replace(',', ''))
            
            total_match = re.search(r'total.*?\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', invoice_text, re.IGNORECASE)
            if total_match:
                data['total'] = float(total_match.group(1).replace(',', ''))
            
            return data
            
        except Exception as e:
            log.error(f"Error extracting invoice data: {e}")
            return {}
    
    def execute_hardcoded_tool(self, tool_name: str, invoice_data: Dict[str, Any], invoice_text: str) -> ToolResult:
        """Execute a hardcoded tool with appropriate parameters"""
        try:
            if tool_name == 'amount_validator':
                return self.hardcoded_tools.amount_validator(
                    amounts=invoice_data.get('amounts', []),
                    invoice_total=invoice_data.get('total')
                )
            elif tool_name == 'tax_calculator':
                return self.hardcoded_tools.tax_calculator(
                    subtotal=invoice_data.get('subtotal', 0),
                    tax_rate=invoice_data.get('tax_rate', 0),
                    stated_tax=invoice_data.get('stated_tax', 0)
                )
            elif tool_name == 'date_analyzer':
                return self.hardcoded_tools.date_analyzer(
                    invoice_date=invoice_data.get('invoice_date', ''),
                    due_date=invoice_data.get('due_date')
                )
            elif tool_name == 'vendor_authenticator':
                # You can maintain an approved vendor list
                approved_vendors = ['ACME Corp', 'Beta Industries', 'Delta LLC', 'Gamma Tech']
                return self.hardcoded_tools.vendor_authenticator(
                    vendor_name=invoice_data.get('vendor_name', ''),
                    vendor_email=invoice_data.get('vendor_email'),
                    approved_vendors=approved_vendors
                )
            elif tool_name == 'format_inspector':
                return self.hardcoded_tools.format_inspector(invoice_text)
            else:
                return ToolResult(False, None, f"Unknown hardcoded tool: {tool_name}")
                
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    async def execute_llm_agent(self, agent_type: str, invoice_data: str) -> AgentResponse:
        """Execute LLM-based agent analysis"""
        start_time = time.time()
        
        if not self.model:
            return AgentResponse(
                agent_type=agent_type,
                analysis="LLM not available - API key missing",
                risk_score=5,
                confidence=1,
                red_flags=["LLM_UNAVAILABLE"],
                execution_time=time.time() - start_time,
                tool_used="llm_unavailable"
            )
        
        agent_config = self.available_agents.get(agent_type, {})
        
        prompt = f"""
        You are a {agent_type.replace('_', ' ').title()} specialist for invoice fraud detection.
        
        TASK: {agent_config.get('description', 'Analyze this invoice for fraud indicators')}
        
        INVOICE DATA:
        {invoice_data}
        
        Please analyze this invoice and provide:
        1. Risk Score (1-10, where 10 is highest fraud risk)
        2. Confidence Level (1-10, where 10 is highest confidence)
        3. Analysis (detailed explanation of findings)
        4. Red Flags (list any concerning patterns)
        
        Format your response as JSON:
        {{
            "risk_score": <1-10>,
            "confidence": <1-10>,
            "analysis": "<detailed analysis>",
            "red_flags": ["<flag1>", "<flag2>", ...]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            result = json.loads(response_text)
            
            return AgentResponse(
                agent_type=agent_type,
                analysis=result.get('analysis', ''),
                risk_score=min(10, max(1, int(result.get('risk_score', 5)))),
                confidence=min(10, max(1, int(result.get('confidence', 5)))),
                red_flags=result.get('red_flags', []),
                execution_time=time.time() - start_time,
                tool_used="llm"
            )
            
        except Exception as e:
            log.error(f"Error in LLM agent {agent_type}: {e}")
            return AgentResponse(
                agent_type=agent_type,
                analysis=f"Error occurred: {str(e)}",
                risk_score=5,
                confidence=1,
                red_flags=["PROCESSING_ERROR"],
                execution_time=time.time() - start_time,
                tool_used="llm_error"
            )
    
    def convert_tool_result_to_agent_response(self, tool_name: str, tool_result: ToolResult) -> AgentResponse:
        """Convert hardcoded tool result to agent response format"""
        if not tool_result.success:
            return AgentResponse(
                agent_type=tool_name,
                analysis=f"Tool execution failed: {tool_result.error}",
                risk_score=5,
                confidence=1,
                red_flags=["TOOL_ERROR"],
                execution_time=tool_result.execution_time,
                tool_used="hardcoded_error"
            )
        
        result_data = tool_result.result
        risk_score = result_data.get('risk_score', 0)
        recommendation = result_data.get('recommendation', 'UNKNOWN')
        fraud_indicators = result_data.get('fraud_indicators', [])
        
        # Generate analysis text from result data
        analysis_parts = [f"Risk Score: {risk_score}/10", f"Recommendation: {recommendation}"]
        
        if fraud_indicators:
            analysis_parts.append(f"Found {len(fraud_indicators)} fraud indicators:")
            for indicator in fraud_indicators[:3]:  # Limit to top 3
                analysis_parts.append(f"- {indicator.get('type', 'Unknown')}: {indicator.get('severity', 'unknown')} severity")
        
        red_flags = [indicator.get('type', 'unknown') for indicator in fraud_indicators]
        
        # Calculate confidence based on data completeness
        confidence = 8 if len(str(result_data)) > 100 else 6
        
        return AgentResponse(
            agent_type=tool_name,
            analysis=". ".join(analysis_parts),
            risk_score=risk_score,
            confidence=confidence,
            red_flags=red_flags,
            execution_time=tool_result.execution_time,
            tool_used="hardcoded"
        )
    
    async def execute_agent_parallel(self, agent_type: str, invoice_text: str, invoice_data: Dict[str, Any]) -> AgentResponse:
        """Execute a single agent (hardcoded or LLM)"""
        agent_config = self.available_agents.get(agent_type, {})
        
        if agent_config.get('type') == 'hardcoded':
            tool_name = agent_config.get('tool')
            tool_result = self.execute_hardcoded_tool(tool_name, invoice_data, invoice_text)
            return self.convert_tool_result_to_agent_response(agent_type, tool_result)
        else:
            return await self.execute_llm_agent(agent_type, invoice_text)
    
    def select_agents(self, invoice_text: str) -> List[str]:
        """Select which agents to use based on invoice content"""
        # For now, use all available agents
        # In a more sophisticated system, you could use an LLM to select agents
        selected = list(self.available_agents.keys())
        log.info(f"Selected {len(selected)} agents: {', '.join(selected)}")
        return selected
    
    async def analyze_invoice_parallel(self, invoice_text: str) -> Dict[str, Any]:
        """Analyze invoice using parallel processing"""
        start_time = time.time()
        log.info("Starting parallel invoice fraud analysis...")
        
        # Extract structured data for hardcoded tools
        invoice_data = self.extract_invoice_data(invoice_text)
        log.info(f"Extracted data: vendor={invoice_data.get('vendor_name', 'N/A')}, amounts={len(invoice_data.get('amounts', []))}")
        
        # Select agents to use
        selected_agents = self.select_agents(invoice_text)
        
        if self.enable_parallel and self.executor:
            # Execute agents in parallel
            loop = asyncio.get_event_loop()
            
            # Create tasks for each agent
            tasks = []
            for agent_type in selected_agents:
                task = loop.run_in_executor(
                    self.executor,
                    lambda at=agent_type: asyncio.run(self.execute_agent_parallel(at, invoice_text, invoice_data))
                )
                tasks.append(task)
            
            # Wait for all agents to complete
            agent_responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_responses = []
            for i, response in enumerate(agent_responses):
                if isinstance(response, Exception):
                    log.error(f"Agent {selected_agents[i]} failed: {response}")
                    processed_responses.append(AgentResponse(
                        agent_type=selected_agents[i],
                        analysis=f"Agent failed: {str(response)}",
                        risk_score=5,
                        confidence=1,
                        red_flags=["AGENT_ERROR"],
                        execution_time=0.0
                    ))
                else:
                    processed_responses.append(response)
        else:
            # Execute agents sequentially
            processed_responses = []
            for agent_type in selected_agents:
                response = await self.execute_agent_parallel(agent_type, invoice_text, invoice_data)
                processed_responses.append(response)
        
        # Compile final results
        final_result = await self.compile_results(invoice_text, processed_responses)
        
        total_time = time.time() - start_time
        final_result['total_execution_time'] = total_time
        final_result['parallel_processing'] = self.enable_parallel
        final_result['agents_used'] = len(processed_responses)
        
        log.info(f"Analysis completed in {total_time:.2f}s using {len(processed_responses)} agents")
        
        return final_result
    
    async def compile_results(self, invoice_text: str, agent_responses: List[AgentResponse]) -> Dict[str, Any]:
        """Compile all agent responses into final fraud assessment"""
        
        # Calculate aggregate metrics
        risk_scores = [r.risk_score for r in agent_responses if r.risk_score > 0]
        confidence_scores = [r.confidence for r in agent_responses if r.confidence > 0]
        
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 5
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 5
        
        # Collect all red flags
        all_red_flags = []
        for response in agent_responses:
            all_red_flags.extend(response.red_flags)
        
        # Count flag occurrences
        flag_counts = {}
        for flag in all_red_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        # Determine final recommendation
        if avg_risk >= 7:
            recommendation = "REJECT"
        elif avg_risk >= 4:
            recommendation = "REVIEW"
        else:
            recommendation = "APPROVE"
        
        # Generate summary
        summary = f"Analysis of {len(agent_responses)} specialist agents reveals "
        if avg_risk >= 7:
            summary += "HIGH FRAUD RISK with multiple concerning indicators."
        elif avg_risk >= 4:
            summary += "MODERATE FRAUD RISK requiring manual review."
        else:
            summary += "LOW FRAUD RISK with minimal concerns."
        
        # Get top concerns
        top_concerns = sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_concern_list = [concern[0] for concern in top_concerns] if top_concerns else ["None"]
        
        return {
            'overall_risk_score': round(avg_risk, 1),
            'confidence_score': round(avg_confidence, 1),
            'recommendation': recommendation,
            'summary': summary,
            'top_concerns': top_concern_list,
            'total_red_flags': len(all_red_flags),
            'unique_red_flags': len(set(all_red_flags)),
            'agent_responses': [
                {
                    'agent': r.agent_type,
                    'risk_score': r.risk_score,
                    'confidence': r.confidence,
                    'analysis': r.analysis,
                    'red_flags': r.red_flags,
                    'execution_time': r.execution_time,
                    'tool_used': r.tool_used
                }
                for r in agent_responses
            ],
            'flag_frequency': flag_counts
        }
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)

def format_results(results: Dict[str, Any], invoice_text: str = "", output_file: str = None):
    """Format and display results"""
    print("\n" + "="*70)
    print("🔍 INVOICE FRAUD DETECTION RESULTS")
    print("="*70)
    
    # Overall assessment
    risk_score = results.get('overall_risk_score', 0)
    recommendation = results.get('recommendation', 'UNKNOWN')
    
    if recommendation == 'APPROVE':
        status_emoji = "✅"
        status_color = "LOW RISK"
    elif recommendation == 'REVIEW':
        status_emoji = "⚠️"
        status_color = "MEDIUM RISK"
    else:
        status_emoji = "❌"
        status_color = "HIGH RISK"
    
    print(f"\n{status_emoji} OVERALL ASSESSMENT: {status_color}")
    print(f"   Risk Score: {risk_score}/10")
    print(f"   Confidence: {results.get('confidence_score', 0)}/10")
    print(f"   Recommendation: {recommendation}")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"   {results.get('summary', 'No summary available')}")
    
    # Key metrics
    print(f"\n📊 ANALYSIS METRICS:")
    print(f"   Agents Used: {results.get('agents_used', 0)}")
    print(f"   Total Red Flags: {results.get('total_red_flags', 0)}")
    print(f"   Unique Issues: {results.get('unique_red_flags', 0)}")
    print(f"   Processing Time: {results.get('total_execution_time', 0):.2f}s")
    print(f"   Parallel Processing: {'Enabled' if results.get('parallel_processing') else 'Disabled'}")
    
    # Top concerns
    top_concerns = results.get('top_concerns', [])
    if top_concerns and top_concerns != ['None']:
        print(f"\n🚨 TOP CONCERNS:")
        for i, concern in enumerate(top_concerns[:3], 1):
            print(f"   {i}. {concern.replace('_', ' ').title()}")
    
    # Agent details
    print(f"\n🤖 AGENT ANALYSIS:")
    agent_responses = results.get('agent_responses', [])
    for response in agent_responses:
        tool_type = "🔧" if response['tool_used'].startswith('hardcoded') else "🧠"
        print(f"   {tool_type} {response['agent'].replace('_', ' ').title()}:")
        print(f"      Risk: {response['risk_score']}/10, Confidence: {response['confidence']}/10")
        print(f"      Time: {response['execution_time']:.3f}s, Tool: {response['tool_used']}")
        if response['red_flags']:
            print(f"      Flags: {', '.join(response['red_flags'][:3])}")
    
    # Final verdict
    print(f"\n{'='*70}")
    if recommendation == 'APPROVE':
        print(f"✅ VERDICT: Invoice appears legitimate - APPROVED for processing")
    elif recommendation == 'REVIEW':
        print(f"⚠️ VERDICT: Manual review recommended before processing")
    else:
        print(f"❌ VERDICT: High fraud risk detected - REJECT this invoice")
    
    print("="*70)
    
    # Save to file if requested
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"\n❌ Failed to save results to {output_file}: {str(e)}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Agent Invoice Fraud Detection with Parallel Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_detector.py --demo
  python main_detector.py --invoice "INVOICE DATA HERE"
  python main_detector.py --file invoice.txt
  python main_detector.py --demo --parallel --max-workers 6
  python main_detector.py --file invoice.json --output results.json

Available Tools:
  Hardcoded Tools: Fast, deterministic fraud detection
  LLM Agents: Contextual analysis for complex patterns
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--invoice", help="Invoice data as string")
    input_group.add_argument("--file", help="File containing invoice data")
    input_group.add_argument("--demo", action="store_true", help="Use demo invoice data")
    
    # Processing options
    parser.add_argument("--parallel", action="store_true", default=True, help="Enable parallel processing (default)")
    parser.add_argument("--sequential", action="store_true", help="Use sequential processing")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel workers (default: 4)")
    
    # Output options
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize detector
    enable_parallel = args.parallel and not args.sequential
    detector = ParallelInvoiceFraudDetector(
        max_workers=args.max_workers,
        enable_parallel=enable_parallel
    )
    
    # Get invoice data
    try:
        if args.demo:
            invoice_text = detector.demo_invoice
            if not args.quiet:
                print("Using demo invoice data...")
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                content = f.read()
                try:
                    # Try to parse as JSON first
                    data = json.loads(content)
                    invoice_text = data.get('invoice_text', content)
                except json.JSONDecodeError:
                    # Use as plain text
                    invoice_text = content
        else:
            invoice_text = args.invoice
            
        if not invoice_text.strip():
            print("❌ Error: No invoice data provided")
            return 3
            
    except FileNotFoundError:
        print(f"❌ Error: File '{args.file}' not found")
        return 3
    except Exception as e:
        print(f"❌ Error reading input: {e}")
        return 3
    
    # Analyze invoice
    try:
        if not args.quiet:
            print(f"\n🔍 Starting fraud analysis...")
            print(f"   Parallel Processing: {'Enabled' if enable_parallel else 'Disabled'}")
            print(f"   Max Workers: {args.max_workers}")
        
        results = await detector.analyze_invoice_parallel(invoice_text)
        
        # Format and display results
        format_results(results, invoice_text, args.output)
        
        # Return appropriate exit code
        recommendation = results.get('recommendation', 'UNKNOWN')
        if recommendation == 'APPROVE':
            return 0  # Success, low risk
        elif recommendation == 'REVIEW':
            return 1  # Medium risk
        else:
            return 2  # High risk
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")
        return 3

if __name__ == "__main__":
    exit_code = asyncio.run(main())
'''
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
'''