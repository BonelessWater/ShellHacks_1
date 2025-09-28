#!/usr/bin/env python3
"""
LLM Fraud Pattern Analyzer

This script analyzes invoice data using multiple LLMs to identify and categorize
different types of fraud patterns. It queries various AI models and stores their
outputs for comparison and analysis.
"""

import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import sys
from dataclasses import dataclass, asdict
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports (install with: pip install openai google-generativeai)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not available. Install with: pip install openai")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Google Generative AI library not available. Install with: pip install google-generativeai")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Data class to store LLM responses"""
    model_name: str
    timestamp: str
    prompt_type: str
    response: str
    analysis_time_seconds: float
    fraud_patterns_identified: List[str]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]

class LLMFraudAnalyzer:
    """Main class for analyzing fraud patterns using multiple LLMs"""
    
    def __init__(self, invoice_data_path: str, output_path: str):
        self.invoice_data_path = invoice_data_path
        self.output_path = output_path
        self.invoice_data = self.load_invoice_data()
        self.results: List[LLMResponse] = []
        self.setup_apis()
    
    def load_invoice_data(self) -> List[Dict]:
        """Load invoice data from JSON file"""
        try:
            with open(self.invoice_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading invoice data: {e}")
            return []
    
    def setup_apis(self):
        """Setup API clients for different LLMs"""
        # OpenAI API setup
        if OPENAI_AVAILABLE:
            self.openai_key = os.getenv('OPENAI_API_KEY')
            if self.openai_key:
                self.openai_client = openai.OpenAI(api_key=self.openai_key)
            else:
                logger.warning("OPENAI_API_KEY not found in environment variables")
                self.openai_client = None
        
        # Google Gemini setup
        if GOOGLE_AVAILABLE:
            gemini_key = os.getenv('GOOGLE_API_KEY')
            if gemini_key:
                genai.configure(api_key=gemini_key)
            else:
                logger.warning("GOOGLE_API_KEY not found in environment variables")
    
    def get_fraud_analysis_prompts(self) -> Dict[str, str]:
        """Generate different types of prompts for fraud analysis"""
        
        # Sample of fraudulent invoices for context
        fraudulent_samples = [inv for inv in self.invoice_data if inv.get('is_fraudulent', False)][:3]
        legitimate_samples = [inv for inv in self.invoice_data if not inv.get('is_fraudulent', False)][:3]
        
        base_context = f"""
        You are an expert fraud detection analyst. I have a dataset of {len(self.invoice_data)} invoices, 
        including both legitimate and fraudulent examples.
        
        Here are 3 examples of fraudulent invoices:
        {json.dumps(fraudulent_samples, indent=2)}
        
        Here are 3 examples of legitimate invoices:
        {json.dumps(legitimate_samples, indent=2)}
        """
        
        return {
            "pattern_identification": base_context + """
            
            TASK: Analyze these invoices and identify all distinct fraud patterns present in the fraudulent examples.
            
            Please provide:
            1. A comprehensive list of fraud patterns you can identify
            2. Specific indicators for each pattern
            3. Risk severity levels (High/Medium/Low) for each pattern
            4. How each pattern could be detected algorithmically
            
            Format your response as a structured analysis with clear categories.
            """,
            
            "detection_rules": base_context + """
            
            TASK: Create detection rules that could identify these fraud patterns automatically.
            
            Please provide:
            1. Specific business rules for each fraud pattern
            2. Threshold values where applicable
            3. Combination rules (when multiple indicators suggest fraud)
            4. False positive mitigation strategies
            
            Format as actionable detection rules that could be implemented in code.
            """,
            
            "risk_scoring": base_context + """
            
            TASK: Design a risk scoring system for invoice fraud detection.
            
            Please provide:
            1. A scoring methodology (0-100 scale)
            2. Weight assignments for different fraud indicators
            3. Scoring examples for the provided invoices
            4. Threshold recommendations for flagging invoices
            
            Include specific numeric scores for the example invoices.
            """,
            
            "comparative_analysis": base_context + """
            
            TASK: Compare and contrast the fraudulent vs legitimate invoices.
            
            Please provide:
            1. Key differences between fraudulent and legitimate invoices
            2. Patterns that appear in both but differ in execution
            3. Statistical observations about amounts, vendors, timing, etc.
            4. Recommendations for improving fraud detection
            
            Focus on actionable insights for fraud prevention.
            """
        }
    
    async def query_openai_gpt(self, prompt: str, prompt_type: str) -> Optional[LLMResponse]:
        """Query OpenAI GPT models"""
        if not OPENAI_AVAILABLE or not hasattr(self, 'openai_client') or not self.openai_client:
            logger.warning("OpenAI not available or API key not set")
            return None
        
        try:
            start_time = time.time()
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert fraud detection analyst with deep knowledge of financial crimes and invoice fraud patterns."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            analysis_time = time.time() - start_time
            response_text = response.choices[0].message.content
            
            # Extract fraud patterns mentioned (simple keyword extraction)
            fraud_keywords = [
                "vendor impersonation", "price inflation", "duplicate billing", "fake vendor",
                "emergency scam", "hours padding", "service bundling", "post-dated",
                "tax manipulation", "address fraud", "payment terms abuse"
            ]
            
            patterns_found = [keyword for keyword in fraud_keywords 
                            if keyword.lower() in response_text.lower()]
            
            # Simple confidence scoring based on response length and detail
            confidence_score = min(len(response_text) / 2000.0, 1.0)
            
            return LLMResponse(
                model_name="GPT-4",
                timestamp=datetime.now().isoformat(),
                prompt_type=prompt_type,
                response=response_text,
                analysis_time_seconds=analysis_time,
                fraud_patterns_identified=patterns_found,
                confidence_scores={"overall": confidence_score},
                metadata={"token_usage": response.usage.total_tokens if response.usage else 0}
            )
        
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            return None
    
    async def query_google_gemini(self, prompt: str, prompt_type: str) -> Optional[LLMResponse]:
        """Query Google Gemini models"""
        if not GOOGLE_AVAILABLE or not os.getenv('GOOGLE_API_KEY'):
            logger.warning("Google Gemini not available or API key not set")
            return None
        
        try:
            start_time = time.time()
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4000,
                    temperature=0.1
                )
            )
            
            analysis_time = time.time() - start_time
            response_text = response.text
            
            # Extract fraud patterns
            fraud_keywords = [
                "vendor impersonation", "price inflation", "duplicate billing", "fake vendor",
                "emergency scam", "hours padding", "service bundling", "post-dated",
                "tax manipulation", "address fraud", "payment terms abuse"
            ]
            
            patterns_found = [keyword for keyword in fraud_keywords 
                            if keyword.lower() in response_text.lower()]
            
            confidence_score = min(len(response_text) / 2000.0, 1.0)
            
            return LLMResponse(
                model_name="Gemini-Pro",
                timestamp=datetime.now().isoformat(),
                prompt_type=prompt_type,
                response=response_text,
                analysis_time_seconds=analysis_time,
                fraud_patterns_identified=patterns_found,
                confidence_scores={"overall": confidence_score},
                metadata={"response_length": len(response_text)}
            )
        
        except Exception as e:
            logger.error(f"Error querying Google Gemini: {e}")
            return None
    

    
    async def run_analysis(self):
        """Run the complete fraud analysis using all available LLMs"""
        logger.info("Starting fraud pattern analysis...")
        
        prompts = self.get_fraud_analysis_prompts()
        
        # Query all LLMs for each prompt type
        for prompt_type, prompt in prompts.items():
            logger.info(f"Analyzing: {prompt_type}")
            
            # Run queries concurrently for better performance
            tasks = []
            
            if OPENAI_AVAILABLE and hasattr(self, 'openai_client') and self.openai_client:
                tasks.append(self.query_openai_gpt(prompt, prompt_type))
            
            if GOOGLE_AVAILABLE and os.getenv('GOOGLE_API_KEY'):
                tasks.append(self.query_google_gemini(prompt, prompt_type))
            
            # Execute all queries
            if tasks:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for response in responses:
                    if isinstance(response, LLMResponse):
                        self.results.append(response)
                        logger.info(f"Successfully analyzed with {response.model_name}")
                    elif isinstance(response, Exception):
                        logger.error(f"Error in analysis: {response}")
            
            # Rate limiting - wait between different prompt types
            await asyncio.sleep(2)
        
        logger.info(f"Analysis complete. Generated {len(self.results)} responses.")
        self.save_results()
    
    def save_results(self):
        """Save all LLM responses to the output file"""
        try:
            # Convert dataclass objects to dictionaries for JSON serialization
            results_dict = [asdict(result) for result in self.results]
            
            # Create comprehensive output structure
            output_data = {
                "analysis_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "invoice_data_source": self.invoice_data_path,
                    "total_invoices_analyzed": len(self.invoice_data),
                    "fraudulent_invoices_count": len([inv for inv in self.invoice_data if inv.get('is_fraudulent', False)]),
                    "legitimate_invoices_count": len([inv for inv in self.invoice_data if not inv.get('is_fraudulent', False)]),
                    "llm_responses_count": len(self.results)
                },
                "llm_responses": results_dict,
                "summary_analysis": self.generate_summary()
            }
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {self.output_path}")
        
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the analysis results"""
        if not self.results:
            return {"error": "No results to summarize"}
        
        # Count models used
        models_used = list(set(result.model_name for result in self.results))
        
        # Count prompt types analyzed
        prompt_types = list(set(result.prompt_type for result in self.results))
        
        # Aggregate fraud patterns identified
        all_patterns = []
        for result in self.results:
            all_patterns.extend(result.fraud_patterns_identified)
        
        pattern_frequency = {}
        for pattern in all_patterns:
            pattern_frequency[pattern] = pattern_frequency.get(pattern, 0) + 1
        
        # Calculate average analysis time
        avg_analysis_time = sum(result.analysis_time_seconds for result in self.results) / len(self.results)
        
        return {
            "models_used": models_used,
            "prompt_types_analyzed": prompt_types,
            "total_responses": len(self.results),
            "fraud_patterns_frequency": pattern_frequency,
            "average_analysis_time_seconds": avg_analysis_time,
            "most_common_patterns": sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        }

# CLI interface
def main():
    """Main function to run the fraud analysis"""
    import argparse
    
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_file = os.path.join(script_dir, ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info("Loaded environment variables from .env file")
    except ImportError:
        logger.warning("python-dotenv not installed, using system environment variables")
    
    parser = argparse.ArgumentParser(description="Analyze invoice fraud patterns using multiple LLMs")
    parser.add_argument("--input", default="../data/invoice_training_data.json", 
                       help="Path to invoice data JSON file")
    parser.add_argument("--output", default="../data/llm_fraud_analysis_results.json",
                       help="Path to output results file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input)
    output_path = os.path.join(script_dir, args.output)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Create analyzer and run analysis
    analyzer = LLMFraudAnalyzer(input_path, output_path)
    
    # Run the async analysis
    try:
        asyncio.run(analyzer.run_analysis())
        logger.info("Fraud analysis completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())