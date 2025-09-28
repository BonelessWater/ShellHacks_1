#!/usr/bin/env python3
"""
Multi-LLM Fraud Pattern Analyzer
Compares Gemini, OpenAI GPT, and Anthropic Claude on fraud detection accuracy
"""

import asyncio
import json
import os
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

class MultiLLMFraudAnalyzer:
    """Compare multiple LLMs on fraud detection accuracy"""
    
    def __init__(self):
        """Initialize the multi-LLM analyzer"""
        load_dotenv('.env')
        self.setup_llm_clients()
        self.fraud_data = None
        
    def setup_llm_clients(self):
        """Setup all available LLM clients"""
        print("🔧 Setting up LLM clients...")
        
        # OpenAI GPT
        try:
            import openai
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key and openai_key != 'your_openai_api_key_here':
                self.openai_client = openai.OpenAI(api_key=openai_key)
                print("✅ OpenAI GPT-4 configured")
            else:
                self.openai_client = None
                print("❌ OpenAI API key not found")
        except ImportError:
            self.openai_client = None
            print("❌ OpenAI library not installed")
        
        # Google Gemini
        try:
            import google.generativeai as genai
            google_key = os.getenv('GOOGLE_API_KEY')
            if google_key and google_key != 'your_google_api_key_here':
                genai.configure(api_key=google_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                print("✅ Google Gemini 2.0 Flash configured")
            else:
                self.gemini_model = None
                print("❌ Google API key not found")
        except ImportError:
            self.gemini_model = None
            print("❌ Google Generative AI library not installed")
        
        # Anthropic Claude
        try:
            import anthropic
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                print("✅ Anthropic Claude configured")
            else:
                self.anthropic_client = None
                print("❌ Anthropic API key not found")
        except ImportError:
            self.anthropic_client = None
            print("❌ Anthropic library not installed")
    
    def load_fraud_data(self):
        """Load BigQuery fraud samples"""
        try:
            with open('../data/bigquery_fraud_samples.json', 'r') as f:
                data = json.load(f)
                self.fraud_data = data['datasets']
                print(f"📊 Loaded fraud data: {sum(len(dataset) for dataset in self.fraud_data.values())} samples")
                return True
        except FileNotFoundError:
            print("❌ Fraud samples file not found. Run bigquery_fraud_analyzer_fixed.py first.")
            return False
    
    def create_fraud_analysis_prompt(self):
        """Create comprehensive fraud analysis prompt"""
        return """
You are a world-class fraud detection expert analyzing real financial transaction data.

TASK: Analyze the provided fraud data samples and identify the TOP 5 FRAUD PATTERNS with specific accuracy.

For each pattern, provide:
1. **Pattern Name**: Clear, descriptive name
2. **Evidence**: Specific data points supporting this pattern
3. **Risk Level**: High/Medium/Low
4. **Detection Strategy**: How to identify this pattern in production
5. **False Positive Risk**: Likelihood of flagging legitimate transactions

Focus on actionable insights that can be implemented in a real fraud detection system.

**Key Areas to Analyze:**
- Transaction types and amounts
- Account balance patterns
- Card/payment method characteristics  
- Geographic/email domain patterns
- Temporal patterns

**Data Context:**
- IEEE CIS Fraud: Credit card fraud competition dataset
- Credit Card Fraud: Classic fraud detection dataset
- PaySim: Mobile money fraud simulation
- Relational Fraud: Multi-table fraud indicators

Be specific, data-driven, and focus on patterns that show clear differences between fraud and legitimate transactions.
        """
    
    async def query_openai_gpt(self, prompt, data_sample):
        """Query OpenAI GPT-4"""
        if not self.openai_client:
            return {"error": "OpenAI client not available", "llm": "OpenAI GPT-4"}
        
        try:
            # Create data summary for prompt
            data_summary = {
                name: {
                    "sample_size": len(dataset),
                    "fraud_cases": sum(1 for record in dataset if record.get('isFraud', record.get('Class', 0)) == 1),
                    "sample_records": dataset[:3]
                }
                for name, dataset in data_sample.items()
            }
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a fraud detection expert with deep experience in financial crime analysis."},
                    {"role": "user", "content": f"{prompt}\n\nFRAUD DATA SAMPLE:\n{json.dumps(data_summary, indent=2, default=str)[:4000]}"}
                ],
                max_tokens=2500,
                temperature=0.2
            )
            
            return {
                "llm": "OpenAI GPT-4o Mini",
                "analysis": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "llm": "OpenAI GPT-4o Mini",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def query_google_gemini(self, prompt, data_sample):
        """Query Google Gemini"""
        if not self.gemini_model:
            return {"error": "Gemini client not available", "llm": "Google Gemini"}
        
        try:
            # Create data summary for prompt
            data_summary = {
                name: {
                    "sample_size": len(dataset),
                    "fraud_cases": sum(1 for record in dataset if record.get('isFraud', record.get('Class', 0)) == 1),
                    "sample_records": dataset[:3]
                }
                for name, dataset in data_sample.items()
            }
            
            full_prompt = f"{prompt}\n\nFRAUD DATA SAMPLE:\n{json.dumps(data_summary, indent=2, default=str)[:4000]}"
            response = self.gemini_model.generate_content(full_prompt)
            
            return {
                "llm": "Google Gemini 2.0 Flash",
                "analysis": response.text,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "llm": "Google Gemini 2.0 Flash", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def query_anthropic_claude(self, prompt, data_sample):
        """Query Anthropic Claude"""
        if not self.anthropic_client:
            return {"error": "Anthropic client not available", "llm": "Anthropic Claude"}
        
        try:
            # Create data summary for prompt
            data_summary = {
                name: {
                    "sample_size": len(dataset),
                    "fraud_cases": sum(1 for record in dataset if record.get('isFraud', record.get('Class', 0)) == 1),
                    "sample_records": dataset[:3]
                }
                for name, dataset in data_sample.items()
            }
            
            full_prompt = f"{prompt}\n\nFRAUD DATA SAMPLE:\n{json.dumps(data_summary, indent=2, default=str)[:4000]}"
            
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-5-sonnet-20241022",
                max_tokens=2500,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            return {
                "llm": "Anthropic Claude 3.5 Sonnet",
                "analysis": response.content[0].text,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "llm": "Anthropic Claude 3.5 Sonnet",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_multi_llm_comparison(self):
        """Run fraud analysis comparison across all LLMs"""
        print("🚀 MULTI-LLM FRAUD ANALYSIS COMPARISON")
        print("=" * 45)
        
        # Load fraud data
        if not self.load_fraud_data():
            return None
        
        # Create analysis prompt
        prompt = self.create_fraud_analysis_prompt()
        
        # Prepare LLM tasks
        llm_tasks = []
        
        if self.openai_client:
            print("🤖 Queuing OpenAI GPT-4o Mini...")
            llm_tasks.append(self.query_openai_gpt(prompt, self.fraud_data))
        
        if self.gemini_model:
            print("🤖 Queuing Google Gemini 2.0 Flash...")
            # Gemini is sync, so wrap it
            async def gemini_task():
                return self.query_google_gemini(prompt, self.fraud_data)
            llm_tasks.append(gemini_task())
        
        if self.anthropic_client:
            print("🤖 Queuing Anthropic Claude 3.5 Sonnet...")
            llm_tasks.append(self.query_anthropic_claude(prompt, self.fraud_data))
        
        if not llm_tasks:
            print("❌ No LLM clients available!")
            return None
        
        print(f"\n🔄 Running analysis with {len(llm_tasks)} LLM(s)...")
        
        # Execute all LLM queries
        results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        
        # Process results
        llm_analyses = []
        for result in results:
            if isinstance(result, Exception):
                llm_analyses.append({
                    "llm": "Unknown",
                    "error": str(result),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                llm_analyses.append(result)
        
        # Create comprehensive comparison
        comparison_results = {
            "comparison_timestamp": datetime.now().isoformat(),
            "bigquery_project": "vaulted-timing-473322-f9",
            "datasets_analyzed": list(self.fraud_data.keys()),
            "total_samples": sum(len(dataset) for dataset in self.fraud_data.values()),
            "llm_count": len(llm_analyses),
            "llm_analyses": llm_analyses,
            "analysis_summary": self.create_analysis_summary(llm_analyses)
        }
        
        # Save results
        output_file = '../data/multi_llm_fraud_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        print(f"\n🎉 MULTI-LLM COMPARISON COMPLETE!")
        print("=" * 35)
        print(f"📁 Results saved to: {output_file}")
        print(f"📊 Analyzed {comparison_results['total_samples']} fraud samples")
        print(f"🤖 Used {len([r for r in llm_analyses if 'error' not in r])} LLM(s) successfully")
        
        # Show summary
        for result in llm_analyses:
            if 'error' not in result:
                print(f"✅ {result['llm']}: Analysis completed")
                print(f"   📝 Length: {len(result.get('analysis', ''))} characters")
                if 'tokens_used' in result:
                    print(f"   🔢 Tokens: {result['tokens_used']}")
            else:
                print(f"❌ {result['llm']}: {result['error']}")
        
        return comparison_results
    
    def create_analysis_summary(self, llm_analyses):
        """Create summary of all LLM analyses"""
        summary = {
            "successful_analyses": len([r for r in llm_analyses if 'error' not in r]),
            "failed_analyses": len([r for r in llm_analyses if 'error' in r]),
            "llms_tested": [r['llm'] for r in llm_analyses],
            "total_analysis_length": sum(len(r.get('analysis', '')) for r in llm_analyses),
            "average_analysis_length": 0
        }
        
        successful = [r for r in llm_analyses if 'error' not in r]
        if successful:
            summary["average_analysis_length"] = summary["total_analysis_length"] / len(successful)
        
        return summary

def main():
    """Main function"""
    analyzer = MultiLLMFraudAnalyzer()
    return asyncio.run(analyzer.run_multi_llm_comparison())

if __name__ == "__main__":
    main()