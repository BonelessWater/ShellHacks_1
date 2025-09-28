#!/usr/bin/env python3
"""
Working Multi-Gemini Model Fraud Analyzer
Tests available Gemini models with proper rate limiting and model names
"""

import asyncio
import json
import os
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

class WorkingMultiGeminiFraudAnalyzer:
    """Compare available Gemini models on fraud detection with proper error handling"""
    
    def __init__(self):
        """Initialize the analyzer"""
        load_dotenv('.env')
        self.setup_gemini_models()
        self.fraud_data = None
        
    def setup_gemini_models(self):
        """Setup working Gemini models with correct names"""
        print("🔧 Setting up available Gemini models...")
        
        try:
            import google.generativeai as genai
            google_key = os.getenv('GOOGLE_API_KEY')
            
            if google_key and google_key != 'your_google_api_key_here':
                genai.configure(api_key=google_key)
                
                # Working Gemini models (tested names)
                self.gemini_models = {
                    'gemini-1.5-pro': {
                        'name': 'Gemini 1.5 Pro',
                        'description': 'Most capable model for complex reasoning',
                        'model': None
                    },
                    'gemini-1.5-flash': {
                        'name': 'Gemini 1.5 Flash', 
                        'description': 'Fast, efficient model for quick analysis',
                        'model': None
                    },
                    'gemini-pro': {
                        'name': 'Gemini Pro',
                        'description': 'Stable production model',
                        'model': None
                    }
                }
                
                # Test each model availability
                for model_id, model_info in self.gemini_models.items():
                    try:
                        model_info['model'] = genai.GenerativeModel(model_id)
                        # Test with a simple prompt
                        test_response = model_info['model'].generate_content("Test: What is 2+2?")
                        if test_response.text:
                            print(f"✅ {model_info['name']} working")
                        else:
                            print(f"⚠️ {model_info['name']} responded but empty")
                    except Exception as e:
                        print(f"❌ {model_info['name']} failed: {str(e)[:100]}...")
                        model_info['error'] = str(e)
                
            else:
                print("❌ Google API key not found")
                self.gemini_models = {}
                
        except ImportError:
            print("❌ Google Generative AI library not installed")
            self.gemini_models = {}
    
    def load_fraud_data(self):
        """Load fraud data samples"""
        try:
            with open('../data/bigquery_fraud_samples.json', 'r') as f:
                data = json.load(f)
                self.fraud_data = data['datasets']
                print(f"📊 Loaded fraud data: {sum(len(dataset) for dataset in self.fraud_data.values())} samples")
                return True
        except FileNotFoundError:
            print("❌ Fraud samples file not found.")
            return False
    
    def create_simple_fraud_prompt(self):
        """Create a single, focused fraud analysis prompt"""
        return """
You are a fraud detection expert. Analyze this transaction data and identify the TOP 3 FRAUD PATTERNS.

For each pattern, provide:
1. **Pattern Name**: Clear, specific name
2. **Evidence**: Specific data points that support this pattern
3. **Risk Level**: High/Medium/Low

Be concise and focus on patterns that clearly distinguish fraud from legitimate transactions.
        """
    
    def query_gemini_model_safe(self, model_id, model_info, prompt, data_sample):
        """Safely query a Gemini model with error handling"""
        if 'error' in model_info or not model_info['model']:
            return {
                "model_id": model_id,
                "model_name": model_info['name'],
                "error": model_info.get('error', 'Model not available'),
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Create smaller data summary to avoid rate limits
            data_summary = {
                "total_samples": sum(len(dataset) for dataset in data_sample.values()),
                "datasets": list(data_sample.keys()),
                "sample_fraud_cases": []
            }
            
            # Add a few fraud examples
            for name, dataset in data_sample.items():
                fraud_examples = [record for record in dataset 
                                if record.get('isFraud', record.get('Class', 0)) == 1][:2]
                if fraud_examples:
                    data_summary["sample_fraud_cases"].extend(fraud_examples)
            
            # Limit to first 3 fraud cases to keep prompt small
            data_summary["sample_fraud_cases"] = data_summary["sample_fraud_cases"][:3]
            
            full_prompt = f"{prompt}\n\nFRAUD DATA:\n{json.dumps(data_summary, indent=2, default=str)[:2000]}"
            
            start_time = time.time()
            response = model_info['model'].generate_content(full_prompt)
            response_time = time.time() - start_time
            
            return {
                "model_id": model_id,
                "model_name": model_info['name'],
                "model_description": model_info['description'],
                "analysis": response.text,
                "response_time": response_time,
                "analysis_length": len(response.text),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                return {
                    "model_id": model_id,
                    "model_name": model_info['name'],
                    "error": "Rate limit exceeded - try again later",
                    "timestamp": datetime.now().isoformat(),
                    "rate_limited": True
                }
            else:
                return {
                    "model_id": model_id,
                    "model_name": model_info['name'],
                    "error": error_msg[:200] + "..." if len(error_msg) > 200 else error_msg,
                    "timestamp": datetime.now().isoformat()
                }
    
    async def run_working_gemini_comparison(self):
        """Run comparison with working models only"""
        print("🚀 WORKING GEMINI MODEL COMPARISON")
        print("=" * 40)
        
        if not self.load_fraud_data():
            return None
        
        prompt = self.create_simple_fraud_prompt()
        
        # Test working models only with delays
        all_results = []
        working_models = [(mid, info) for mid, info in self.gemini_models.items() 
                         if 'error' not in info and info['model']]
        
        print(f"🧪 Testing {len(working_models)} working models...")
        
        for i, (model_id, model_info) in enumerate(working_models):
            print(f"🔄 [{i+1}/{len(working_models)}] Testing {model_info['name']}...")
            
            result = self.query_gemini_model_safe(model_id, model_info, prompt, self.fraud_data)
            all_results.append(result)
            
            # Add longer delay to avoid rate limits
            if i < len(working_models) - 1:  # Don't wait after last model
                print("⏳ Waiting 10 seconds to avoid rate limits...")
                time.sleep(10)
        
        # Add failed models to results
        failed_models = [(mid, info) for mid, info in self.gemini_models.items() 
                        if 'error' in info or not info['model']]
        
        for model_id, model_info in failed_models:
            all_results.append({
                "model_id": model_id,
                "model_name": model_info['name'],
                "error": model_info.get('error', 'Model not available'),
                "timestamp": datetime.now().isoformat()
            })
        
        # Analyze results
        successful_results = [r for r in all_results if 'error' not in r]
        failed_results = [r for r in all_results if 'error' in r]
        rate_limited = [r for r in failed_results if r.get('rate_limited', False)]
        
        # Create results summary
        comparison_results = {
            "comparison_timestamp": datetime.now().isoformat(),
            "total_models_attempted": len(self.gemini_models),
            "successful_models": len(successful_results),
            "failed_models": len(failed_results),
            "rate_limited_models": len(rate_limited),
            "success_rate": len(successful_results) / len(all_results) * 100 if all_results else 0,
            "model_results": all_results,
            "performance_summary": self.create_performance_summary(successful_results),
            "bigquery_project": "vaulted-timing-473322-f9",
            "datasets_analyzed": list(self.fraud_data.keys()),
            "total_fraud_samples": sum(len(dataset) for dataset in self.fraud_data.values())
        }
        
        # Save results
        output_file = '../data/working_gemini_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        self.print_results_summary(comparison_results)
        
        return comparison_results
    
    def create_performance_summary(self, successful_results):
        """Create performance summary for successful models"""
        if not successful_results:
            return {"message": "No successful results to analyze"}
        
        summary = {}
        for result in successful_results:
            model_name = result['model_name']
            summary[model_name] = {
                "analysis_length": result.get('analysis_length', 0),
                "response_time": result.get('response_time', 0),
                "model_description": result.get('model_description', ''),
                "success": True
            }
        
        # Find best performing model
        if summary:
            best_model = max(summary.items(), key=lambda x: x[1]['analysis_length'])
            summary['best_overall'] = best_model[0]
            summary['fastest_model'] = min(summary.items(), key=lambda x: x[1]['response_time'])[0]
        
        return summary
    
    def print_results_summary(self, results):
        """Print comprehensive results summary"""
        print(f"\n🎉 WORKING GEMINI COMPARISON COMPLETE!")
        print("=" * 40)
        
        print(f"📊 Results Summary:")
        print(f"   • Models Attempted: {results['total_models_attempted']}")
        print(f"   • Successful Models: {results['successful_models']}")
        print(f"   • Failed Models: {results['failed_models']}")
        print(f"   • Rate Limited: {results['rate_limited_models']}")
        print(f"   • Success Rate: {results['success_rate']:.1f}%")
        
        successful_results = [r for r in results['model_results'] if 'error' not in r]
        failed_results = [r for r in results['model_results'] if 'error' in r]
        
        if successful_results:
            print(f"\n✅ Successful Models:")
            for result in successful_results:
                print(f"   • {result['model_name']}")
                print(f"     📝 Analysis: {result.get('analysis_length', 0)} characters")
                print(f"     ⚡ Response Time: {result.get('response_time', 0):.2f}s")
                print(f"     🎯 Preview: {result.get('analysis', '')[:100]}...")
        
        if failed_results:
            print(f"\n❌ Failed Models:")
            for result in failed_results:
                print(f"   • {result['model_name']}: {result.get('error', 'Unknown error')[:100]}...")
        
        perf_summary = results.get('performance_summary', {})
        if perf_summary.get('best_overall'):
            print(f"\n🏆 Best Overall: {perf_summary['best_overall']}")
        if perf_summary.get('fastest_model'):
            print(f"⚡ Fastest Model: {perf_summary['fastest_model']}")
        
        print(f"\n💾 Results saved to: ../data/working_gemini_comparison.json")

def main():
    """Main function"""
    analyzer = WorkingMultiGeminiFraudAnalyzer()
    return asyncio.run(analyzer.run_working_gemini_comparison())

if __name__ == "__main__":
    main()