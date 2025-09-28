#!/usr/bin/env python3
"""
Multi-Gemini Model Fraud Analyzer
Tests different Gemini models on fraud detection accuracy
"""

import asyncio
import json
import os
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

class MultiGeminiFraudAnalyzer:
    """Compare multiple Gemini models on fraud detection accuracy"""
    
    def __init__(self):
        """Initialize the multi-Gemini analyzer"""
        load_dotenv('.env')
        self.setup_gemini_models()
        self.fraud_data = None
        
    def setup_gemini_models(self):
        """Setup multiple Gemini model clients"""
        print("🔧 Setting up Gemini model variants...")
        
        try:
            import google.generativeai as genai
            google_key = os.getenv('GOOGLE_API_KEY')
            
            if google_key and google_key != 'your_google_api_key_here':
                genai.configure(api_key=google_key)
                
                # Available Gemini models to test
                self.gemini_models = {
                    'gemini-2.0-flash-exp': {
                        'name': 'Gemini 2.0 Flash Experimental',
                        'description': 'Latest experimental model with enhanced reasoning',
                        'model': None
                    },
                    'gemini-1.5-flash': {
                        'name': 'Gemini 1.5 Flash',
                        'description': 'Fast, efficient model for quick analysis',
                        'model': None
                    },
                    'gemini-1.5-flash-8b': {
                        'name': 'Gemini 1.5 Flash 8B',
                        'description': 'Smaller 8B parameter model',
                        'model': None
                    },
                    'gemini-1.5-pro': {
                        'name': 'Gemini 1.5 Pro',
                        'description': 'Most capable model for complex reasoning',
                        'model': None
                    },
                    'gemini-1.0-pro': {
                        'name': 'Gemini 1.0 Pro',
                        'description': 'Original stable Gemini model',
                        'model': None
                    }
                }
                
                # Initialize each model
                for model_id, model_info in self.gemini_models.items():
                    try:
                        model_info['model'] = genai.GenerativeModel(model_id)
                        print(f"✅ {model_info['name']} configured")
                    except Exception as e:
                        print(f"❌ {model_info['name']} failed: {e}")
                        model_info['error'] = str(e)
                
            else:
                print("❌ Google API key not found")
                self.gemini_models = {}
                
        except ImportError:
            print("❌ Google Generative AI library not installed")
            self.gemini_models = {}
    
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
    
    def create_fraud_analysis_prompts(self):
        """Create different prompts to test model capabilities"""
        return {
            'pattern_detection': """
You are a fraud detection expert. Analyze the provided transaction data and identify the TOP 3 FRAUD PATTERNS.

For each pattern, provide:
1. **Pattern Name**: Clear, specific name
2. **Evidence**: Data points supporting this pattern  
3. **Detection Rule**: How to identify this in production

Focus on patterns that clearly distinguish fraud from legitimate transactions.
            """,
            
            'risk_scoring': """
You are designing a fraud risk scoring system. Based on the transaction data, create a risk scoring model.

Provide:
1. **Top 5 Risk Factors**: Most predictive features
2. **Scoring Algorithm**: How to calculate risk scores (0-100)
3. **Thresholds**: Low/Medium/High risk cutoffs
4. **Validation**: How to test the model

Be specific and actionable.
            """,
            
            'anomaly_detection': """
You are an anomaly detection specialist. Analyze the transaction data for unusual patterns.

Identify:
1. **Statistical Anomalies**: Unusual amounts, frequencies, timings
2. **Behavioral Anomalies**: Unexpected user actions
3. **Contextual Anomalies**: Transactions that seem wrong given context

Focus on mathematically detectable anomalies.
            """
        }
    
    def query_gemini_model(self, model_id, model_info, prompt_name, prompt, data_sample):
        """Query a specific Gemini model"""
        if 'error' in model_info or not model_info['model']:
            return {
                "model_id": model_id,
                "model_name": model_info['name'],
                "prompt_type": prompt_name,
                "error": model_info.get('error', 'Model not available'),
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Create data summary for prompt
            data_summary = {
                name: {
                    "sample_size": len(dataset),
                    "fraud_cases": sum(1 for record in dataset if record.get('isFraud', record.get('Class', 0)) == 1),
                    "key_features": list(dataset[0].keys()) if dataset else [],
                    "sample_records": dataset[:2]  # Smaller sample for multiple models
                }
                for name, dataset in data_sample.items()
            }
            
            full_prompt = f"{prompt}\n\nFRAUD DATA SAMPLE:\n{json.dumps(data_summary, indent=2, default=str)[:3000]}"
            
            start_time = time.time()
            response = model_info['model'].generate_content(full_prompt)
            response_time = time.time() - start_time
            
            return {
                "model_id": model_id,
                "model_name": model_info['name'],
                "model_description": model_info['description'],
                "prompt_type": prompt_name,
                "analysis": response.text,
                "response_time": response_time,
                "analysis_length": len(response.text),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "model_id": model_id,
                "model_name": model_info['name'],
                "prompt_type": prompt_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_multi_gemini_comparison(self):
        """Run fraud analysis comparison across all Gemini models"""
        print("🚀 MULTI-GEMINI MODEL FRAUD ANALYSIS")
        print("=" * 40)
        
        # Load fraud data
        if not self.load_fraud_data():
            return None
        
        # Get analysis prompts
        prompts = self.create_fraud_analysis_prompts()
        
        # Test each model with each prompt
        all_results = []
        total_tests = len(self.gemini_models) * len(prompts)
        current_test = 0
        
        print(f"🧪 Running {total_tests} model-prompt combinations...")
        
        for model_id, model_info in self.gemini_models.items():
            for prompt_name, prompt in prompts.items():
                current_test += 1
                print(f"🔄 [{current_test}/{total_tests}] Testing {model_info['name']} on {prompt_name}...")
                
                result = self.query_gemini_model(model_id, model_info, prompt_name, prompt, self.fraud_data)
                all_results.append(result)
                
                # Add delay between requests to avoid rate limiting
                if 'error' not in result:
                    time.sleep(1)
        
        # Analyze results
        analysis_summary = self.analyze_results(all_results)
        
        # Create comprehensive comparison
        comparison_results = {
            "comparison_timestamp": datetime.now().isoformat(),
            "total_models_tested": len(self.gemini_models),
            "total_prompts_tested": len(prompts),
            "total_combinations": total_tests,
            "bigquery_project": "vaulted-timing-473322-f9",
            "datasets_analyzed": list(self.fraud_data.keys()),
            "total_fraud_samples": sum(len(dataset) for dataset in self.fraud_data.values()),
            "model_results": all_results,
            "performance_analysis": analysis_summary,
            "model_configurations": {
                model_id: {
                    "name": info['name'],
                    "description": info['description'],
                    "available": 'error' not in info
                }
                for model_id, info in self.gemini_models.items()
            }
        }
        
        # Save results
        output_file = '../data/multi_gemini_fraud_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        self.print_results_summary(comparison_results)
        
        return comparison_results
    
    def analyze_results(self, all_results):
        """Analyze performance across models and prompts"""
        successful_results = [r for r in all_results if 'error' not in r]
        failed_results = [r for r in all_results if 'error' in r]
        
        # Model performance analysis
        model_performance = {}
        for result in successful_results:
            model_name = result['model_name']
            if model_name not in model_performance:
                model_performance[model_name] = {
                    'successful_tests': 0,
                    'total_analysis_length': 0,
                    'total_response_time': 0,
                    'prompt_types': []
                }
            
            model_performance[model_name]['successful_tests'] += 1
            model_performance[model_name]['total_analysis_length'] += result.get('analysis_length', 0)
            model_performance[model_name]['total_response_time'] += result.get('response_time', 0)
            model_performance[model_name]['prompt_types'].append(result['prompt_type'])
        
        # Calculate averages
        for model_name, perf in model_performance.items():
            if perf['successful_tests'] > 0:
                perf['avg_analysis_length'] = perf['total_analysis_length'] / perf['successful_tests']
                perf['avg_response_time'] = perf['total_response_time'] / perf['successful_tests']
                perf['success_rate'] = perf['successful_tests'] / len(self.create_fraud_analysis_prompts()) * 100
        
        # Prompt performance analysis
        prompt_performance = {}
        for result in successful_results:
            prompt_type = result['prompt_type']
            if prompt_type not in prompt_performance:
                prompt_performance[prompt_type] = {
                    'successful_models': 0,
                    'total_analysis_length': 0,
                    'models_tested': []
                }
            
            prompt_performance[prompt_type]['successful_models'] += 1
            prompt_performance[prompt_type]['total_analysis_length'] += result.get('analysis_length', 0)
            prompt_performance[prompt_type]['models_tested'].append(result['model_name'])
        
        return {
            'total_successful': len(successful_results),
            'total_failed': len(failed_results),
            'success_rate': len(successful_results) / len(all_results) * 100 if all_results else 0,
            'model_performance': model_performance,
            'prompt_performance': prompt_performance,
            'best_model': max(model_performance.items(), key=lambda x: x[1]['success_rate'])[0] if model_performance else None,
            'most_responsive_prompt': max(prompt_performance.items(), key=lambda x: x[1]['successful_models'])[0] if prompt_performance else None
        }
    
    def print_results_summary(self, results):
        """Print comprehensive results summary"""
        print(f"\n🎉 MULTI-GEMINI ANALYSIS COMPLETE!")
        print("=" * 40)
        
        analysis = results['performance_analysis']
        
        print(f"📊 Overall Results:")
        print(f"   • Models Tested: {results['total_models_tested']}")
        print(f"   • Prompt Types: {results['total_prompts_tested']}")
        print(f"   • Total Tests: {results['total_combinations']}")
        print(f"   • Success Rate: {analysis['success_rate']:.1f}%")
        print(f"   • Successful Tests: {analysis['total_successful']}")
        print(f"   • Failed Tests: {analysis['total_failed']}")
        
        print(f"\n🏆 Model Performance Rankings:")
        model_perf = analysis['model_performance']
        sorted_models = sorted(model_perf.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        
        for i, (model_name, perf) in enumerate(sorted_models, 1):
            print(f"   {i}. {model_name}")
            print(f"      ✅ Success Rate: {perf['success_rate']:.1f}%")
            print(f"      📝 Avg Analysis Length: {perf['avg_analysis_length']:.0f} chars")
            print(f"      ⚡ Avg Response Time: {perf['avg_response_time']:.2f}s")
        
        print(f"\n🎯 Prompt Performance:")
        prompt_perf = analysis['prompt_performance']
        for prompt_type, perf in prompt_perf.items():
            print(f"   • {prompt_type}: {perf['successful_models']}/{results['total_models_tested']} models succeeded")
        
        if analysis['best_model']:
            print(f"\n🥇 Best Overall Model: {analysis['best_model']}")
        
        print(f"\n💾 Detailed results saved to: ../data/multi_gemini_fraud_comparison.json")

def main():
    """Main function"""
    analyzer = MultiGeminiFraudAnalyzer()
    return asyncio.run(analyzer.run_multi_gemini_comparison())

if __name__ == "__main__":
    main()