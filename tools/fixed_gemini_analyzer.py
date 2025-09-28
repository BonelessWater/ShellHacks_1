#!/usr/bin/env python3
"""
Fixed Multi-Gemini Model Analyzer
Tests multiple Gemini models with correct API configuration
"""

import asyncio
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

class FixedGeminiModelAnalyzer:
    """Compare multiple Gemini models on fraud detection"""
    
    def __init__(self):
        """Initialize the analyzer"""
        load_dotenv('.env')
        self.setup_gemini_models()
        self.fraud_data = None
        
    def setup_gemini_models(self):
        """Setup all available Gemini models with correct configuration"""
        print("🔧 Setting up Gemini models with updated API...")
        
        try:
            import google.generativeai as genai
            
            # Try both API keys
            google_key = os.getenv('GOOGLE_API_KEY')
            google_key_ilan = os.getenv('GOOGLE_API_KEY_ILAN')
            
            if google_key and google_key != 'your_google_api_key_here':
                genai.configure(api_key=google_key)
                print(f"✅ Using primary Google API key: {google_key[:20]}...")
                self.primary_key = google_key
                self.backup_key = google_key_ilan
            elif google_key_ilan:
                genai.configure(api_key=google_key_ilan)
                print(f"✅ Using backup Google API key: {google_key_ilan[:20]}...")
                self.primary_key = google_key_ilan
                self.backup_key = google_key
            else:
                print("❌ No valid Google API keys found")
                self.models = {}
                return
            
            # Current working model names (updated for 2025)
            model_names = [
                'gemini-2.0-flash-exp',      # Latest experimental
                'gemini-1.5-pro',            # Stable pro version
                'gemini-1.5-flash',          # Fast version 
                'gemini-pro',                # Legacy pro
                'gemini-1.5-pro-latest',     # Latest pro
                'gemini-1.5-flash-latest'    # Latest flash
            ]
            
            self.models = {}
            
            for model_name in model_names:
                try:
                    # Test if model is available
                    model = genai.GenerativeModel(model_name)
                    
                    # Quick test to see if it responds
                    test_response = model.generate_content("Hello")
                    if test_response and test_response.text:
                        self.models[model_name] = model
                        print(f"✅ {model_name}: Available")
                    else:
                        print(f"❌ {model_name}: No response")
                        
                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg:
                        print(f"❌ {model_name}: Model not found")
                    elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                        print(f"⚠️ {model_name}: Rate limited")
                    else:
                        print(f"❌ {model_name}: {error_msg[:50]}...")
                
                # Add delay between model tests
                time.sleep(2)
            
            print(f"\n📊 Found {len(self.models)} working Gemini models")
            
        except ImportError:
            print("❌ Google Generative AI library not installed")
            self.models = {}
    
    def load_fraud_data(self):
        """Load BigQuery fraud samples"""
        try:
            with open('../data/bigquery_fraud_samples.json', 'r') as f:
                data = json.load(f)
                self.fraud_data = data['datasets']
                print(f"📊 Loaded fraud data: {sum(len(dataset) for dataset in self.fraud_data.values())} samples")
                return True
        except FileNotFoundError:
            print("❌ Fraud samples file not found")
            return False
    
    def create_fraud_analysis_prompt(self):
        """Create fraud analysis prompt"""
        return """
You are a fraud detection expert analyzing real financial transaction data.

TASK: Analyze the provided fraud samples and identify the TOP 3 FRAUD PATTERNS.

For each pattern, provide:
1. **Pattern Name**
2. **Evidence from data**  
3. **Risk Level** (High/Medium/Low)
4. **How to detect in production**

Focus on actionable insights for a fraud detection system.

**Data includes:**
- IEEE CIS Fraud: Credit card transactions
- PaySim: Mobile money transfers  
- Credit Card: Traditional fraud dataset
- Relational: Multi-table fraud indicators

Be specific and data-driven.
        """
    
    async def test_gemini_model(self, model_name, model, prompt, data_sample):
        """Test a specific Gemini model"""
        try:
            print(f"🧪 Testing {model_name}...")
            
            # Create concise data summary
            data_summary = {
                name: {
                    "records": len(dataset),
                    "fraud": sum(1 for r in dataset if r.get('isFraud', r.get('Class', 0)) == 1),
                    "sample": dataset[0] if dataset else {}
                }
                for name, dataset in data_sample.items()
            }
            
            full_prompt = f"{prompt}\n\nDATA SAMPLE:\n{json.dumps(data_summary, indent=2, default=str)[:2000]}"
            
            # Generate response with timeout
            start_time = time.time()
            response = model.generate_content(full_prompt)
            end_time = time.time()
            
            if response and response.text:
                return {
                    "model": model_name,  
                    "status": "SUCCESS",
                    "analysis": response.text,
                    "analysis_length": len(response.text),
                    "response_time": round(end_time - start_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "model": model_name,
                    "status": "FAILED",
                    "error": "No response text",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            error_msg = str(e)
            return {
                "model": model_name,
                "status": "FAILED", 
                "error": error_msg[:200],
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_multi_gemini_comparison(self):
        """Run comparison across all working Gemini models"""
        print("🚀 MULTI-GEMINI MODEL COMPARISON")
        print("=" * 40)
        
        if not self.models:
            print("❌ No working Gemini models found!")
            return None
        
        # Load fraud data
        if not self.load_fraud_data():
            return None
        
        # Create analysis prompt
        prompt = self.create_fraud_analysis_prompt()
        
        print(f"\n🧪 Testing {len(self.models)} Gemini models...")
        
        # Test each model with delays
        results = []
        for i, (model_name, model) in enumerate(self.models.items()):
            result = await self.test_gemini_model(model_name, model, prompt, self.fraud_data)
            results.append(result)
            
            # Show progress
            if result['status'] == 'SUCCESS':
                print(f"✅ {model_name}: {result['analysis_length']} chars in {result['response_time']}s")
            else:
                print(f"❌ {model_name}: {result['error'][:50]}...")
            
            # Add delay between requests
            if i < len(self.models) - 1:
                print("⏱️ Waiting 5 seconds between models...")
                await asyncio.sleep(5)
        
        # Compile results
        comparison_results = {
            "comparison_timestamp": datetime.now().isoformat(),
            "total_models_tested": len(self.models),
            "successful_models": len([r for r in results if r['status'] == 'SUCCESS']),
            "failed_models": len([r for r in results if r['status'] == 'FAILED']),
            "fraud_samples_analyzed": sum(len(dataset) for dataset in self.fraud_data.values()),
            "model_results": results,
            "analysis_summary": self.create_analysis_summary(results)
        }
        
        # Save results
        output_file = '../data/final_gemini_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        print(f"\n🎉 MULTI-GEMINI COMPARISON COMPLETE!")
        print("=" * 40)
        print(f"📁 Results saved to: {output_file}")
        print(f"📊 Models tested: {len(self.models)}")
        print(f"✅ Successful: {comparison_results['successful_models']}")
        print(f"❌ Failed: {comparison_results['failed_models']}")
        print(f"🎯 Success rate: {comparison_results['successful_models']/len(self.models)*100:.1f}%")
        
        # Show detailed results
        successful_results = [r for r in results if r['status'] == 'SUCCESS']
        if successful_results:
            print(f"\n🏆 SUCCESSFUL MODELS:")
            for result in successful_results:
                print(f"   • {result['model']}: {result['analysis_length']} chars, {result['response_time']}s")
        
        failed_results = [r for r in results if r['status'] == 'FAILED']
        if failed_results:
            print(f"\n❌ FAILED MODELS:")
            for result in failed_results:
                print(f"   • {result['model']}: {result['error'][:60]}...")
        
        return comparison_results
    
    def create_analysis_summary(self, results):
        """Create summary of model comparison"""
        successful = [r for r in results if r['status'] == 'SUCCESS']
        
        if not successful:
            return {
                "best_model": None,
                "average_response_time": 0,
                "average_analysis_length": 0,
                "total_successful": 0
            }
        
        return {
            "best_model": max(successful, key=lambda x: x['analysis_length'])['model'],
            "fastest_model": min(successful, key=lambda x: x['response_time'])['model'],
            "average_response_time": sum(r['response_time'] for r in successful) / len(successful),
            "average_analysis_length": sum(r['analysis_length'] for r in successful) / len(successful),
            "total_successful": len(successful)
        }

def main():
    """Main function"""
    analyzer = FixedGeminiModelAnalyzer()
    return asyncio.run(analyzer.run_multi_gemini_comparison())

if __name__ == "__main__":
    main()