#!/usr/bin/env python3
"""
Final Working Gemini Model Comparison
Uses correct model names and tests available Gemini variants
"""

import asyncio
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

class FinalGeminiComparison:
    """Test all available Gemini models with correct names"""
    
    def __init__(self):
        load_dotenv('.env')
        self.setup_gemini()
        
    def setup_gemini(self):
        """Setup Gemini with correct model names"""
        try:
            import google.generativeai as genai
            google_key = os.getenv('GOOGLE_API_KEY')
            
            if google_key:
                genai.configure(api_key=google_key)
                
                # Correct model names based on current API
                self.models_to_test = {
                    'gemini-2.0-flash-exp': 'Gemini 2.0 Flash Experimental',
                    'gemini-1.5-pro-latest': 'Gemini 1.5 Pro Latest',
                    'gemini-1.5-flash-latest': 'Gemini 1.5 Flash Latest',
                    'gemini-1.5-flash-8b-latest': 'Gemini 1.5 Flash 8B Latest',
                    'gemini-pro': 'Gemini Pro (Legacy)',
                    'gemini-pro-vision': 'Gemini Pro Vision'
                }
                
                print("🔧 Testing available Gemini models...")
                self.working_models = {}
                
                for model_id, model_name in self.models_to_test.items():
                    try:
                        model = genai.GenerativeModel(model_id)
                        # Quick test
                        test_response = model.generate_content("Hello")
                        if test_response.text:
                            self.working_models[model_id] = {
                                'name': model_name,
                                'model': model,
                                'tested': True
                            }
                            print(f"✅ {model_name} - Working")
                        else:
                            print(f"⚠️ {model_name} - Empty response")
                    except Exception as e:
                        error = str(e)
                        if "404" in error:
                            print(f"❌ {model_name} - Not available")
                        elif "429" in error:
                            print(f"⏳ {model_name} - Rate limited")
                        else:
                            print(f"❌ {model_name} - Error: {error[:50]}...")
                
                print(f"\n✅ Found {len(self.working_models)} working models")
                
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            self.working_models = {}
    
    def load_fraud_data(self):
        """Load fraud data"""
        try:
            with open('../data/bigquery_fraud_samples.json', 'r') as f:
                data = json.load(f)
                return data['datasets']
        except FileNotFoundError:
            print("❌ No fraud data found")
            return None
    
    def test_model_on_fraud(self, model_id, model_info, fraud_data):
        """Test a single model on fraud detection"""
        prompt = """
Analyze this fraud data and identify the TOP 3 fraud patterns:

1. **Pattern Name**: Clear name
2. **Evidence**: Supporting data
3. **Risk Level**: High/Medium/Low

Be concise and specific.
        """
        
        # Create simple data summary
        sample_data = {
            "total_samples": sum(len(dataset) for dataset in fraud_data.values()),
            "fraud_examples": []
        }
        
        # Get 2 fraud examples
        for dataset in fraud_data.values():
            fraud_cases = [r for r in dataset if r.get('isFraud', r.get('Class', 0)) == 1][:2]
            sample_data["fraud_examples"].extend(fraud_cases)
        
        full_prompt = f"{prompt}\n\nData: {json.dumps(sample_data, default=str)[:1500]}"
        
        try:
            start_time = time.time()
            response = model_info['model'].generate_content(full_prompt)
            response_time = time.time() - start_time
            
            return {
                'model_id': model_id,
                'model_name': model_info['name'],
                'success': True,
                'analysis': response.text,
                'analysis_length': len(response.text),
                'response_time': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'model_id': model_id,
                'model_name': model_info['name'],
                'success': False,
                'error': str(e)[:200],
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_comparison(self):
        """Run the comparison"""
        print("🚀 FINAL GEMINI MODEL COMPARISON")
        print("=" * 35)
        
        fraud_data = self.load_fraud_data()
        if not fraud_data:
            return None
        
        if not self.working_models:
            print("❌ No working models found")
            return None
        
        print(f"🧪 Testing {len(self.working_models)} models on fraud detection...")
        
        results = []
        for i, (model_id, model_info) in enumerate(self.working_models.items()):
            print(f"🔄 [{i+1}/{len(self.working_models)}] Testing {model_info['name']}...")
            
            result = self.test_model_on_fraud(model_id, model_info, fraud_data)
            results.append(result)
            
            if result['success']:
                print(f"   ✅ Success: {result['analysis_length']} chars in {result['response_time']:.2f}s")
                print(f"   📝 Preview: {result['analysis'][:100]}...")
            else:
                print(f"   ❌ Failed: {result['error']}")
            
            # Wait between requests
            if i < len(self.working_models) - 1:
                print("   ⏳ Waiting 15 seconds...")
                time.sleep(15)
        
        # Analyze results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'total_models_tested': len(results),
            'successful_models': len(successful),
            'failed_models': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'results': results,
            'fraud_data_samples': sum(len(dataset) for dataset in fraud_data.values())
        }
        
        # Save results
        with open('../data/final_gemini_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # Print summary
        print(f"\n🎉 COMPARISON COMPLETE!")
        print("=" * 25)
        print(f"📊 Results:")
        print(f"   • Total Models: {len(results)}")
        print(f"   • Successful: {len(successful)}")
        print(f"   • Failed: {len(failed)}")
        print(f"   • Success Rate: {comparison_results['success_rate']:.1f}%")
        
        if successful:
            print(f"\n🏆 Working Models:")
            for result in successful:
                print(f"   ✅ {result['model_name']}")
                print(f"      📝 {result['analysis_length']} characters")
                print(f"      ⚡ {result['response_time']:.2f} seconds")
        
        if failed:
            print(f"\n❌ Failed Models:")
            for result in failed:
                print(f"   • {result['model_name']}: {result['error'][:100]}...")
        
        print(f"\n💾 Results saved to: ../data/final_gemini_comparison.json")
        
        return comparison_results

def main():
    """Main function"""
    analyzer = FinalGeminiComparison()
    return asyncio.run(analyzer.run_comparison())

if __name__ == "__main__":
    main()