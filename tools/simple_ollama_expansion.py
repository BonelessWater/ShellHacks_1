#!/usr/bin/env python3
"""
Simple Multi-Model Ollama Setup
Install a few additional popular models with correct naming
"""

import requests
import json
import time
from datetime import datetime

class SimpleOllamaExpansion:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        
        # Simple, proven models to add
        self.additional_models = [
            'llama3.2:1b',      # Smaller, faster LLaMA
            'phi3:3.8b',        # Microsoft Phi-3
            'gemma2:2b',        # Google Gemma 2  
            'qwen2.5:3b',       # Qwen 2.5 (newer)
        ]
        
        self.load_fraud_data()
    
    def load_fraud_data(self):
        """Load fraud data for testing"""
        try:
            with open('../data/bigquery_fraud_samples.json', 'r') as f:
                data = json.load(f)
                self.fraud_samples = data.get('samples', [])[:2]  # Use 2 samples for faster testing
            print(f"✅ Loaded {len(self.fraud_samples)} fraud samples")
        except FileNotFoundError:
            print("❌ Using sample fraud data")
            self.fraud_samples = [
                {
                    "dataset": "test", "transaction_type": "TRANSFER", 
                    "amount": 181.0, "oldbalanceOrg": 181.0,
                    "newbalanceOrig": 0.0, "is_fraud": 1
                }
            ]
    
    def get_current_models(self):
        """Get currently installed models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except:
            return []
    
    def install_model(self, model_name):
        """Install a single model"""
        print(f"📥 Installing {model_name}...")
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300
            )
            
            if response.status_code == 200:
                print(f"✅ {model_name} installed successfully")
                return True
            else:
                print(f"❌ Failed to install {model_name}")
                return False
        except Exception as e:
            print(f"❌ Error installing {model_name}: {e}")
            return False
    
    def test_model(self, model_name):
        """Test a model with simple fraud analysis"""
        print(f"🧪 Testing {model_name}...")
        
        prompt = f"""
Analyze this transaction for fraud:
- Type: TRANSFER, Amount: $181
- Balance before: $181, Balance after: $0
- Flagged as fraud: Yes

What makes this suspicious?
"""
        
        try:
            start_time = datetime.now()
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 800}
                },
                timeout=120
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', '')
                
                print(f"✅ SUCCESS: {len(analysis):,} chars in {response_time:.1f}s")
                
                return {
                    'model': model_name,
                    'status': 'success', 
                    'analysis_length': len(analysis),
                    'response_time': response_time,
                    'efficiency': len(analysis) / response_time if response_time > 0 else 0,
                    'analysis': analysis
                }
            else:
                print(f"❌ Test failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            return None
    
    def expand_ollama_models(self):
        """Install and test additional models"""
        print("🚀 EXPANDING OLLAMA MODEL COLLECTION")
        print("=" * 45)
        
        # Check current models
        current_models = self.get_current_models()
        print(f"📦 Current models: {current_models}")
        
        # Install additional models
        successful_installs = []
        for model in self.additional_models:
            if model not in current_models:
                print(f"\n📥 Installing {model}...")
                if self.install_model(model):
                    successful_installs.append(model)
                    time.sleep(2)
            else:
                print(f"✅ {model} already installed")
                successful_installs.append(model)
        
        # Test all available models
        all_models = current_models + successful_installs
        all_models = list(set(all_models))  # Remove duplicates
        
        print(f"\n🧪 Testing {len(all_models)} models...")
        
        results = []
        for model in all_models:
            result = self.test_model(model)
            if result:
                results.append(result)
            time.sleep(1)
        
        # Save results
        if results:
            output_file = 'multi_ollama_results.json'
            with open(output_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'models_tested': len(results),
                    'results': results
                }, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
            
            # Show summary
            print(f"\n📊 MULTI-MODEL COMPARISON:")
            print("=" * 35)
            results.sort(key=lambda x: x['analysis_length'], reverse=True)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. 🤖 {result['model']:<15} | {result['analysis_length']:>4,} chars | {result['response_time']:>5.1f}s | {result['efficiency']:>3.0f} c/s")
            
            # Compare to Gemini
            print(f"\n🆚 VS GEMINI COMPARISON:")
            best_local = results[0] if results else None
            if best_local:
                print(f"   🏆 Best Local: {best_local['model']} ({best_local['analysis_length']:,} chars)")
                print(f"   🌟 Best Gemini: 2.0-flash-exp (8,962 chars)")
                print(f"   📊 Local achieves {(best_local['analysis_length']/8962)*100:.1f}% of Gemini's length")
                print(f"   💰 Cost: FREE vs ~$0.02 per analysis")
        
        return results

def main():
    """Main function"""
    expander = SimpleOllamaExpansion()
    
    print("🦙 OLLAMA MODEL EXPANSION")
    print("=" * 30)
    print("Adding more models for comprehensive comparison!")
    print()
    
    results = expander.expand_ollama_models()
    
    if results:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ {len(results)} models ready for comparison")
        print(f"🔄 Run the comparison dashboard to see full results")
    else:
        print(f"\n❌ No models successfully tested")

if __name__ == "__main__":
    main()