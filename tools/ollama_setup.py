#!/usr/bin/env python3
"""
Ollama Local Model Setup and Test
Easiest way to get OpenAI/Claude-like models running locally for free
"""

import json
import requests
import time
from datetime import datetime

class OllamaSetup:
    """Setup and test Ollama local models"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        
    def check_ollama_status(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, None
        except:
            return False, None
    
    def install_model(self, model_name):
        """Install/pull a model"""
        print(f"📥 Installing {model_name}...")
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
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
    
    def test_model_fraud_analysis(self, model_name):
        """Test a model with fraud detection"""
        print(f"🧪 Testing {model_name} for fraud analysis...")
        
        # Load sample fraud data
        fraud_prompt = """
You are a fraud detection expert. Analyze this transaction:

Transaction Details:
- Type: TRANSFER
- Amount: $181.00
- Account balance before: $181.00
- Account balance after: $0.00
- Status: Flagged as fraud

Questions:
1. What makes this transaction suspicious?
2. What fraud patterns do you see?
3. What would you recommend to prevent this?

Please provide a detailed analysis.
"""
        
        try:
            start_time = datetime.now()
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": fraud_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1000
                    }
                },
                timeout=120
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', '')
                
                print(f"✅ SUCCESS!")
                print(f"📊 Analysis length: {len(analysis):,} characters")
                print(f"⚡ Response time: {response_time:.2f} seconds")
                print(f"\n📝 ANALYSIS PREVIEW:")
                print("-" * 50)
                print(analysis[:300] + "..." if len(analysis) > 300 else analysis)
                print("-" * 50)
                
                return {
                    'model': model_name,
                    'status': 'success',
                    'analysis_length': len(analysis),
                    'response_time': response_time,
                    'analysis': analysis
                }
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            return None
    
    def run_complete_setup(self):
        """Complete Ollama setup and testing"""
        print("🦙 OLLAMA LOCAL MODEL SETUP")
        print("=" * 40)
        
        # Check if Ollama is running
        print("1️⃣ Checking Ollama status...")
        is_running, models_data = self.check_ollama_status()
        
        if not is_running:
            print("❌ Ollama is not running!")
            print("\n🛠️ SETUP INSTRUCTIONS:")
            print("1. Download Ollama from: https://ollama.ai")
            print("2. Install and start Ollama")
            print("3. Open terminal/PowerShell and run: ollama --version")
            print("4. Then run this script again")
            return False
        
        print("✅ Ollama is running!")
        
        # List current models
        current_models = [m['name'] for m in models_data.get('models', [])]
        print(f"📋 Current models: {current_models}")
        
        # Recommended models for fraud analysis
        recommended_models = [
            'llama3.2:3b',      # Fast, good quality
            'mistral:7b',       # Excellent for analysis
            'phi3:mini',        # Microsoft model, fast
        ]
        
        print(f"\n2️⃣ Installing recommended models...")
        installed_models = []
        
        for model in recommended_models:
            if model not in current_models:
                print(f"\n📥 Installing {model} (this may take a few minutes)...")
                if self.install_model(model):
                    installed_models.append(model)
                    time.sleep(2)  # Brief pause between installations
            else:
                print(f"✅ {model} already installed")
                installed_models.append(model)
        
        print(f"\n3️⃣ Testing models for fraud analysis...")
        test_results = []
        
        for model in installed_models:
            print(f"\n🧪 Testing {model}...")
            result = self.test_model_fraud_analysis(model)
            if result:
                test_results.append(result)
        
        # Save results
        if test_results:
            output_file = 'ollama_fraud_analysis_results.json'
            with open(output_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_models': len(test_results),
                    'results': test_results
                }, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
            
            # Summary comparison with Gemini
            print(f"\n🏆 OLLAMA VS GEMINI COMPARISON:")
            print("=" * 40)
            for result in test_results:
                print(f"🤖 {result['model']}:")
                print(f"   📊 Length: {result['analysis_length']:,} chars")
                print(f"   ⚡ Speed: {result['response_time']:.2f}s")
                print(f"   💰 Cost: FREE (local)")
                print()
            
            print("📈 Compare these results with your Gemini analysis!")
            return True
        else:
            print("❌ No models successfully tested")
            return False

def main():
    """Main setup function"""
    setup = OllamaSetup()
    
    print("🚀 EASY LOCAL LLM SETUP FOR GEMINI COMPARISON")
    print("=" * 55)
    print("This will set up FREE local models to compare with Gemini!")
    print()
    
    success = setup.run_complete_setup()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("✅ You now have local models to compare with Gemini")
        print("🔄 Run the comprehensive analyzer to compare all models")
    else:
        print("\n🛠️ Setup needed - follow instructions above")

if __name__ == "__main__":
    main()