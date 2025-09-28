#!/usr/bin/env python3
"""
Comprehensive Ollama Model Setup
Installs and tests multiple Ollama models for fraud detection comparison
"""

import json
import requests
import time
from datetime import datetime
import asyncio
import aiohttp

class ComprehensiveOllamaSetup:
    """Setup and test multiple Ollama models"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        
        # Comprehensive list of high-quality models to test
        self.recommended_models = {
            # LLaMA Family (Meta)
            'llama3.2:3b': {'size': '2GB', 'speed': 'Fast', 'quality': 'Good', 'description': 'Latest LLaMA 3.2, 3B params'},
            'llama3.2:1b': {'size': '1.3GB', 'speed': 'Very Fast', 'quality': 'Fair', 'description': 'Smallest LLaMA 3.2, fastest'},
            'llama3.1:8b': {'size': '4.7GB', 'speed': 'Medium', 'quality': 'Excellent', 'description': 'LLaMA 3.1, 8B params, high quality'},
            'llama3:8b': {'size': '4.7GB', 'speed': 'Medium', 'quality': 'Excellent', 'description': 'Original LLaMA 3, proven performance'},
            
            # Mistral Family (Mistral AI)
            'mistral:7b': {'size': '4.1GB', 'speed': 'Medium', 'quality': 'Excellent', 'description': 'Mistral 7B, great for analysis'},
            'mistral-nemo:12b': {'size': '7GB', 'speed': 'Slow', 'quality': 'Excellent', 'description': 'Larger Mistral model'},
            'mistral:instruct': {'size': '4.1GB', 'speed': 'Medium', 'quality': 'Excellent', 'description': 'Mistral fine-tuned for instructions'},
            
            # Specialized Models
            'phi3:mini': {'size': '2.3GB', 'speed': 'Fast', 'quality': 'Good', 'description': 'Microsoft Phi-3, optimized'},
            'phi3:medium': {'size': '7.9GB', 'speed': 'Medium', 'quality': 'Excellent', 'description': 'Larger Phi-3 model'},
            'gemma2:9b': {'size': '5.4GB', 'speed': 'Medium', 'quality': 'Excellent', 'description': 'Google Gemma 2, latest'},
            'gemma2:2b': {'size': '1.6GB', 'speed': 'Fast', 'quality': 'Good', 'description': 'Smaller Gemma 2'},
            
            # Code-focused (good for technical analysis)
            'codellama:7b': {'size': '3.8GB', 'speed': 'Medium', 'quality': 'Good', 'description': 'Code-focused LLaMA'},
            'codegemma:7b': {'size': '5GB', 'speed': 'Medium', 'quality': 'Good', 'description': 'Code-focused Gemma'},
            
            # Fast/Efficient Models
            'qwen2:7b': {'size': '4.4GB', 'speed': 'Fast', 'quality': 'Excellent', 'description': 'Alibaba Qwen 2, very efficient'},
            'qwen2:1.5b': {'size': '934MB', 'speed': 'Very Fast', 'quality': 'Fair', 'description': 'Tiny but capable Qwen'},
            
            # Specialized Analysis Models
            'neural-chat:7b': {'size': '4.1GB', 'speed': 'Medium', 'quality': 'Good', 'description': 'Intel neural chat model'},
            'orca-mini:3b': {'size': '1.9GB', 'speed': 'Fast', 'quality': 'Good', 'description': 'Microsoft Orca, instruction-tuned'},
        }
        
        self.load_fraud_data()
    
    def load_fraud_data(self):
        """Load fraud data for analysis"""
        try:
            with open('../data/bigquery_fraud_samples.json', 'r') as f:
                data = json.load(f)
                self.fraud_samples = data.get('samples', [])[:3]  # Use 3 samples for testing
            print(f"✅ Loaded {len(self.fraud_samples)} fraud samples for testing")
        except FileNotFoundError:
            print("❌ Fraud data not found, using sample data")
            self.fraud_samples = [
                {
                    "dataset": "test",
                    "transaction_type": "TRANSFER", 
                    "amount": 181.0,
                    "oldbalanceOrg": 181.0,
                    "newbalanceOrig": 0.0,
                    "is_fraud": 1
                }
            ]
    
    def check_ollama_status(self):
        """Check if Ollama is running and get current models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, None
        except:
            return False, None
    
    def install_model(self, model_name):
        """Install/pull a model with progress tracking"""
        print(f"📥 Installing {model_name}...")
        model_info = self.recommended_models.get(model_name, {})
        if model_info:
            print(f"   📊 Size: {model_info['size']}, Speed: {model_info['speed']}, Quality: {model_info['quality']}")
            print(f"   💡 {model_info['description']}")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600  # 10 minutes timeout for large models
            )
            
            if response.status_code == 200:
                print(f"✅ {model_name} installed successfully")
                return True
            else:
                print(f"❌ Failed to install {model_name}: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error installing {model_name}: {e}")
            return False
    
    def test_model_fraud_analysis(self, model_name):
        """Test a model with fraud detection"""
        print(f"🧪 Testing {model_name} for fraud analysis...")
        
        # Create comprehensive fraud prompt
        fraud_prompt = f"""
You are an expert financial fraud analyst. Analyze these transactions for fraud patterns:

TRANSACTIONS DATA:
{json.dumps(self.fraud_samples, indent=2)}

Please provide:
1. Overall fraud risk assessment
2. Specific suspicious patterns identified
3. Key red flags and anomalies
4. Recommendations for prevention
5. Confidence level in your analysis

Focus on patterns like:
- Balance inconsistencies
- Unusual amounts
- Transaction type anomalies
- Account behavior patterns

Provide a detailed professional analysis.
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
                        "num_predict": 1200,  # Allow longer responses
                        "top_p": 0.9,
                        "top_k": 40
                    }
                },
                timeout=180  # 3 minutes timeout
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', '')
                
                # Calculate efficiency metrics
                chars_per_second = len(analysis) / response_time if response_time > 0 else 0
                
                print(f"✅ SUCCESS!")
                print(f"   📊 Analysis length: {len(analysis):,} characters")
                print(f"   ⚡ Response time: {response_time:.2f} seconds")
                print(f"   🚀 Efficiency: {chars_per_second:.0f} chars/second")
                
                # Show preview
                print(f"   📝 Preview: {analysis[:150]}...")
                
                return {
                    'model': model_name,
                    'status': 'success',
                    'analysis_length': len(analysis),
                    'response_time': response_time,
                    'efficiency': chars_per_second,
                    'analysis': analysis,
                    'model_info': self.recommended_models.get(model_name, {})
                }
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            return None
    
    def install_recommended_models(self, selection='fast'):
        """Install a selection of recommended models"""
        print(f"🚀 COMPREHENSIVE OLLAMA MODEL INSTALLATION")
        print("=" * 55)
        
        # Define model selections
        selections = {
            'fast': ['llama3.2:1b', 'qwen2:1.5b', 'gemma2:2b', 'phi3:mini'],
            'balanced': ['llama3.2:3b', 'mistral:7b', 'phi3:mini', 'gemma2:2b', 'qwen2:7b'],
            'quality': ['llama3.1:8b', 'mistral:7b', 'phi3:medium', 'gemma2:9b', 'qwen2:7b'],
            'comprehensive': ['llama3.2:1b', 'llama3.2:3b', 'llama3.1:8b', 'mistral:7b', 
                            'phi3:mini', 'phi3:medium', 'gemma2:2b', 'gemma2:9b', 'qwen2:1.5b', 'qwen2:7b']
        }
        
        models_to_install = selections.get(selection, selections['balanced'])
        
        print(f"📋 Installing {selection} selection: {len(models_to_install)} models")
        print("Models to install:")
        for model in models_to_install:
            info = self.recommended_models.get(model, {})
            print(f"  • {model} - {info.get('description', 'No description')}")
        
        print()
        
        # Check current status
        is_running, current_data = self.check_ollama_status()
        if not is_running:
            print("❌ Ollama is not running! Please start Ollama first.")
            return []
        
        current_models = [m['name'] for m in current_data.get('models', [])]
        print(f"📦 Currently installed: {current_models}")
        
        # Install models
        installed_models = []
        total_models = len(models_to_install)
        
        for i, model in enumerate(models_to_install, 1):
            print(f"\n🔄 Installing model {i}/{total_models}: {model}")
            
            if model in current_models:
                print(f"✅ {model} already installed, skipping...")
                installed_models.append(model)
            else:
                if self.install_model(model):
                    installed_models.append(model)
                    time.sleep(1)  # Brief pause between installations
                else:
                    print(f"⚠️ Skipping {model} due to installation failure")
        
        print(f"\n✅ Installation complete! {len(installed_models)}/{total_models} models ready")
        return installed_models
    
    def run_comprehensive_fraud_analysis(self, models_to_test=None):
        """Run fraud analysis on multiple models"""
        print(f"\n🧪 COMPREHENSIVE FRAUD ANALYSIS")
        print("=" * 40)
        
        if models_to_test is None:
            # Get all available models
            is_running, current_data = self.check_ollama_status()
            if not is_running:
                print("❌ Ollama is not running!")
                return []
            
            models_to_test = [m['name'] for m in current_data.get('models', [])]
        
        print(f"🎯 Testing {len(models_to_test)} models...")
        
        results = []
        total_models = len(models_to_test)
        
        for i, model in enumerate(models_to_test, 1):
            print(f"\n🧪 Testing model {i}/{total_models}: {model}")
            result = self.test_model_fraud_analysis(model)
            if result:
                results.append(result)
            time.sleep(1)  # Brief pause between tests
        
        # Save comprehensive results
        if results:
            output_file = 'comprehensive_ollama_analysis.json'
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'total_models_tested': len(results),
                'models_info': self.recommended_models,
                'results': results
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
            
            # Show summary
            self.print_analysis_summary(results)
        
        return results
    
    def print_analysis_summary(self, results):
        """Print comprehensive analysis summary"""
        print(f"\n📊 COMPREHENSIVE OLLAMA ANALYSIS SUMMARY")
        print("=" * 55)
        
        if not results:
            print("❌ No successful analyses")
            return
        
        # Sort by different metrics
        by_length = sorted(results, key=lambda x: x['analysis_length'], reverse=True)
        by_speed = sorted(results, key=lambda x: x['response_time'])
        by_efficiency = sorted(results, key=lambda x: x['efficiency'], reverse=True)
        
        print(f"🏆 BEST PERFORMERS:")
        print(f"   📊 Most Comprehensive: {by_length[0]['model']} ({by_length[0]['analysis_length']:,} chars)")
        print(f"   ⚡ Fastest: {by_speed[0]['model']} ({by_speed[0]['response_time']:.2f}s)")
        print(f"   🚀 Most Efficient: {by_efficiency[0]['model']} ({by_efficiency[0]['efficiency']:.0f} chars/s)")
        
        print(f"\n📈 STATISTICS:")
        avg_length = sum(r['analysis_length'] for r in results) / len(results)
        avg_time = sum(r['response_time'] for r in results) / len(results)
        avg_efficiency = sum(r['efficiency'] for r in results) / len(results)
        
        print(f"   📊 Average analysis length: {avg_length:,.0f} characters")
        print(f"   ⚡ Average response time: {avg_time:.2f} seconds")
        print(f"   🚀 Average efficiency: {avg_efficiency:.0f} chars/second")
        
        print(f"\n🎯 MODEL COMPARISON:")
        for result in by_length:
            print(f"   🤖 {result['model']:<20} | {result['analysis_length']:>6,} chars | {result['response_time']:>6.1f}s | {result['efficiency']:>4.0f} c/s")
    
    def run_full_setup(self, selection='balanced'):
        """Run complete setup and analysis"""
        print("🦙 COMPREHENSIVE OLLAMA SETUP & ANALYSIS")
        print("=" * 55)
        print("This will install multiple high-quality models and test them all!")
        print()
        
        # Install models
        installed_models = self.install_recommended_models(selection)
        
        if not installed_models:
            print("❌ No models installed successfully")
            return False
        
        # Run comprehensive analysis
        results = self.run_comprehensive_fraud_analysis(installed_models)
        
        if results:
            print(f"\n🎉 SUCCESS!")
            print(f"✅ {len(results)} models successfully analyzed fraud data")
            print(f"🔄 Run the comparison dashboard to see results vs Gemini")
            return True
        else:
            print(f"\n❌ No successful analyses")
            return False

def main():
    """Main function"""
    setup = ComprehensiveOllamaSetup()
    
    print("🚀 COMPREHENSIVE OLLAMA MODEL SETUP")
    print("=" * 45)
    print("Choose your model selection:")
    print("1. 🏃 Fast (4 small/fast models) - ~5GB total")
    print("2. ⚖️ Balanced (5 mixed models) - ~15GB total") 
    print("3. 🏆 Quality (5 large models) - ~25GB total")
    print("4. 🌟 Comprehensive (10 models) - ~35GB total")
    print()
    
    selections = {
        '1': 'fast',
        '2': 'balanced', 
        '3': 'quality',
        '4': 'comprehensive'
    }
    
    # For automation, use balanced selection
    choice = 'balanced'  # You can change this
    
    print(f"🎯 Running {choice} selection...")
    success = setup.run_full_setup(choice)
    
    if success:
        print(f"\n🎉 SETUP COMPLETE!")
        print(f"🔄 Now run: python gemini_vs_local_comparator.py")
    else:
        print(f"\n🛠️ Setup had issues - check Ollama installation")

if __name__ == "__main__":
    main()