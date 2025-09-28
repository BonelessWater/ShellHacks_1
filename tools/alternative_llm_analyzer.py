#!/usr/bin/env python3
"""
Alternative LLM Model Analyzer
Uses free/local alternatives to OpenAI and Claude for fraud detection comparison
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime
from dotenv import load_dotenv

class AlternativeLLMAnalyzer:
    """Test alternative LLM models for fraud detection comparison"""
    
    def __init__(self):
        """Initialize alternative LLM clients"""
        load_dotenv()
        self.load_fraud_data()
        
        # API endpoints for free alternatives
        self.endpoints = {
            'huggingface': 'https://api-inference.huggingface.co/models/',
            'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
            'together': 'https://api.together.xyz/v1/chat/completions'
        }
        
        # Free model options
        self.free_models = {
            'huggingface_models': [
                'microsoft/DialoGPT-large',
                'facebook/blenderbot-400M-distill',
                'microsoft/DialoGPT-medium'
            ],
            'openrouter_free': [
                'mistralai/mistral-7b-instruct:free',
                'huggingfaceh4/zephyr-7b-beta:free',
                'openchat/openchat-7b:free'
            ],
            'together_free': [
                'mistralai/Mistral-7B-Instruct-v0.1',
                'togethercomputer/RedPajama-INCITE-7B-Chat',
                'NousResearch/Nous-Hermes-Llama2-13b'
            ]
        }
    
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
    
    async def test_huggingface_models(self):
        """Test Hugging Face Inference API models (free tier)"""
        print("🤗 TESTING HUGGING FACE MODELS")
        print("=" * 40)
        
        results = []
        
        for model_name in self.free_models['huggingface_models']:
            try:
                print(f"🚀 Testing: {model_name}")
                
                # Create simplified prompt for smaller models
                prompt = self.create_simple_fraud_prompt()
                
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'Authorization': f'Bearer {os.getenv("HUGGINGFACE_API_KEY", "hf_dummy")}',
                        'Content-Type': 'application/json'
                    }
                    
                    payload = {
                        'inputs': prompt,
                        'parameters': {
                            'max_new_tokens': 500,
                            'temperature': 0.1
                        }
                    }
                    
                    start_time = datetime.now()
                    
                    async with session.post(
                        f"{self.endpoints['huggingface']}{model_name}",
                        headers=headers,
                        json=payload
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            end_time = datetime.now()
                            response_time = (end_time - start_time).total_seconds()
                            
                            if isinstance(result, list) and len(result) > 0:
                                analysis = result[0].get('generated_text', '')
                                analysis_length = len(analysis)
                                
                                print(f"✅ SUCCESS: {analysis_length:,} chars in {response_time:.2f}s")
                                
                                results.append({
                                    'model': model_name,
                                    'provider': 'huggingface',
                                    'status': 'success',
                                    'analysis_length': analysis_length,
                                    'response_time': response_time,
                                    'analysis': analysis
                                })
                            else:
                                print(f"❌ Unexpected response format")
                        else:
                            print(f"❌ HTTP {response.status}: {await response.text()}")
                            
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                
        return results
    
    async def test_local_ollama_models(self):
        """Test local Ollama models if available"""
        print("🦙 TESTING LOCAL OLLAMA MODELS")
        print("=" * 40)
        
        results = []
        
        # Check if Ollama is running
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:11434/api/tags') as response:
                    if response.status == 200:
                        models_data = await response.json()
                        available_models = [m['name'] for m in models_data.get('models', [])]
                        
                        print(f"📋 Found {len(available_models)} local models: {available_models}")
                        
                        # Test available models
                        for model_name in available_models[:3]:  # Test first 3
                            result = await self.test_ollama_model(model_name)
                            if result:
                                results.append(result)
                    else:
                        print("❌ Ollama not responding")
                        
        except Exception as e:
            print(f"❌ Ollama not available: {str(e)}")
            print("💡 Install Ollama from https://ollama.ai to test local models")
            
        return results
    
    async def test_ollama_model(self, model_name):
        """Test a specific Ollama model"""
        try:
            print(f"🚀 Testing Ollama: {model_name}")
            
            prompt = self.create_fraud_prompt()
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    'model': model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 800
                    }
                }
                
                start_time = datetime.now()
                
                async with session.post(
                    'http://localhost:11434/api/generate',
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        end_time = datetime.now()
                        response_time = (end_time - start_time).total_seconds()
                        
                        analysis = result.get('response', '')
                        analysis_length = len(analysis)
                        
                        print(f"✅ SUCCESS: {analysis_length:,} chars in {response_time:.2f}s")
                        
                        return {
                            'model': model_name,
                            'provider': 'ollama_local',
                            'status': 'success',
                            'analysis_length': analysis_length,
                            'response_time': response_time,
                            'analysis': analysis
                        }
                    else:
                        print(f"❌ HTTP {response.status}")
                        return None
                        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def create_simple_fraud_prompt(self):
        """Create simplified prompt for smaller models"""
        return f"""
Analyze this transaction for fraud:
Transaction: {json.dumps(self.fraud_samples[0], indent=2) if self.fraud_samples else 'No data'}

Is this fraudulent? Why?
"""
    
    def create_fraud_prompt(self):
        """Create comprehensive fraud detection prompt"""
        return f"""
You are a fraud detection expert. Analyze these transactions:

{json.dumps(self.fraud_samples, indent=2)}

Identify fraud patterns and provide recommendations.
"""
    
    async def run_comprehensive_test(self):
        """Run tests on all available alternative models"""
        print("🔬 ALTERNATIVE LLM TESTING SUITE")
        print("=" * 50)
        print("Testing free/local alternatives to OpenAI and Claude")
        print()
        
        all_results = []
        
        # Test Hugging Face models
        hf_results = await self.test_huggingface_models()
        all_results.extend(hf_results)
        
        print()
        
        # Test local Ollama models
        ollama_results = await self.test_local_ollama_models()
        all_results.extend(ollama_results)
        
        # Save results
        if all_results:
            output_file = 'alternative_llm_results.json'
            with open(output_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_models_tested': len(all_results),
                    'results': all_results
                }, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
            
            # Summary
            print(f"\n📊 SUMMARY:")
            print(f"   • Total models tested: {len(all_results)}")
            print(f"   • Successful analyses: {len([r for r in all_results if r['status'] == 'success'])}")
            
            if all_results:
                avg_length = sum(r['analysis_length'] for r in all_results) / len(all_results)
                avg_time = sum(r['response_time'] for r in all_results) / len(all_results)
                print(f"   • Average analysis length: {avg_length:,.0f} characters")
                print(f"   • Average response time: {avg_time:.2f} seconds")
        
        return all_results

def setup_instructions():
    """Print setup instructions for alternative models"""
    print("🛠️ SETUP INSTRUCTIONS FOR ALTERNATIVE MODELS")
    print("=" * 55)
    print()
    
    print("1️⃣ **HUGGING FACE (Free Tier)**:")
    print("   • Go to: https://huggingface.co/settings/tokens")
    print("   • Create a free account")
    print("   • Generate an API token")
    print("   • Add to .env: HUGGINGFACE_API_KEY=your_token")
    print()
    
    print("2️⃣ **OLLAMA (Local Models)**:")
    print("   • Download: https://ollama.ai")
    print("   • Install and run: ollama run llama2")
    print("   • Or try: ollama run mistral")
    print("   • Models run locally (no API key needed)")
    print()
    
    print("3️⃣ **OPENROUTER (Pay-per-use with free credits)**:")
    print("   • Go to: https://openrouter.ai")
    print("   • Sign up for free credits")
    print("   • Add to .env: OPENROUTER_API_KEY=your_key")
    print()
    
    print("4️⃣ **TOGETHER AI (Free tier)**:")
    print("   • Go to: https://api.together.xyz")
    print("   • Sign up for free tier")
    print("   • Add to .env: TOGETHER_API_KEY=your_key")
    print()
    
    print("🎯 **RECOMMENDATION**: Start with Ollama for immediate testing!")

async def main():
    """Main function"""
    analyzer = AlternativeLLMAnalyzer()
    
    print("🆚 ALTERNATIVE LLM COMPARISON TO GEMINI")
    print("=" * 55)
    print()
    
    # Show setup instructions
    setup_instructions()
    print()
    
    # Ask user what to test
    print("What would you like to test?")
    print("1. Run available tests now")
    print("2. Show detailed setup instructions")
    
    # For now, run available tests
    results = await analyzer.run_comprehensive_test()
    
    if results:
        print("\n🎉 SUCCESS! Alternative models tested")
        print("💡 You can now compare these results with your Gemini analysis")
    else:
        print("\n📋 No models available currently")
        print("🛠️ Follow setup instructions above to add more models")

if __name__ == "__main__":
    asyncio.run(main())