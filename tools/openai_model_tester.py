#!/usr/bin/env python3
"""
OpenAI Model Test Script
Tests the newest OpenAI model (GPT-4o) with fraud detection analysis
"""

import os
import json
import asyncio
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

class OpenAIModelTester:
    """Test OpenAI models for fraud detection"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        load_dotenv()
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in environment variables")
        
        # Check if API key format looks correct (OpenAI keys start with 'sk-')
        if not self.api_key.startswith('sk-'):
            print(f"⚠️ WARNING: OpenAI API key should start with 'sk-', yours starts with '{self.api_key[:6]}...'")
            print("   This might be a Google API key instead of an OpenAI key")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.load_fraud_data()
    
    def load_fraud_data(self):
        """Load fraud data for analysis"""
        try:
            with open('../data/bigquery_fraud_samples.json', 'r') as f:
                data = json.load(f)
                self.fraud_samples = data.get('samples', [])[:5]  # Use first 5 samples for testing
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
    
    async def test_latest_model(self):
        """Test OpenAI models starting with most accessible"""
        print("🧪 TESTING OPENAI MODELS")
        print("=" * 40)
        
        # Try multiple models in order of likely availability
        models_to_test = [
            "gpt-3.5-turbo",     # Most affordable and accessible
            "gpt-4o-mini",       # Smaller GPT-4o version
            "gpt-4o",            # Latest full model
            "gpt-4-turbo",       # GPT-4 Turbo
            "gpt-4"              # Standard GPT-4
        ]
        
        for model_name in models_to_test:
            print(f"\n🚀 Testing model: {model_name}")
            success, result = await self.test_single_model(model_name)
            if success:
                return True, result
            print(f"   ❌ {model_name} failed, trying next model...")
        
        return False, {"error": "All models failed"}
    
    async def test_single_model(self, model_name):
        """Test a single OpenAI model"""
        
        try:
            # Create fraud detection prompt
            prompt = self.create_fraud_prompt()
            
            start_time = datetime.now()
            
            # Test API call
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert fraud detection analyst. Analyze the provided transaction data and identify patterns that indicate fraudulent activity."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=1500,  # Reduced to save quota
                temperature=0.1
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Extract analysis
            analysis = response.choices[0].message.content
            analysis_length = len(analysis)
            
            # Display results
            print(f"✅ SUCCESS: {model_name}")
            print(f"📊 Analysis Length: {analysis_length:,} characters")
            print(f"⚡ Response Time: {response_time:.2f} seconds") 
            print(f"🔍 Token Usage: {response.usage.total_tokens} tokens")
            print(f"💰 Estimated Cost: ${(response.usage.total_tokens * 0.03 / 1000):.4f}")
            
            print(f"\n📝 ANALYSIS PREVIEW:")
            print("-" * 50)
            print(analysis[:500] + "..." if len(analysis) > 500 else analysis)
            print("-" * 50)
            
            # Save results
            result = {
                "model": model_name,
                "status": "success",
                "analysis_length": analysis_length,
                "response_time": response_time,
                "token_usage": response.usage.total_tokens,
                "estimated_cost": response.usage.total_tokens * 0.03 / 1000,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            with open('openai_test_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n💾 Results saved to: openai_test_results.json")
            
            return True, result
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ⚠️ Error: {error_msg}")
            
            # Check for common errors  
            if "401" in error_msg or "authentication" in error_msg.lower():
                print("   🔑 Authentication error - API key issue")
                
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                print("   💳 Quota/billing error - need to add payment method")
                
            elif "model" in error_msg.lower():
                print("   🤖 Model not available or incorrect name")
                
            elif "429" in error_msg:
                print("   📊 Rate limit or quota exceeded")
            
            return False, {"error": error_msg}
    
    def create_fraud_prompt(self):
        """Create fraud detection prompt with sample data"""
        prompt = f"""
Please analyze the following transaction data for fraud patterns:

TRANSACTION DATA:
{json.dumps(self.fraud_samples, indent=2)}

Please provide a comprehensive fraud analysis including:
1. Overall fraud risk assessment
2. Specific patterns that indicate fraud
3. Transaction anomalies and red flags  
4. Recommendations for fraud prevention
5. Confidence level in your assessment

Focus on identifying key fraud indicators such as:
- Balance inconsistencies
- Unusual transaction amounts
- Suspicious transaction types
- Account behavior patterns
"""
        return prompt
    
    async def run_test(self):
        """Run the complete test"""
        print("🔬 OPENAI MODEL TESTING SUITE")
        print("=" * 50)
        print(f"🔑 API Key Status: {'✅ Set' if self.api_key else '❌ Missing'}")
        print(f"📊 Test Data: {len(self.fraud_samples)} fraud samples loaded")
        print()
        
        # Test latest model
        success, result = await self.test_latest_model()
        
        if success:
            print("\n🎉 OPENAI TEST SUCCESSFUL!")
            print("✅ Ready to proceed with comprehensive OpenAI model analysis")
            return True
        else:
            print("\n❌ OPENAI TEST FAILED")
            print("🔧 Please fix the issues above before proceeding")
            return False

async def main():
    """Main test function"""
    tester = OpenAIModelTester()
    return await tester.run_test()

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🚀 Next step: Run comprehensive OpenAI model analyzer")
    else:
        print("\n🛠️ Fix API key issues first, then retry")