#!/usr/bin/env python3
"""
Anthropic Claude Model Test Script
Tests the newest Claude model (Claude 3.5 Sonnet) with fraud detection analysis
"""

import os
import json
import asyncio
from datetime import datetime
import anthropic
from dotenv import load_dotenv

class ClaudeModelTester:
    """Test Anthropic Claude models for fraud detection"""
    
    def __init__(self):
        """Initialize Anthropic client"""
        load_dotenv()
        
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("❌ ANTHROPIC_API_KEY not found in environment variables")
        
        # Check if API key format looks correct (Anthropic keys start with 'sk-ant-')
        print(f"🔑 API Key format: {self.api_key[:15]}... (length: {len(self.api_key)})")
        if not self.api_key.startswith('sk-ant-'):
            print(f"⚠️ WARNING: Anthropic API key should start with 'sk-ant-', yours starts with '{self.api_key[:10]}...'")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
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
    
    async def test_latest_claude_model(self):
        """Test the latest Claude model (Claude 3.5 Sonnet)"""
        print("🧪 TESTING LATEST CLAUDE MODEL")
        print("=" * 40)
        
        # Try multiple Claude model versions to find one that works
        models_to_test = [
            "claude-3-5-sonnet-latest",      # Latest alias
            "claude-3-5-sonnet",             # Generic latest
            "claude-3-sonnet-20240229",      # Stable Claude 3 Sonnet
            "claude-3-haiku-20240307"        # Faster/cheaper option
        ]
        
        for model_name in models_to_test:
            print(f"🚀 Testing model: {model_name}")
            success, result = await self.test_single_model(model_name)
            if success:
                return True, result
            print(f"   ❌ {model_name} failed, trying next model...")
        
        return False, {"error": "All models failed"}
    
    async def test_single_model(self, model_name):
        """Test a single Claude model"""
        
        try:
            # Create fraud detection prompt
            prompt = self.create_fraud_prompt()
            
            start_time = datetime.now()
            
            # Test API call
            message = self.client.messages.create(
                model=model_name,
                max_tokens=1500,  # Reduced to save credits
                temperature=0.1,
                system="You are an expert fraud detection analyst with years of experience identifying financial fraud patterns. Analyze the provided transaction data and provide comprehensive insights about potential fraudulent activity.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Extract analysis
            analysis = message.content[0].text
            analysis_length = len(analysis)
            
            # Display results
            print(f"✅ SUCCESS: {model_name}")
            print(f"📊 Analysis Length: {analysis_length:,} characters")
            print(f"⚡ Response Time: {response_time:.2f} seconds") 
            print(f"🔍 Token Usage: Input: {message.usage.input_tokens}, Output: {message.usage.output_tokens}")
            print(f"💰 Estimated Cost: ${self.calculate_cost(message.usage.input_tokens, message.usage.output_tokens):.4f}")
            
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
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
                "estimated_cost": self.calculate_cost(message.usage.input_tokens, message.usage.output_tokens),
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            with open('claude_test_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n💾 Results saved to: claude_test_results.json")
            
            return True, result
            
        except Exception as e:
            print(f"❌ ERROR testing {model_name}: {str(e)}")
            
            # Check for common errors
            if "401" in str(e) or "authentication" in str(e).lower():
                print("🔑 This looks like an authentication error.")
                print("   Please check that your Anthropic API key is correct and active.")
                print(f"   Your key starts with: {self.api_key[:10]}...")
                
            elif "400" in str(e) and "credit" in str(e).lower():
                print("💳 This looks like a credits/billing error.")
                print("   Please check your Anthropic account credits and billing.")
                
            elif "model" in str(e).lower():
                print("🤖 This looks like a model availability error.")
                print("   The model might not be available or the name might be incorrect.")
                
            elif "rate" in str(e).lower():
                print("📊 This looks like a rate limiting error.")
                print("   Please wait a moment and try again.")
            
            return False, {"error": str(e)}
    
    def calculate_cost(self, input_tokens, output_tokens):
        """Calculate estimated cost for Claude 3.5 Sonnet"""
        # Claude 3.5 Sonnet pricing (as of 2024)
        input_cost_per_1k = 0.003    # $3 per 1M tokens
        output_cost_per_1k = 0.015   # $15 per 1M tokens
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
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
6. Detailed reasoning for each conclusion

Focus on identifying key fraud indicators such as:
- Balance inconsistencies
- Unusual transaction amounts
- Suspicious transaction types
- Account behavior patterns
- Geographic anomalies
- Timing patterns

Please be thorough and provide actionable insights that could help prevent similar fraud in the future.
"""
        return prompt
    
    async def run_test(self):
        """Run the complete test"""
        print("🔬 CLAUDE MODEL TESTING SUITE")
        print("=" * 50)
        print(f"🔑 API Key Status: {'✅ Set' if self.api_key else '❌ Missing'}")
        print(f"📊 Test Data: {len(self.fraud_samples)} fraud samples loaded")
        print()
        
        # Test latest model
        success, result = await self.test_latest_claude_model()
        
        if success:
            print("\n🎉 CLAUDE TEST SUCCESSFUL!")
            print("✅ Ready to proceed with comprehensive Claude model analysis")
            print(f"🏆 Claude 3.5 Sonnet generated {result['analysis_length']:,} characters of fraud analysis")
            print(f"⚡ Response time: {result['response_time']:.2f} seconds")
            return True
        else:
            print("\n❌ CLAUDE TEST FAILED")
            print("🔧 Please fix the issues above before proceeding")
            return False

async def main():
    """Main test function"""
    tester = ClaudeModelTester()
    return await tester.run_test()

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🚀 Next step: Run comprehensive Claude model analyzer")
    else:
        print("\n🛠️ Fix API key issues first, then retry")