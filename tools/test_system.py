#!/usr/bin/env python3
"""
Test runner for LLM Fraud Analyzer - Demo Mode
This runs the fraud analyzer without requiring API keys for testing purposes.
"""

import json
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class MockLLMResponse:
    """Mock response for testing"""
    model_name: str
    timestamp: str
    prompt_type: str
    response: str
    analysis_time_seconds: float
    fraud_patterns_identified: List[str]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]

class MockLLMFraudAnalyzer:
    """Mock version of the fraud analyzer for testing"""
    
    def __init__(self, invoice_data_path: str, output_path: str):
        self.invoice_data_path = invoice_data_path
        self.output_path = output_path
        self.invoice_data = self.load_invoice_data()
        self.results: List[MockLLMResponse] = []
    
    def load_invoice_data(self) -> List[Dict]:
        """Load invoice data from JSON file"""
        try:
            with open(self.invoice_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Loaded {len(data)} invoices from {self.invoice_data_path}")
                return data
        except Exception as e:
            print(f"❌ Error loading invoice data: {e}")
            return []
    
    def generate_mock_responses(self) -> List[MockLLMResponse]:
        """Generate mock responses for testing"""
        
        # Count fraud patterns in actual data
        fraud_count = len([inv for inv in self.invoice_data if inv.get('is_fraudulent', False)])
        legit_count = len(self.invoice_data) - fraud_count
        
        mock_responses = [
            MockLLMResponse(
                model_name="GPT-4-Mock",
                timestamp=datetime.now().isoformat(),
                prompt_type="pattern_identification",
                response=f"""Based on my analysis of {len(self.invoice_data)} invoices ({fraud_count} fraudulent, {legit_count} legitimate), I've identified the following fraud patterns:

1. **Vendor Impersonation** (High Risk)
   - Slight variations in legitimate company names
   - Example: "Microsoft Solutions Inc." instead of "Microsoft Corporation"
   - Detection: Cross-reference vendor names with verified databases

2. **Price Inflation** (High Risk)
   - Services priced significantly above market rates
   - Emergency services with 300-500% markup
   - Detection: Compare prices against industry benchmarks

3. **Fake Address Fraud** (Medium Risk)
   - Non-existent or suspicious addresses
   - Generic addresses like "123 Fake Street"
   - Detection: Address validation APIs and geographic verification

4. **Payment Terms Manipulation** (Medium Risk)
   - Unusually short payment terms (immediate payment)
   - Pressure tactics in payment language
   - Detection: Flag payment terms shorter than industry standard

5. **Service Description Vagueness** (Low Risk)
   - Generic descriptions like "Emergency Services"
   - Lack of detailed breakdowns
   - Detection: Natural language processing for vague terms

These patterns appear in various combinations in the fraudulent invoices analyzed.""",
                analysis_time_seconds=2.3,
                fraud_patterns_identified=["vendor_impersonation", "price_inflation", "fake_address", "payment_terms_abuse", "vague_descriptions"],
                confidence_scores={"overall": 0.92},
                metadata={"token_usage": 1850}
            ),
            
            MockLLMResponse(
                model_name="Gemini-Pro-Mock",
                timestamp=datetime.now().isoformat(),
                prompt_type="detection_rules",
                response=f"""Here are specific detection rules for the {fraud_count} fraud patterns identified:

**Rule 1: Vendor Name Similarity Check**
- Compare vendor names against known legitimate companies
- Flag similarity scores > 80% but not exact matches
- Threshold: Levenshtein distance of 1-3 characters

**Rule 2: Price Anomaly Detection**
- Calculate price per unit for similar services
- Flag invoices with prices > 2 standard deviations from mean
- Emergency services: Flag markup > 150%

**Rule 3: Address Validation**
- Validate addresses using postal service APIs
- Flag non-existent ZIP codes or street addresses
- Cross-reference with business registration databases

**Rule 4: Payment Terms Analysis**
- Flag payment terms < 7 days as suspicious
- Flag language indicating "immediate payment required"
- Normal business terms: Net 15, Net 30, Net 45

**Rule 5: Tax ID Verification**
- Check for duplicate tax IDs across different vendors
- Validate tax ID format against government standards
- Flag tax IDs with patterns like "00-0000000"

**Combination Rules:**
- 2+ indicators = Medium risk (manual review)
- 3+ indicators = High risk (block payment)
- Single indicator = Low risk (automated approval with logging)""",
                analysis_time_seconds=1.8,
                fraud_patterns_identified=["vendor_verification", "price_anomaly", "address_validation", "payment_terms", "tax_id_fraud"],
                confidence_scores={"overall": 0.87},
                metadata={"response_length": 1246}
            ),
            
            MockLLMResponse(
                model_name="Claude-3-Mock",
                timestamp=datetime.now().isoformat(),
                prompt_type="risk_scoring",
                response=f"""Risk Scoring System (0-100 scale) for {len(self.invoice_data)} invoice dataset:

**Scoring Methodology:**

**Vendor Indicators (40 points max):**
- Vendor impersonation: +25 points
- Unverified vendor: +15 points  
- Duplicate tax ID: +20 points
- Suspicious address: +10 points

**Financial Indicators (35 points max):**
- Price inflation (>200% market rate): +25 points
- Round number amounts: +5 points
- Missing tax calculations: +10 points
- Unusual payment terms: +15 points

**Service Indicators (25 points max):**
- Vague service descriptions: +10 points
- Emergency service claims: +15 points
- No detailed breakdown: +8 points
- Unverifiable deliverables: +12 points

**Risk Thresholds:**
- 0-25: Low Risk (Auto-approve)
- 26-50: Medium Risk (Manual review)
- 51-75: High Risk (Additional verification required)
- 76-100: Critical Risk (Block payment)

**Example Scores from Dataset:**
- Legitimate invoice (Tech Solutions): 8/100
- Fraudulent invoice (Quick Fix Solutions): 85/100
- Fraudulent invoice (Dubious Tech LLC): 95/100

**Recommended Implementation:**
- Use weighted scoring with machine learning calibration
- Regular threshold adjustment based on false positive rates
- Maintain audit trail for all scoring decisions""",
                analysis_time_seconds=2.1,
                fraud_patterns_identified=["comprehensive_scoring", "risk_thresholds", "weighted_indicators"],
                confidence_scores={"overall": 0.94},
                metadata={"input_tokens": 1500, "output_tokens": 1100}
            )
        ]
        
        return mock_responses
    
    async def run_mock_analysis(self):
        """Run mock analysis for testing"""
        print("🔍 Running Mock LLM Fraud Analysis...")
        print("=" * 50)
        
        if not self.invoice_data:
            print("❌ No invoice data loaded. Cannot proceed.")
            return
        
        # Generate mock responses
        self.results = self.generate_mock_responses()
        
        print(f"✅ Generated {len(self.results)} mock LLM responses")
        print("📊 Analysis Summary:")
        
        for result in self.results:
            print(f"   - {result.model_name}: {result.prompt_type}")
            print(f"     Patterns found: {len(result.fraud_patterns_identified)}")
            print(f"     Confidence: {result.confidence_scores.get('overall', 0):.2f}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save mock results to output file"""
        try:
            # Convert dataclass objects to dictionaries
            results_dict = [asdict(result) for result in self.results]
            
            # Count fraud patterns
            all_patterns = []
            for result in self.results:
                all_patterns.extend(result.fraud_patterns_identified)
            
            pattern_frequency = {}
            for pattern in all_patterns:
                pattern_frequency[pattern] = pattern_frequency.get(pattern, 0) + 1
            
            # Create output structure
            output_data = {
                "analysis_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "invoice_data_source": self.invoice_data_path,
                    "total_invoices_analyzed": len(self.invoice_data),
                    "fraudulent_invoices_count": len([inv for inv in self.invoice_data if inv.get('is_fraudulent', False)]),
                    "legitimate_invoices_count": len([inv for inv in self.invoice_data if not inv.get('is_fraudulent', False)]),
                    "llm_responses_count": len(self.results),
                    "analysis_type": "MOCK_DEMO"
                },
                "llm_responses": results_dict,
                "summary_analysis": {
                    "models_used": [result.model_name for result in self.results],
                    "prompt_types_analyzed": [result.prompt_type for result in self.results],
                    "total_responses": len(self.results),
                    "fraud_patterns_frequency": pattern_frequency,
                    "average_analysis_time_seconds": sum(r.analysis_time_seconds for r in self.results) / len(self.results),
                    "most_common_patterns": sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)
                }
            }
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Mock results saved to {self.output_path}")
            print(f"📄 File size: {os.path.getsize(self.output_path)} bytes")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main function to run mock analysis"""
    print("🧪 LLM Fraud Analyzer - Demo Mode")
    print("=" * 40)
    print("This is a demonstration run without real API calls")
    print()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "..", "data", "invoice_training_data.json")
    output_path = os.path.join(script_dir, "..", "data", "llm_fraud_analysis_results.json")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return 1
    
    # Run mock analysis
    analyzer = MockLLMFraudAnalyzer(input_path, output_path)
    
    try:
        asyncio.run(analyzer.run_mock_analysis())
        print("\n🎉 Mock analysis completed successfully!")
        print(f"📁 Check results in: {output_path}")
        return 0
    except Exception as e:
        print(f"❌ Mock analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())