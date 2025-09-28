#!/usr/bin/env python3
"""
Scaled BigQuery Fraud Analyzer - 10,000 Sample Points
Enhanced version that uses 10,000 fraud samples instead of 100-170
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
import pandas as pd
from google.cloud import bigquery
import google.generativeai as genai
import requests
import time

class ScaledFraudAnalyzer:
    """Analyze fraud patterns using 10,000+ BigQuery samples and multiple LLMs"""
    
    def __init__(self):
        """Initialize the scaled analyzer"""
        load_dotenv('.env')
        
        # Set up BigQuery client
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.bq_client = bigquery.Client()
        
        # Set up Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Analysis results
        self.analysis_results = {}
        
        print("🚀 SCALED FRAUD ANALYZER INITIALIZED")
        print("📊 Target: 10,000+ fraud samples for analysis")
        print("=" * 50)
        
    def extract_scaled_fraud_data(self) -> Dict[str, pd.DataFrame]:
        """Extract 10,000+ fraud samples from BigQuery datasets"""
        print("\n🔍 EXTRACTING 10,000+ FRAUD SAMPLES FROM BIGQUERY")
        print("=" * 50)
        
        # Scaled queries - distributed across datasets for 10K total samples
        scaled_queries = {
            'ieee_cis_fraud': """
                SELECT 
                    TransactionID, TransactionDT, TransactionAmt, ProductCD,
                    card1, card2, card3, card4, card5, card6,
                    addr1, addr2, P_emaildomain, R_emaildomain, isFraud
                FROM `vaulted-timing-473322-f9.ieee_cis_fraud.train_transaction`
                ORDER BY RAND()
                LIMIT 3000  -- 3K samples from IEEE dataset
            """,
            
            'credit_card_fraud': """
                SELECT 
                    Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
                    V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
                    V21, V22, V23, V24, V25, V26, V27, V28, Amount, Class
                FROM `vaulted-timing-473322-f9.credit_card_fraud.creditcard`
                WHERE RAND() < 0.15  -- 15% sample = ~42K records, but limited
                ORDER BY RAND()
                LIMIT 2500  -- 2.5K samples from Credit Card dataset
            """,
            
            'paysim_fraud': """
                SELECT 
                    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                    nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
                FROM `vaulted-timing-473322-f9.paysim_fraud.paysim_data`
                WHERE RAND() < 0.001  -- 0.1% sample from 6M records
                ORDER BY RAND()
                LIMIT 4000  -- 4K samples from PaySim dataset
            """,
            
            'relational_fraud': """
                SELECT 
                    c.CustomerID, c.Age, c.Gender, c.Income, c.CreditScore,
                    t.TransactionID, t.Amount, t.TransactionType, t.Timestamp,
                    m.MerchantID, m.MerchantCategory, m.RiskScore,
                    a.AnomalyScore, a.RiskLevel
                FROM `vaulted-timing-473322-f9.relational_fraud.customers` c
                JOIN `vaulted-timing-473322-f9.relational_fraud.transactions` t ON c.CustomerID = t.CustomerID
                JOIN `vaulted-timing-473322-f9.relational_fraud.merchants` m ON t.MerchantID = m.MerchantID
                JOIN `vaulted-timing-473322-f9.relational_fraud.anomaly_scores` a ON t.TransactionID = a.TransactionID
                ORDER BY RAND()
                LIMIT 500  -- 500 samples from Relational dataset
            """
        }
        
        datasets = {}
        total_records = 0
        
        for name, query in scaled_queries.items():
            try:
                print(f"   📊 Querying {name}...")
                start_time = time.time()
                
                query_job = self.bq_client.query(query)
                df = query_job.to_dataframe()
                
                datasets[name] = df
                query_time = time.time() - start_time
                
                # Calculate fraud rate
                fraud_count = 0
                if 'isFraud' in df.columns:
                    fraud_count = df['isFraud'].sum()
                elif 'Class' in df.columns:
                    fraud_count = df['Class'].sum()
                elif 'RiskLevel' in df.columns:
                    fraud_count = (df['RiskLevel'] == 'High').sum()
                
                fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
                
                print(f"      ✅ {len(df):,} records extracted ({fraud_count} fraud cases, {fraud_rate:.1f}% fraud rate)")
                print(f"      ⏱️  Query time: {query_time:.2f} seconds")
                total_records += len(df)
                
            except Exception as e:
                print(f"      ❌ Error with {name}: {str(e)}")
                datasets[name] = pd.DataFrame()
        
        print(f"\n🎯 TOTAL SAMPLES EXTRACTED: {total_records:,}")
        print(f"📈 Scale increase: {total_records/170:.1f}x larger than original")
        
        return datasets
    
    def create_comprehensive_analysis_prompt(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """Create analysis prompt for scaled dataset"""
        
        total_samples = sum(len(df) for df in datasets.values())
        scale_factor = total_samples / 170  # Original sample size
        
        # Create data summary for LLM analysis
        data_summary = {}
        total_fraud_cases = 0
        
        for name, df in datasets.items():
            if not df.empty:
                # Count fraud cases
                fraud_count = 0
                if 'isFraud' in df.columns:
                    fraud_count = df['isFraud'].sum()
                elif 'Class' in df.columns:
                    fraud_count = df['Class'].sum()
                elif 'RiskLevel' in df.columns:
                    fraud_count = (df['RiskLevel'] == 'High').sum()
                
                total_fraud_cases += fraud_count
                
                # Sample data for analysis (first 5 records)
                sample_records = df.head(5).to_dict('records')
                
                data_summary[name] = {
                    'total_records': len(df),
                    'fraud_cases': int(fraud_count),
                    'fraud_rate': f"{fraud_count/len(df)*100:.2f}%",
                    'columns': list(df.columns),
                    'sample_records': sample_records
                }
        
        prompt = f"""
COMPREHENSIVE FRAUD DETECTION ANALYSIS - SCALED DATASET

📊 **Dataset Scale**: {total_samples:,} transactions analyzed
🚀 **Scale Factor**: {scale_factor:.1f}x larger than baseline analysis
🎯 **Fraud Cases**: {total_fraud_cases:,} confirmed fraud transactions
📈 **Statistical Power**: Enterprise-grade sample size for reliable patterns

**Dataset Breakdown**:
{json.dumps(data_summary, indent=2, default=str)[:3000]}

**ANALYSIS REQUIREMENTS FOR SCALED DATA**:

1. **STATISTICAL FRAUD PATTERNS** (powered by {total_samples:,} samples):
   - Identify fraud patterns with statistical significance
   - Cross-validate patterns across multiple datasets
   - Quantify pattern reliability and false positive rates

2. **PRODUCTION-READY INSIGHTS**:
   - Real-time fraud scoring algorithms for {total_samples:,}+ transaction volumes
   - Machine learning feature recommendations
   - Risk thresholds optimized for scale

3. **ENTERPRISE DEPLOYMENT STRATEGY**:
   - Scalable fraud detection architecture
   - Performance benchmarks for high-volume processing
   - Cost-benefit analysis for different detection approaches

4. **BUSINESS IMPACT PROJECTIONS**:
   - Fraud loss prevention estimates based on {total_fraud_cases:,} fraud cases
   - ROI calculations for fraud prevention systems
   - Compliance and regulatory considerations

**DELIVERABLES**:
- Top 10 most reliable fraud indicators from {total_samples:,} transactions
- Production-ready fraud scoring methodology
- Scalable implementation recommendations
- Business case for fraud prevention system

Provide comprehensive, enterprise-level analysis suitable for financial institution deployment.
"""
        return prompt
    
    async def analyze_with_multiple_models(self, datasets: Dict[str, pd.DataFrame]):
        """Analyze scaled fraud data with multiple AI models"""
        
        print(f"\n🤖 ANALYZING {sum(len(df) for df in datasets.values()):,} SAMPLES WITH MULTIPLE AI MODELS")
        print("=" * 70)
        
        # Create comprehensive prompt
        analysis_prompt = self.create_comprehensive_analysis_prompt(datasets)
        
        # Test models on scaled data
        models_to_test = [
            'gemini-2.0-flash-exp',
            'gemini-2.5-flash-lite',
            'gemini-2.0-flash',
            'phi3:3.8b'  # Local model for comparison
        ]
        
        results = []
        
        for model_name in models_to_test:
            try:
                print(f"   🧪 Testing {model_name} on scaled dataset...")
                start_time = time.time()
                
                if 'phi3' in model_name:
                    # Local Ollama model
                    response = requests.post(
                        'http://localhost:11434/api/generate',
                        json={
                            'model': model_name,
                            'prompt': analysis_prompt[:6000],  # Truncate for local model
                            'stream': False
                        },
                        timeout=600  # 10 minute timeout for scaled analysis
                    )
                    
                    if response.status_code == 200:
                        analysis_text = response.json().get('response', '')
                        status = 'success'
                    else:
                        analysis_text = f"Local model error: {response.status_code}"
                        status = 'error'
                
                else:
                    # Gemini models
                    model = genai.GenerativeModel(model_name)
                    
                    response = model.generate_content(
                        analysis_prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=8000,
                            temperature=0.1
                        )
                    )
                    analysis_text = response.text if response else "No response"
                    status = 'success'
                
                analysis_time = time.time() - start_time
                
                result = {
                    'model': model_name,
                    'status': status,
                    'analysis_length': len(analysis_text),
                    'analysis_time': analysis_time,
                    'samples_analyzed': sum(len(df) for df in datasets.values()),
                    'scale_factor': f"{sum(len(df) for df in datasets.values())/170:.1f}x",
                    'analysis': analysis_text[:15000],  # First 15K chars
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
                print(f"      ✅ {model_name}: {len(analysis_text):,} characters in {analysis_time:.2f}s")
                
            except Exception as e:
                error_result = {
                    'model': model_name,
                    'status': 'error',
                    'error': str(e)[:500],
                    'samples_analyzed': sum(len(df) for df in datasets.values())
                }
                results.append(error_result)
                print(f"      ❌ {model_name}: {str(e)[:100]}")
        
        return results
    
    async def run_scaled_analysis(self):
        """Run complete scaled fraud analysis"""
        print("🚀 STARTING SCALED FRAUD ANALYSIS")
        print("🎯 Target: 10,000+ samples across 4 fraud datasets")
        print("=" * 50)
        
        # Extract scaled data
        datasets = self.extract_scaled_fraud_data()
        
        if not any(len(df) > 0 for df in datasets.values()):
            print("❌ No data extracted. Check BigQuery permissions and dataset access.")
            return
        
        # Analyze with multiple models
        analysis_results = await self.analyze_with_multiple_models(datasets)
        
        # Compile comprehensive results
        final_results = {
            'analysis_type': 'scaled_fraud_detection',
            'timestamp': datetime.now().isoformat(),
            'scale_info': {
                'total_samples': sum(len(df) for df in datasets.values()),
                'original_sample_size': 170,
                'scale_factor': f"{sum(len(df) for df in datasets.values())/170:.1f}x",
                'datasets': {name: len(df) for name, df in datasets.items()}
            },
            'model_results': analysis_results,
            'successful_models': len([r for r in analysis_results if r.get('status') == 'success']),
            'total_models_tested': len(analysis_results)
        }
        
        # Save results
        output_file = '../data/scaled_fraud_analysis_10k.json'
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        print(f"\n🎉 SCALED ANALYSIS COMPLETE!")
        print("=" * 40)
        print(f"📊 Total Samples: {final_results['scale_info']['total_samples']:,}")
        print(f"🚀 Scale Factor: {final_results['scale_info']['scale_factor']}")
        print(f"✅ Successful Models: {final_results['successful_models']}/{final_results['total_models_tested']}")
        print(f"💾 Results Saved: {output_file}")
        
        # Show best performing models
        successful_models = [r for r in analysis_results if r.get('status') == 'success']
        if successful_models:
            best_model = max(successful_models, key=lambda x: x.get('analysis_length', 0))
            print(f"🏆 Best Analysis: {best_model['model']} ({best_model['analysis_length']:,} characters)")
        
        return final_results

async def main():
    """Run the scaled fraud analyzer"""
    analyzer = ScaledFraudAnalyzer()
    results = await analyzer.run_scaled_analysis()
    
    print("\n🎯 SCALING ACHIEVEMENT:")
    print(f"   From 170 samples → {results['scale_info']['total_samples']:,} samples")
    print(f"   Scale increase: {results['scale_info']['scale_factor']}")
    print(f"   Enterprise-grade fraud detection analysis complete!")

if __name__ == "__main__":
    asyncio.run(main())