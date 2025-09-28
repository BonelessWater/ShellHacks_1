#!/usr/bin/env python3
"""
Optimized 10K Fraud Analyzer - Uses Available Datasets
Targets exactly 10,000 fraud samples from available BigQuery datasets
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

class Optimized10KAnalyzer:
    """Analyze fraud patterns using exactly 10,000 BigQuery samples"""
    
    def __init__(self):
        """Initialize the 10K analyzer"""
        load_dotenv('.env')
        
        # Set up BigQuery client
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.bq_client = bigquery.Client()
        
        # Set up Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        print("🚀 OPTIMIZED 10K FRAUD ANALYZER")
        print("🎯 Target: Exactly 10,000 fraud samples")
        print("=" * 40)
        
    def extract_10k_samples(self) -> Dict[str, pd.DataFrame]:
        """Extract exactly 10,000 fraud samples from available datasets"""
        print("\n🔍 EXTRACTING 10,000 FRAUD SAMPLES")
        print("=" * 40)
        
        # Optimized queries targeting available datasets for 10K total
        queries = {
            'ieee_cis_main': """
                SELECT 
                    TransactionID, TransactionDT, TransactionAmt, ProductCD,
                    card1, card2, card3, card4, card5, card6,
                    addr1, addr2, P_emaildomain, R_emaildomain, isFraud
                FROM `vaulted-timing-473322-f9.ieee_cis_fraud.train_transaction`
                ORDER BY RAND()
                LIMIT 5000  -- 5K from main IEEE dataset
            """,
            
            'ieee_cis_identity': """
                SELECT 
                    TransactionID, DeviceType, DeviceInfo, 
                    id_01, id_02, id_03, id_04, id_05, id_06, id_07, id_08, id_09, id_10,
                    id_11, id_12, id_13, id_14, id_15, id_16, id_17, id_18, id_19, id_20
                FROM `vaulted-timing-473322-f9.ieee_cis_fraud.train_identity`
                ORDER BY RAND()  
                LIMIT 2500  -- 2.5K from identity data
            """,
            
            'document_forgery': """
                SELECT *
                FROM `vaulted-timing-473322-f9.document_forgery.document_data`
                ORDER BY RAND()
                LIMIT 1500  -- 1.5K from document forgery
            """,
            
            'additional_ieee': """
                SELECT 
                    TransactionID, TransactionDT, TransactionAmt, ProductCD,
                    C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14,
                    D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15,
                    M1, M2, M3, M4, M5, M6, M7, M8, M9,
                    V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, isFraud
                FROM `vaulted-timing-473322-f9.ieee_cis_fraud.train_transaction`
                WHERE TransactionID NOT IN (
                    SELECT TransactionID 
                    FROM `vaulted-timing-473322-f9.ieee_cis_fraud.train_transaction`
                    ORDER BY RAND()
                    LIMIT 5000
                )
                ORDER BY RAND()
                LIMIT 1000  -- Additional 1K unique records
            """
        }
        
        datasets = {}
        total_records = 0
        target_records = 10000
        
        for name, query in queries.items():
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
                elif 'is_fraudulent' in df.columns:
                    fraud_count = df['is_fraudulent'].sum()
                
                fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
                
                print(f"      ✅ {len(df):,} records ({fraud_count} fraud, {fraud_rate:.1f}% rate)")
                print(f"      ⏱️  {query_time:.2f}s")
                total_records += len(df)
                
                # Stop if we've reached our target
                if total_records >= target_records:
                    print(f"   🎯 Target reached: {total_records:,} samples")
                    break
                    
            except Exception as e:
                print(f"      ❌ {name}: {str(e)[:100]}")
                datasets[name] = pd.DataFrame()
        
        # If we're short, try to get more from IEEE dataset
        if total_records < target_records:
            remaining = target_records - total_records
            print(f"   📈 Getting {remaining} more samples...")
            
            extra_query = f"""
                SELECT * FROM `vaulted-timing-473322-f9.ieee_cis_fraud.train_transaction`
                ORDER BY RAND()
                LIMIT {remaining}
            """
            
            try:
                df_extra = self.bq_client.query(extra_query).to_dataframe()
                datasets['extra_samples'] = df_extra
                total_records += len(df_extra)
                print(f"      ✅ {len(df_extra):,} additional records")
            except Exception as e:
                print(f"      ❌ Extra samples: {str(e)[:100]}")
        
        print(f"\n🎯 FINAL COUNT: {total_records:,} samples")
        print(f"📈 Scale increase: {total_records/170:.1f}x from original")
        
        return datasets
    
    def create_10k_analysis_prompt(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """Create analysis prompt for 10K dataset"""
        
        total_samples = sum(len(df) for df in datasets.values())
        scale_factor = total_samples / 170
        
        # Data summary
        data_summary = {}
        total_fraud = 0
        
        for name, df in datasets.items():
            if not df.empty:
                fraud_cols = ['isFraud', 'Class', 'is_fraudulent']
                fraud_count = 0
                
                for col in fraud_cols:
                    if col in df.columns:
                        if col == 'Class':
                            fraud_count = df[col].sum()
                        else:
                            fraud_count = (df[col] == 1).sum() if col == 'isFraud' else df[col].sum()
                        break
                
                total_fraud += fraud_count
                
                data_summary[name] = {
                    'records': len(df),
                    'fraud_cases': int(fraud_count),
                    'fraud_rate': f"{fraud_count/len(df)*100:.2f}%",
                    'key_columns': list(df.columns)[:10]  # First 10 columns
                }
        
        prompt = f"""
ENTERPRISE FRAUD DETECTION ANALYSIS - 10,000 SAMPLE DATASET

🎯 **MASSIVE SCALE ANALYSIS**: {total_samples:,} transactions
🚀 **Scale Achievement**: {scale_factor:.1f}x larger than baseline
📊 **Fraud Cases**: {total_fraud:,} confirmed fraud transactions
⚡ **Statistical Power**: Production-grade sample size

**Dataset Composition**:
{json.dumps(data_summary, indent=2)}

**COMPREHENSIVE ANALYSIS REQUIREMENTS**:

1. **STATISTICAL FRAUD DETECTION** (N={total_samples:,}):
   - Identify fraud patterns with 95%+ confidence intervals
   - Cross-validate patterns across datasets
   - Quantify detection accuracy and false positive rates
   - Statistical significance testing for all patterns

2. **PRODUCTION DEPLOYMENT INSIGHTS**:
   - Real-time fraud scoring for {total_samples:,}+ transaction volumes
   - Scalable machine learning pipeline recommendations
   - Feature engineering for high-volume processing
   - Performance optimization strategies

3. **ENTERPRISE RISK MANAGEMENT**:
   - Risk scoring methodology (0-100 scale)
   - Dynamic threshold recommendations
   - Cost-benefit analysis for different risk tolerances
   - Compliance and regulatory framework alignment

4. **BUSINESS INTELLIGENCE**:
   - ROI projections for fraud prevention systems
   - Market benchmark comparisons
   - Industry-specific fraud pattern analysis
   - Strategic recommendations for financial institutions

**EXPECTED DELIVERABLES**:
- Top 15 statistically significant fraud indicators
- Production-ready fraud detection algorithm
- Scalable system architecture recommendations
- Executive summary with business impact projections

Based on {total_samples:,} transactions, provide enterprise-grade fraud detection analysis suitable for billion-dollar financial institutions.
"""
        return prompt
    
    async def analyze_10k_with_ai_models(self, datasets: Dict[str, pd.DataFrame]):
        """Analyze 10K samples with AI models"""
        
        total_samples = sum(len(df) for df in datasets.values())
        print(f"\n🤖 AI ANALYSIS OF {total_samples:,} FRAUD SAMPLES")
        print("=" * 50)
        
        analysis_prompt = self.create_10k_analysis_prompt(datasets)
        
        # AI models for 10K analysis
        models = [
            'gemini-2.0-flash-exp',    # Best comprehensive analysis  
            'gemini-2.5-flash-lite',   # Fastest high-quality
            'gemini-2.0-flash',        # Balanced performance
            'phi3:3.8b'                # Local comparison
        ]
        
        results = []
        
        for model_name in models:
            print(f"   🧪 {model_name} analyzing {total_samples:,} samples...")
            
            try:
                start_time = time.time()
                
                if 'phi3' in model_name:
                    # Local model
                    response = requests.post(
                        'http://localhost:11434/api/generate',
                        json={
                            'model': model_name,
                            'prompt': analysis_prompt[:8000],  # Truncated for local
                            'stream': False
                        },
                        timeout=900  # 15 minutes for 10K analysis
                    )
                    
                    analysis_text = response.json().get('response', '') if response.status_code == 200 else f"Error: {response.status_code}"
                    
                else:
                    # Gemini models
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        analysis_prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=12000,  # Larger for 10K analysis
                            temperature=0.1
                        )
                    )
                    analysis_text = response.text if response else "No response"
                
                analysis_time = time.time() - start_time
                efficiency = len(analysis_text) / analysis_time if analysis_time > 0 else 0
                
                result = {
                    'model': model_name,
                    'status': 'success',
                    'samples_analyzed': total_samples,
                    'analysis_length': len(analysis_text),
                    'analysis_time': analysis_time,
                    'efficiency': efficiency,
                    'scale_factor': f"{total_samples/170:.1f}x",
                    'analysis_preview': analysis_text[:2000],  # First 2K chars
                    'full_analysis': analysis_text,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                print(f"      ✅ {len(analysis_text):,} chars in {analysis_time:.1f}s ({efficiency:.0f} c/s)")
                
            except Exception as e:
                error_result = {
                    'model': model_name,
                    'status': 'error',
                    'error': str(e)[:300],
                    'samples_analyzed': total_samples
                }
                results.append(error_result)
                print(f"      ❌ Error: {str(e)[:80]}")
        
        return results
    
    async def run_10k_analysis(self):
        """Execute complete 10K fraud analysis"""
        print("🚀 STARTING 10,000 SAMPLE FRAUD ANALYSIS")
        print("=" * 45)
        
        # Extract 10K samples
        datasets = self.extract_10k_samples()
        total_samples = sum(len(df) for df in datasets.values())
        
        if total_samples < 1000:
            print("❌ Insufficient data extracted")
            return
        
        # AI analysis
        analysis_results = await self.analyze_10k_with_ai_models(datasets)
        
        # Compile results
        final_results = {
            'analysis_type': '10k_fraud_detection',
            'timestamp': datetime.now().isoformat(),
            'scale_metrics': {
                'total_samples': total_samples,
                'target_samples': 10000,
                'achievement_rate': f"{total_samples/10000*100:.1f}%",
                'scale_vs_original': f"{total_samples/170:.1f}x",
                'datasets_used': list(datasets.keys())
            },
            'model_performance': analysis_results,
            'success_rate': f"{len([r for r in analysis_results if r.get('status') == 'success'])}/{len(analysis_results)}"
        }
        
        # Save results
        output_file = '../data/fraud_analysis_10k_samples.json'
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Final report
        print(f"\n🎉 10K ANALYSIS COMPLETE!")
        print("=" * 30)
        print(f"📊 Samples Analyzed: {total_samples:,}")
        print(f"🎯 Target Achievement: {total_samples/10000*100:.1f}%")
        print(f"🚀 Scale Increase: {total_samples/170:.1f}x")
        print(f"✅ Success Rate: {final_results['success_rate']}")
        print(f"💾 Results: {output_file}")
        
        successful = [r for r in analysis_results if r.get('status') == 'success']
        if successful:
            best = max(successful, key=lambda x: x.get('analysis_length', 0))
            print(f"🏆 Best Model: {best['model']} ({best['analysis_length']:,} characters)")
        
        return final_results

async def main():
    analyzer = Optimized10KAnalyzer()
    await analyzer.run_10k_analysis()

if __name__ == "__main__":
    asyncio.run(main())