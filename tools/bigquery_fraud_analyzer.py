#!/usr/bin/env python3
"""
BigQuery-powered LLM Fraud Pattern Analyzer
Uses your massive GCP datasets for comprehensive fraud pattern analysis
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
import pandas as pd
from google.cloud import bigquery

class BigQueryFraudAnalyzer:
    """Analyze fraud patterns using BigQuery data and multiple LLMs"""
    
    def __init__(self):
        """Initialize the analyzer"""
        load_dotenv('.env')
        
        # Set up BigQuery client
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.bq_client = bigquery.Client()
        
        # Initialize LLM clients
        self.setup_llm_clients()
        
        # Analysis results
        self.analysis_results = {}
        
    def setup_llm_clients(self):
        """Set up LLM API clients"""
        # Google Gemini
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Google Gemini configured")
        except Exception as e:
            print(f"⚠️  Gemini setup failed: {e}")
            self.gemini_model = None
        
        # OpenAI GPT (if available)
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            print("✅ OpenAI GPT configured")
        except Exception as e:
            print(f"⚠️  OpenAI setup failed: {e}")
            self.openai_client = None
    
    def extract_fraud_samples(self) -> Dict[str, pd.DataFrame]:
        """Extract representative fraud samples from BigQuery"""
        print("\n🔍 EXTRACTING FRAUD SAMPLES FROM BIGQUERY")
        print("=" * 45)
        
        datasets = {}
        
        # 1. IEEE CIS Fraud - Get fraud and non-fraud samples
        print("📊 Extracting IEEE CIS fraud data...")
        ieee_query = """
        SELECT 
            TransactionID,
            isFraud,
            TransactionAmt,
            ProductCD,
            card1,
            card2,
            card3,
            card4,
            card5,
            card6,
            addr1,
            addr2,
            dist1,
            dist2,
            P_emaildomain,
            R_emaildomain,
            C1, C2, C3, C4, C5
        FROM `vaulted-timing-473322-f9.ieee_cis_fraud.train_transaction`
        WHERE isFraud = 1  -- Fraud cases
        LIMIT 25
        
        UNION ALL
        
        SELECT 
            TransactionID,
            isFraud,
            TransactionAmt,
            ProductCD,
            card1,
            card2,
            card3,
            card4,
            card5,
            card6,
            addr1,
            addr2,
            dist1,
            dist2,
            P_emaildomain,
            R_emaildomain,
            C1, C2, C3, C4, C5
        FROM `vaulted-timing-473322-f9.ieee_cis_fraud.train_transaction`
        WHERE isFraud = 0  -- Normal cases
        ORDER BY RAND()
        LIMIT 25
        """
        
        try:
            ieee_df = self.bq_client.query(ieee_query).to_dataframe()
            datasets['ieee_cis_fraud'] = ieee_df
            print(f"  ✅ Extracted {len(ieee_df)} IEEE CIS transactions")
        except Exception as e:
            print(f"  ❌ IEEE extraction failed: {e}")
        
        # 2. Credit Card Fraud
        print("💳 Extracting credit card fraud data...")
        cc_query = """
        SELECT 
            Time,
            V1, V2, V3, V4, V5,
            Amount,
            Class
        FROM `vaulted-timing-473322-f9.transactional_fraud.credit_card_fraud`
        WHERE Class = 1  -- Fraud
        LIMIT 25
        
        UNION ALL
        
        SELECT 
            Time,
            V1, V2, V3, V4, V5,
            Amount,
            Class
        FROM `vaulted-timing-473322-f9.transactional_fraud.credit_card_fraud`
        WHERE Class = 0  -- Normal
        ORDER BY RAND()
        LIMIT 25
        """
        
        try:
            cc_df = self.bq_client.query(cc_query).to_dataframe()
            datasets['credit_card_fraud'] = cc_df
            print(f"  ✅ Extracted {len(cc_df)} credit card transactions")
        except Exception as e:
            print(f"  ❌ Credit card extraction failed: {e}")
        
        # 3. PaySim Mobile Money Fraud
        print("📱 Extracting PaySim mobile money fraud...")
        paysim_query = """
        SELECT 
            step,
            type,
            amount,
            nameOrig,
            oldbalanceOrg,
            newbalanceOrig,
            nameDest,
            oldbalanceDest,
            newbalanceDest,
            isFraud,
            isFlaggedFraud
        FROM `vaulted-timing-473322-f9.transactional_fraud.paysim`
        WHERE isFraud = 1  -- Fraud cases
        LIMIT 20
        
        UNION ALL
        
        SELECT 
            step,
            type,
            amount,
            nameOrig,
            oldbalanceOrg,
            newbalanceOrig,
            nameDest,
            oldbalanceDest,
            newbalanceDest,
            isFraud,
            isFlaggedFraud
        FROM `vaulted-timing-473322-f9.transactional_fraud.paysim`
        WHERE isFraud = 0  -- Normal cases
        ORDER BY RAND()
        LIMIT 20
        """
        
        try:
            paysim_df = self.bq_client.query(paysim_query).to_dataframe()
            datasets['paysim_fraud'] = paysim_df
            print(f"  ✅ Extracted {len(paysim_df)} PaySim transactions")
        except Exception as e:
            print(f"  ❌ PaySim extraction failed: {e}")
        
        # 4. Relational Fraud Data
        print("🔗 Extracting relational fraud indicators...")
        relational_query = """
        SELECT 
            f.TransactionID,
            f.FraudIndicator,
            t.Amount as TransactionAmount,
            t.Timestamp,
            s.SuspiciousFlag,
            a.AnomalyScore
        FROM `vaulted-timing-473322-f9.fraud_detection_relational.fraudulent_patterns_fraud_indicators` f
        LEFT JOIN `vaulted-timing-473322-f9.fraud_detection_relational.transaction_amounts_amount_data` t
        ON f.TransactionID = t.TransactionID
        LEFT JOIN `vaulted-timing-473322-f9.fraud_detection_relational.fraudulent_patterns_suspicious_activity` s
        ON LEFT(f.TransactionID, 10) = s.CustomerID  -- Rough join
        LEFT JOIN `vaulted-timing-473322-f9.fraud_detection_relational.transaction_amounts_anomaly_scores` a
        ON f.TransactionID = a.TransactionID
        WHERE f.FraudIndicator IS NOT NULL
        LIMIT 30
        """
        
        try:
            relational_df = self.bq_client.query(relational_query).to_dataframe()
            datasets['relational_fraud'] = relational_df
            print(f"  ✅ Extracted {len(relational_df)} relational fraud records")
        except Exception as e:
            print(f"  ❌ Relational extraction failed: {e}")
        
        return datasets
    
    async def analyze_fraud_patterns_with_llm(self, datasets: Dict[str, pd.DataFrame], llm_name: str) -> Dict:
        """Analyze fraud patterns using specified LLM"""
        print(f"\n🤖 ANALYZING WITH {llm_name.upper()}")
        print("=" * 30)
        
        analysis_prompt = f"""
        You are a fraud detection expert analyzing real-world financial fraud datasets.
        
        I have extracted samples from multiple fraud detection datasets:
        
        1. **IEEE CIS Fraud Detection Dataset**: Credit card fraud with {len(datasets.get('ieee_cis_fraud', []))} samples
        2. **Credit Card Fraud Dataset**: Traditional CC fraud with {len(datasets.get('credit_card_fraud', []))} samples  
        3. **PaySim Mobile Money Fraud**: Mobile payment fraud with {len(datasets.get('paysim_fraud', []))} samples
        4. **Relational Fraud Data**: Multi-table fraud indicators with {len(datasets.get('relational_fraud', []))} samples
        
        **Your Task:**
        Analyze these datasets and identify:
        
        1. **Top 5 Fraud Patterns** - What are the most common fraud indicators?
        2. **Transaction Characteristics** - What makes fraudulent transactions different?
        3. **Amount Patterns** - Are there specific amount ranges or patterns in fraud?
        4. **Behavioral Indicators** - What user behaviors signal fraud?
        5. **Risk Scoring Factors** - What features are most predictive of fraud?
        6. **Prevention Strategies** - How can these patterns be used to prevent fraud?
        
        **Sample Data Context:**
        - IEEE dataset has features like TransactionAmt, ProductCD, email domains, card info
        - Credit card dataset has PCA-transformed features V1-V28 plus Amount and Time
        - PaySim has mobile money transfer types: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
        - Relational data has customer profiles, merchant info, and anomaly scores
        
        Please provide a comprehensive analysis focusing on actionable fraud detection insights.
        """
        
        try:
            if llm_name == "gemini" and self.gemini_model:
                response = self.gemini_model.generate_content(analysis_prompt)
                return {
                    "llm": "Google Gemini",
                    "analysis": response.text,
                    "timestamp": datetime.now().isoformat(),
                    "datasets_analyzed": list(datasets.keys()),
                    "total_samples": sum(len(df) for df in datasets.values())
                }
            
            elif llm_name == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a fraud detection expert analyzing financial datasets."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.3
                )
                
                return {
                    "llm": "OpenAI GPT-4",
                    "analysis": response.choices[0].message.content,
                    "timestamp": datetime.now().isoformat(),
                    "datasets_analyzed": list(datasets.keys()),
                    "total_samples": sum(len(df) for df in datasets.values())
                }
            
            else:
                return {
                    "llm": llm_name,
                    "error": f"{llm_name} not available",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "llm": llm_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_comprehensive_analysis(self):
        """Run complete fraud pattern analysis"""
        print("🚀 BIGQUERY LLM FRAUD ANALYSIS")
        print("=" * 35)
        
        # Extract data
        datasets = self.extract_fraud_samples()
        
        if not datasets:
            print("❌ No datasets extracted. Check BigQuery access.")
            return
        
        print(f"\n📊 Successfully extracted {len(datasets)} datasets:")
        for name, df in datasets.items():
            fraud_count = 0
            if 'isFraud' in df.columns:
                fraud_count = df['isFraud'].sum()
            elif 'Class' in df.columns:
                fraud_count = df['Class'].sum()
            elif 'FraudIndicator' in df.columns:
                fraud_count = df['FraudIndicator'].notna().sum()
            
            print(f"  - {name}: {len(df)} samples ({fraud_count} fraud cases)")
        
        # Save extracted data
        dataset_summary = {}
        for name, df in datasets.items():
            dataset_summary[name] = df.to_dict('records')
        
        with open('../data/bigquery_fraud_samples.json', 'w') as f:
            json.dump({
                'extraction_timestamp': datetime.now().isoformat(),
                'total_datasets': len(datasets),
                'datasets': dataset_summary
            }, f, indent=2, default=str)
        
        print("💾 Raw data saved to: ../data/bigquery_fraud_samples.json")
        
        # Analyze with available LLMs
        llm_tasks = []
        
        if self.gemini_model:
            llm_tasks.append(self.analyze_fraud_patterns_with_llm(datasets, "gemini"))
        
        if self.openai_client:
            llm_tasks.append(self.analyze_fraud_patterns_with_llm(datasets, "openai"))
        
        if not llm_tasks:
            print("❌ No LLMs available for analysis")
            return
        
        # Run LLM analyses
        print(f"\n🤖 Running analysis with {len(llm_tasks)} LLM(s)...")
        results = await asyncio.gather(*llm_tasks)
        
        # Compile results
        final_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'bigquery_project': 'vaulted-timing-473322-f9',
            'datasets_analyzed': list(datasets.keys()),
            'total_samples_analyzed': sum(len(df) for df in datasets.values()),
            'llm_analyses': results,
            'data_summary': {
                name: {
                    'total_records': len(df),
                    'columns': list(df.columns),
                    'sample_record': df.iloc[0].to_dict() if len(df) > 0 else {}
                }
                for name, df in datasets.items()
            }
        }
        
        # Save results
        output_file = '../data/bigquery_llm_fraud_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print("=" * 25)
        print(f"📁 Results saved to: {output_file}")
        print(f"📊 Analyzed {sum(len(df) for df in datasets.values())} fraud samples")
        print(f"🤖 Used {len([r for r in results if 'error' not in r])} LLM(s)")
        
        # Show summary
        for result in results:
            if 'error' not in result:
                print(f"✅ {result['llm']}: Analysis completed")
                print(f"   📝 Analysis length: {len(result.get('analysis', ''))} characters")
            else:
                print(f"❌ {result['llm']}: {result['error']}")
        
        return final_results

def main():
    """Main function"""
    analyzer = BigQueryFraudAnalyzer()
    return asyncio.run(analyzer.run_comprehensive_analysis())

if __name__ == "__main__":
    main()