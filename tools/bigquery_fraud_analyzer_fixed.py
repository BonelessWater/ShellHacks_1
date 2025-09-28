#!/usr/bin/env python3
"""
Fixed BigQuery-powered LLM Fraud Pattern Analyzer
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
        
        # 1. IEEE CIS Fraud - Get fraud samples first
        print("📊 Extracting IEEE CIS fraud data...")
        
        # First get fraud cases
        ieee_fraud_query = """
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
        WHERE isFraud = 1
        LIMIT 25
        """
        
        # Then get normal cases
        ieee_normal_query = """
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
        WHERE isFraud = 0
        ORDER BY RAND()
        LIMIT 25
        """
        
        try:
            fraud_df = self.bq_client.query(ieee_fraud_query).to_dataframe()
            normal_df = self.bq_client.query(ieee_normal_query).to_dataframe()
            ieee_df = pd.concat([fraud_df, normal_df], ignore_index=True)
            datasets['ieee_cis_fraud'] = ieee_df
            print(f"  ✅ Extracted {len(ieee_df)} IEEE CIS transactions ({len(fraud_df)} fraud, {len(normal_df)} normal)")
        except Exception as e:
            print(f"  ❌ IEEE extraction failed: {e}")
        
        # 2. Credit Card Fraud
        print("💳 Extracting credit card fraud data...")
        
        cc_fraud_query = """
        SELECT 
            Time,
            V1, V2, V3, V4, V5,
            Amount,
            Class
        FROM `vaulted-timing-473322-f9.transactional_fraud.credit_card_fraud`
        WHERE Class = 1
        LIMIT 25
        """
        
        cc_normal_query = """
        SELECT 
            Time,
            V1, V2, V3, V4, V5,
            Amount,
            Class
        FROM `vaulted-timing-473322-f9.transactional_fraud.credit_card_fraud`
        WHERE Class = 0
        ORDER BY RAND()
        LIMIT 25
        """
        
        try:
            cc_fraud_df = self.bq_client.query(cc_fraud_query).to_dataframe()
            cc_normal_df = self.bq_client.query(cc_normal_query).to_dataframe()
            cc_df = pd.concat([cc_fraud_df, cc_normal_df], ignore_index=True)
            datasets['credit_card_fraud'] = cc_df
            print(f"  ✅ Extracted {len(cc_df)} credit card transactions ({len(cc_fraud_df)} fraud, {len(cc_normal_df)} normal)")
        except Exception as e:
            print(f"  ❌ Credit card extraction failed: {e}")
        
        # 3. PaySim Mobile Money Fraud
        print("📱 Extracting PaySim mobile money fraud...")
        
        paysim_fraud_query = """
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
        WHERE isFraud = 1
        LIMIT 20
        """
        
        paysim_normal_query = """
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
        WHERE isFraud = 0
        ORDER BY RAND()
        LIMIT 20
        """
        
        try:
            paysim_fraud_df = self.bq_client.query(paysim_fraud_query).to_dataframe()
            paysim_normal_df = self.bq_client.query(paysim_normal_query).to_dataframe()
            paysim_df = pd.concat([paysim_fraud_df, paysim_normal_df], ignore_index=True)
            datasets['paysim_fraud'] = paysim_df
            print(f"  ✅ Extracted {len(paysim_df)} PaySim transactions ({len(paysim_fraud_df)} fraud, {len(paysim_normal_df)} normal)")
        except Exception as e:
            print(f"  ❌ PaySim extraction failed: {e}")
        
        # 4. Relational Fraud Data - Simplified query
        print("🔗 Extracting relational fraud indicators...")
        relational_query = """
        SELECT 
            TransactionID,
            FraudIndicator
        FROM `vaulted-timing-473322-f9.fraud_detection_relational.fraudulent_patterns_fraud_indicators`
        WHERE FraudIndicator IS NOT NULL
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
        
        # Create detailed dataset summary for LLM
        dataset_descriptions = []
        total_fraud_cases = 0
        total_normal_cases = 0
        
        for name, df in datasets.items():
            fraud_count = 0
            normal_count = 0
            
            if 'isFraud' in df.columns:
                fraud_count = int(df['isFraud'].sum())
                normal_count = int((df['isFraud'] == 0).sum())
            elif 'Class' in df.columns:
                fraud_count = int(df['Class'].sum())
                normal_count = int((df['Class'] == 0).sum())
            elif 'FraudIndicator' in df.columns:
                fraud_count = int(df['FraudIndicator'].notna().sum())
            
            total_fraud_cases += fraud_count
            total_normal_cases += normal_count
            
            # Get sample of actual data
            sample_data = df.head(3).to_dict('records')
            
            dataset_descriptions.append(f"""
**{name.upper()}** ({len(df)} records):
- Fraud cases: {fraud_count}
- Normal cases: {normal_count}
- Key columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}
- Sample data: {json.dumps(sample_data, default=str, indent=2)[:500]}...
            """)
        
        analysis_prompt = f"""
You are a world-class fraud detection expert analyzing REAL financial fraud datasets from Google BigQuery.

**DATASETS ANALYZED:**
{chr(10).join(dataset_descriptions)}

**TOTAL SCOPE:**
- {total_fraud_cases} confirmed fraud cases
- {total_normal_cases} normal transactions  
- {len(datasets)} different fraud detection datasets
- Real-world data from IEEE CIS competition, credit card companies, and mobile payment systems

**YOUR EXPERT ANALYSIS TASK:**

Please provide a comprehensive fraud pattern analysis covering:

## 1. 🎯 TOP FRAUD PATTERNS IDENTIFIED
What are the 5 most critical fraud patterns you observe across these datasets?

## 2. 💰 TRANSACTION AMOUNT ANALYSIS  
What patterns do you see in transaction amounts for fraud vs legitimate transactions?

## 3. 🕒 TEMPORAL PATTERNS
Any time-based patterns in fraudulent activity?

## 4. 🏦 CARD/PAYMENT METHOD PATTERNS
What payment method characteristics indicate higher fraud risk?

## 5. 🌐 GEOGRAPHIC/DOMAIN PATTERNS
Any location or email domain patterns in fraud?

## 6. 🚨 HIGH-RISK INDICATORS
What are the strongest predictive features for fraud detection?

## 7. 🛡️ PREVENTION STRATEGIES
Based on these patterns, what are your top 5 fraud prevention recommendations?

## 8. 🎢 RISK SCORING MODEL
How would you design a risk scoring system using these insights?

**Please focus on actionable, data-driven insights that can be implemented in a real fraud detection system.**
        """
        
        try:
            if llm_name == "gemini" and self.gemini_model:
                response = self.gemini_model.generate_content(analysis_prompt)
                return {
                    "llm": "Google Gemini 2.0 Flash",
                    "analysis": response.text,
                    "timestamp": datetime.now().isoformat(),
                    "datasets_analyzed": list(datasets.keys()),
                    "total_samples": sum(len(df) for df in datasets.values()),
                    "fraud_cases": total_fraud_cases,
                    "normal_cases": total_normal_cases
                }
            
            elif llm_name == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a world-class fraud detection expert with deep experience in financial crime analysis."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.2
                )
                
                return {
                    "llm": "OpenAI GPT-4o Mini",
                    "analysis": response.choices[0].message.content,
                    "timestamp": datetime.now().isoformat(),
                    "datasets_analyzed": list(datasets.keys()),
                    "total_samples": sum(len(df) for df in datasets.values()),
                    "fraud_cases": total_fraud_cases,
                    "normal_cases": total_normal_cases
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
        total_fraud = 0
        total_normal = 0
        
        for name, df in datasets.items():
            fraud_count = 0
            normal_count = 0
            
            if 'isFraud' in df.columns:
                fraud_count = int(df['isFraud'].sum())
                normal_count = int((df['isFraud'] == 0).sum())
            elif 'Class' in df.columns:
                fraud_count = int(df['Class'].sum())
                normal_count = int((df['Class'] == 0).sum())
            elif 'FraudIndicator' in df.columns:
                fraud_count = int(df['FraudIndicator'].notna().sum())
            
            total_fraud += fraud_count
            total_normal += normal_count
            
            print(f"  - {name}: {len(df)} samples ({fraud_count} fraud, {normal_count} normal)")
        
        print(f"\n🎯 TOTAL SCOPE: {total_fraud + total_normal} transactions ({total_fraud} fraud, {total_normal} legitimate)")
        
        # Save extracted data
        dataset_summary = {}
        for name, df in datasets.items():
            dataset_summary[name] = df.to_dict('records')
        
        with open('../data/bigquery_fraud_samples.json', 'w') as f:
            json.dump({
                'extraction_timestamp': datetime.now().isoformat(),
                'total_datasets': len(datasets),
                'total_fraud_cases': total_fraud,
                'total_normal_cases': total_normal,
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
            'total_fraud_cases': total_fraud,
            'total_normal_cases': total_normal,
            'llm_analyses': results,
            'data_summary': {
                name: {
                    'total_records': len(df),
                    'columns': list(df.columns),
                    'fraud_cases': int(df['isFraud'].sum()) if 'isFraud' in df.columns else (int(df['Class'].sum()) if 'Class' in df.columns else 0),
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
        print(f"🎯 {total_fraud} fraud cases vs {total_normal} legitimate transactions")
        print(f"🤖 Used {len([r for r in results if 'error' not in r])} LLM(s)")
        
        # Show summary of analyses
        for result in results:
            if 'error' not in result:
                print(f"✅ {result['llm']}: Analysis completed")
                print(f"   📝 Analysis length: {len(result.get('analysis', ''))} characters")
                # Show first few lines of analysis
                analysis_preview = result.get('analysis', '')[:200] + "..."
                print(f"   🔍 Preview: {analysis_preview}")
            else:
                print(f"❌ {result['llm']}: {result['error']}")
        
        return final_results

def main():
    """Main function"""
    analyzer = BigQueryFraudAnalyzer()
    return asyncio.run(analyzer.run_comprehensive_analysis())

if __name__ == "__main__":
    main()