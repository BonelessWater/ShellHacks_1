#!/usr/bin/env python3
"""
Simple results viewer for the fraud analysis
"""
import json
import os

def main():
    results_path = os.path.join("data", "llm_fraud_analysis_results.json")
    
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        print("🔍 FRAUD ANALYSIS RESULTS SUMMARY")
        print("=" * 40)
        
        metadata = data["analysis_metadata"]
        print(f"📊 Dataset Overview:")
        print(f"   Total invoices analyzed: {metadata['total_invoices_analyzed']}")
        print(f"   Fraudulent invoices: {metadata['fraudulent_invoices_count']}")    
        print(f"   Legitimate invoices: {metadata['legitimate_invoices_count']}")
        print(f"   Analysis reports: {metadata['llm_responses_count']}")
        print(f"   Analysis engine: {metadata['analysis_engine']}")
        
        summary = data["summary_analysis"]
        print(f"\n🎯 Analysis Results:")
        print(f"   Unique fraud patterns found: {len(summary['fraud_patterns_frequency'])}")
        print(f"   Average analysis time: {summary['average_analysis_time_seconds']:.1f} seconds")
        
        print(f"\n🚨 Top 5 Most Common Fraud Patterns:")
        for i, (pattern, count) in enumerate(summary["most_common_patterns"][:5], 1):
            print(f"   {i}. {pattern.replace('_', ' ').title()}: {count} occurrences")
        
        print(f"\n📄 Reports Generated:")
        for response in data["llm_responses"]:
            print(f"   - {response['model_name']}: {response['prompt_type']}")
            print(f"     Confidence: {response['confidence_scores']['overall']:.2f}")
            print(f"     Patterns identified: {len(response['fraud_patterns_identified'])}")
        
        print(f"\n✅ Full results saved in: {results_path}")
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")

if __name__ == "__main__":
    main()