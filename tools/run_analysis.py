#!/usr/bin/env python3
"""
Simple runner script for LLM Fraud Analyzer
This script provides an easy way to run the fraud analysis with sensible defaults.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'openai': 'openai',
        'google-generativeai': 'google.generativeai'
    }
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("   OR")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_api_keys():
    """Check if API keys are configured"""
    api_keys = {
        'OPENAI_API_KEY': 'OpenAI (ChatGPT)',
        'GOOGLE_API_KEY': 'Google (Gemini)'
    }
    
    configured_keys = []
    missing_keys = []
    
    for key, service in api_keys.items():
        if os.getenv(key):
            configured_keys.append(service)
        else:
            missing_keys.append(f"{key} ({service})")
    
    if configured_keys:
        print(f"✅ API keys configured for: {', '.join(configured_keys)}")
    
    if missing_keys:
        print("⚠️  Missing API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n🔑 To add API keys:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your API keys to the .env file")
        print("   3. Or set environment variables directly")
    
    return len(configured_keys) > 0

def run_analysis():
    """Run the fraud analysis"""
    script_dir = Path(__file__).parent
    analyzer_script = script_dir / "llm_fraud_analyzer.py"
    
    print("🚀 Starting LLM Fraud Analysis...")
    print("=" * 50)
    
    try:
        # Run the analyzer script
        result = subprocess.run([
            sys.executable, 
            str(analyzer_script),
            "--verbose"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Analysis completed successfully!")
            print(f"📄 Results saved to: data/llm_fraud_analysis_results.json")
        else:
            print("❌ Analysis failed. Check the logs above for details.")
            return False
    
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🔍 LLM Fraud Pattern Analyzer")
    print("=" * 40)
    
    # Load environment variables from .env file if it exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print("✅ Loaded environment variables from .env file")
        except ImportError:
            print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
            print("   You can still use environment variables set directly in your shell")
    
    # Check requirements
    if not check_requirements():
        return 1
    
    print()
    
    # Check API keys
    if not check_api_keys():
        print("\n❌ No API keys configured. The analysis will not be able to query any LLMs.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    print()
    
    # Run the analysis
    if run_analysis():
        print("\n🎉 Analysis complete! Check the results file for detailed outputs.")
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())