#!/usr/bin/env python3
"""
Check Available Ollama Models
"""

import requests
import json

def check_available_models():
    """Check what models are actually available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            print("📋 AVAILABLE OLLAMA MODELS:")
            print("=" * 35)
            for model in models:
                print(f"🤖 {model['name']}")
                print(f"   📊 Size: {model.get('size', 'Unknown')}")
                print(f"   📅 Modified: {model.get('modified_at', 'Unknown')}")
                print()
            
            return [m['name'] for m in models]
        else:
            print(f"❌ HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    available = check_available_models()
    print(f"Total models: {len(available)}")
    for model in available:
        print(f"  - {model}")