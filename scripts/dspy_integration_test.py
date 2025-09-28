#!/usr/bin/env python3
"""
Integration Testing Script for DSPy Agent System
Run this to validate your current setup and identify issues
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("integration_test")


async def test_dspy_integration():
    """Test DSPy integration with your agents"""
    print("🔍 Testing DSPy Integration...")
    
    try:
        # Test 1: Import check
        from backend.dspy_signatures import module_manager
        from backend.main_pipeline import get_pipeline
        from backend.agents import create_agent_coordinator
        
        print("✅ All imports successful")
        
        # Test 2: DSPy module initialization
        if module_manager.initialize_modules():
            print("✅ DSPy modules initialized")
            print(f"   Available modules: {list(module_manager.modules.keys())}")
        else:
            print("❌ DSPy module initialization failed")
            return False
        
        # Test 3: Agent coordinator
        coordinator = create_agent_coordinator()
        status = coordinator.get_agent_status()
        print(f"✅ Agent coordinator status: {status['coordinator_status']}")
        
        # Test 4: Pipeline integration
        pipeline = get_pipeline()
        system_status = pipeline.get_system_status()
        print(f"✅ Pipeline ready: {system_status['pipeline_ready']}")
        print(f"   LLM working: {system_status['llm_working']}")
        print(f"   Available API keys: {system_status['available_api_keys']}")
        
        # Test 5: End-to-end test with sample data
        test_invoice = {
            "invoice_id": "TEST-001",
            "vendor": "Test Vendor Inc",
            "date": "2024-01-15",
            "items": [
                {"description": "Test Service", "quantity": 1, "unit_price": 100.0}
            ],
            "total": 100.0
        }
        
        print("🧪 Running end-to-end test...")
        result = pipeline.run_detection(test_invoice, max_iterations=2)
        
        print(f"✅ End-to-end test completed")
        print(f"   Status: {result.status}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   Fraud risk: {result.summary.get('fraud_risk', 'Unknown')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 Testing API Endpoints...")
    
    try:
        from backend.api.index import handler
        
        # Test health check
        health_request = {
            'method': 'GET',
            'path': '/health'
        }
        
        response = handler(health_request)
        print(f"✅ Health endpoint: {response.get('statusCode', 'Unknown')}")
        
        # Test fraud detection endpoint
        fraud_request = {
            'method': 'POST',
            'path': '/detect',
            'body': json.dumps({
                "invoice_id": "API-TEST-001",
                "vendor": "API Test Vendor",
                "total": 150.0,
                "items": [{"description": "API Test", "quantity": 1, "unit_price": 150.0}]
            })
        }
        
        response = handler(fraud_request)
        print(f"✅ Fraud detection endpoint: {response.get('statusCode', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def identify_missing_components():
    """Identify what components need to be completed"""
    print("\n🔧 Identifying Missing Components...")
    
    # Load .env file first
    import pathlib
    repo_root = pathlib.Path('.')
    env_path = repo_root / '.env'
    if env_path.exists():
        for ln in env_path.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith('#') or '=' not in ln:
                continue
            k, v = ln.split('=', 1)
            k = k.strip(); v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            os.environ[k] = v
    
    missing = []
    issues = []
    
    # Check for environment variables / API keys using the APIKeyManager
    try:
        # Try to import and use APIKeyManager to check for working keys
        import importlib.util
        spec = importlib.util.spec_from_file_location('llm', 'backend/llm.py')
        llm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llm_module)
        
        manager = llm_module.APIKeyManager()
        working_key = manager.get_current_key()
        if not working_key:
            missing.append("No working Google API Key found (tried validation)")
        else:
            print(f"✅ Working Google API Key found: {working_key[:10]}...")
    except Exception as e:
        # Fallback to simple env var check
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GOOGLE_API_KEY_0"):
            missing.append("Google API Key not set (and validation check failed)")
        print(f"⚠️  Could not validate API key: {e}")
    
    # Check for required files
    required_files = [
        "backend/main_pipeline.py",
        "backend/agents.py", 
        "backend/dspy_signatures.py",
        "backend/fraud_loop.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(f"Missing file: {file_path}")
    
    if missing:
        print("❌ Missing components:")
        for item in missing:
            print(f"   - {item}")
    else:
        print("✅ All required components present")
    
    return missing


async def main():
    """Run comprehensive integration test"""
    print("🚀 DSPy Agent System Integration Test")
    print("=" * 50)
    
    # Step 1: Check missing components
    missing = identify_missing_components()
    if missing:
        print(f"\n⚠️  Please fix {len(missing)} missing components first")
        return
    
    # Step 2: Test DSPy integration
    dspy_ok = await test_dspy_integration()
    
    # Step 3: Test API endpoints
    api_ok = await test_api_endpoints()
    
    # Summary
    print("\n" + "=" * 50)
    if dspy_ok and api_ok:
        print("✅ All tests passed! System ready for production")
        print("\n📋 Recommended next steps:")
        print("   1. Deploy to Azure/Vercel")
        print("   2. Set up monitoring")
        print("   3. Load test with real data")
        print("   4. Implement additional agents")
    else:
        print("❌ Some tests failed. Please fix issues before proceeding")
        print("\n🔧 Troubleshooting:")
        if not dspy_ok:
            print("   - Check DSPy configuration and API keys")
            print("   - Verify agent imports")
        if not api_ok:
            print("   - Check API endpoint configuration")
            print("   - Verify request handling")


if __name__ == "__main__":
    asyncio.run(main())
