#!/usr/bin/env python3
"""
Multi-key sanity check for Google Generative AI (consumer API).

Checks, in order:
  GOOGLE_API_KEY
  GOOGLE_API_KEY_0 .. GOOGLE_API_KEY_9

For the first working key:
  - prints SDK version
  - lists models that support generateContent (shows first few)
  - runs a tiny generation on a preferred flash model

Exit codes:
 0 = success
 1 = no keys found
 2 = SDK/config/import problem
 3 = no usable models or generation failed for all keys
"""

import os
import sys
import logging
from typing import List
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_gemini_keys")

PREFERRED_MODELS = [
    "models/gemini-2.5-flash",  # fast/cheaper
    "models/gemini-2.5-pro",    # stronger
]

def _parse_version(v: str) -> List[int]:
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except Exception:
            break
    return parts or [0, 0, 0]

def _gather_keys() -> List[str]:
    load_dotenv()
    keys: List[str] = []

    # Plain key first
    main_key = os.getenv("GOOGLE_API_KEY")
    if main_key and main_key.strip():
        keys.append(main_key.strip())
        log.info("✅ Found GOOGLE_API_KEY")

    # Numbered keys next
    for i in range(0, 10):
        k = os.getenv(f"GOOGLE_API_KEY_{i}")
        if k and k.strip():
            keys.append(k.strip())
            log.info(f"✅ Found GOOGLE_API_KEY_{i}")

    return keys

def try_key_detailed(api_key: str) -> dict:
    """Enhanced version that returns detailed results instead of just True/False"""
    result = {
        "success": False,
        "sdk_version": "unknown",
        "models_found": [],
        "generation_worked": False,
        "generation_response": "",
        "error_type": None,
        "error_message": "",
        "recommendations": []
    }
    
    try:
        import google.generativeai as genai
    except Exception as e:
        result["error_type"] = "import_error"
        result["error_message"] = str(e)
        result["recommendations"].append("Install google-generativeai: poetry add google-generativeai@latest")
        return result

    # Version check
    sdk_ver = getattr(genai, "__version__", "unknown")
    result["sdk_version"] = sdk_ver
    if _parse_version(sdk_ver) < [0, 7, 0]:
        result["recommendations"].append("Update SDK: poetry add google-generativeai@latest")

    # Configure client
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        result["error_type"] = "config_error"
        result["error_message"] = str(e)
        result["recommendations"].append("Check API key format and validity")
        return result

    # List models
    try:
        usable = []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                usable.append(m.name)
        
        result["models_found"] = usable
        if not usable:
            result["error_type"] = "no_models"
            result["error_message"] = "No models with generateContent found"
            result["recommendations"].extend([
                "Check if using correct API (consumer vs Vertex)",
                "Verify account has access to Gemini models",
                "Check for org/billing restrictions"
            ])
            return result
            
    except Exception as e:
        result["error_type"] = "list_models_error"
        result["error_message"] = str(e)
        if "v1beta" in str(e):
            result["recommendations"].append("Update SDK to fix v1beta errors")
        return result

    # Test generation with more interesting prompt
    target = next((p for p in PREFERRED_MODELS if p in usable), usable[0])
    try:
        model = genai.GenerativeModel(target)
        # More engaging test prompt
        test_prompt = "Who do you think was the best US president and why? Give a brief 2-3 sentence answer."
        resp = model.generate_content(test_prompt)
        text = getattr(resp, "text", "") or ""
        
        result["generation_worked"] = True
        result["generation_response"] = text.strip()
        result["success"] = True
        result["test_prompt"] = test_prompt
        
    except Exception as e:
        result["error_type"] = "generation_error"
        result["error_message"] = str(e)
        
        if "500" in str(e) or "Internal error" in str(e):
            result["recommendations"].extend([
                "Google API temporary issue - retry later",
                "Check API quota/billing status",
                "Try different model name"
            ])
        elif "PermissionDenied" in str(e):
            result["recommendations"].append("Check API key permissions and billing")
        elif "NotFound" in str(e):
            result["recommendations"].append("Use a model from the available list")
        elif "v1beta" in str(e):
            result["recommendations"].append("Update SDK to fix v1beta model name issues")
    
    return result
def try_key(api_key: str) -> bool:
    """Original try_key function for backward compatibility"""
    detailed_result = try_key_detailed(api_key)
    
    # Log detailed information
    log.info(f"SDK version: {detailed_result['sdk_version']}")
    
    if detailed_result["error_type"]:
        log.error(f"Error type: {detailed_result['error_type']}")
        log.error(f"Error: {detailed_result['error_message']}")
        for rec in detailed_result["recommendations"]:
            log.warning(f"💡 {rec}")
        return False
    
    if detailed_result["models_found"]:
        log.info("Models supporting generateContent (first few):")
        for name in detailed_result["models_found"][:8]:
            log.info(f"  - {name}")
    
    if detailed_result["generation_worked"]:
        log.info(f"Test prompt: {detailed_result.get('test_prompt', 'N/A')}")
        log.info(f"Generation OK. Reply: {detailed_result['generation_response']!r}")
        print("\n✅ SUCCESS: This API key works and generation succeeded.")
        return True
    
    return False

def test_all_keys_systematically() -> dict:
    """Test all keys systematically and return detailed results"""
    keys = _gather_keys()
    if not keys:
        log.error("❌ No API keys found. Add GOOGLE_API_KEY and/or GOOGLE_API_KEY_0..9 to your .env.")
        return {"status": "no_keys", "exit_code": 1, "results": []}

    results = []
    working_keys = 0
    
    for idx, key in enumerate(keys):
        key_name = "GOOGLE_API_KEY" if idx == 0 else f"GOOGLE_API_KEY_{idx-1 if idx > 0 else idx}"
        log.info(f"--- Testing {key_name} (key #{idx}) ---")
        
        key_result = {
            "key_name": key_name,
            "key_index": idx,
            "key_preview": f"{key[:8]}...{key[-8:]}" if len(key) > 16 else "short_key",
            "status": "unknown",
            "error": None,
            "models_available": [],
            "generation_test": False,
        "generation_response": "",
        "test_prompt": ""
        }
        
        try:
            detailed_result = try_key_detailed(key)
            key_result.update({
                "status": "success" if detailed_result["success"] else "failed",
                "error": detailed_result["error_message"] if detailed_result["error_type"] else None,
                "error_type": detailed_result["error_type"],
                "models_available": detailed_result["models_found"],
                "generation_test": detailed_result["generation_worked"],
                "generation_response": detailed_result["generation_response"],
                "sdk_version": detailed_result["sdk_version"],
                "recommendations": detailed_result["recommendations"]
            })
            
            if detailed_result["success"]:
                working_keys += 1
                log.info(f"✅ {key_name} WORKS")
                log.info(f"   📝 Prompt: {detailed_result.get('test_prompt', 'N/A')}")
                log.info(f"   🤖 Response: {detailed_result['generation_response'][:100]}{'...' if len(detailed_result['generation_response']) > 100 else ''}")
            else:
                log.error(f"❌ {key_name} FAILED - {detailed_result['error_type']}: {detailed_result['error_message']}")
                for rec in detailed_result["recommendations"]:
                    log.warning(f"   💡 {rec}")
        except Exception as e:
            key_result["status"] = "error"
            key_result["error"] = str(e)
            log.error(f"❌ {key_name} ERROR: {e}")
        
        results.append(key_result)
        log.info("-" * 50)
    
    # Summary
    total_keys = len(keys)
    failed_keys = total_keys - working_keys
    
    summary = {
        "total_keys_found": total_keys,
        "working_keys": working_keys,
        "failed_keys": failed_keys,
        "success_rate": f"{(working_keys/total_keys)*100:.1f}%" if total_keys > 0 else "0%",
        "overall_status": "success" if working_keys > 0 else "all_failed"
    }
    
    log.info("=" * 60)
    log.info("🔍 SYSTEMATIC TEST SUMMARY")
    log.info("=" * 60)
    log.info(f"📊 Total keys tested: {total_keys}")
    log.info(f"✅ Working keys: {working_keys}")
    log.info(f"❌ Failed keys: {failed_keys}")
    log.info(f"📈 Success rate: {summary['success_rate']}")
    
    for result in results:
        status_emoji = "✅" if result["status"] == "success" else "❌"
        log.info(f"{status_emoji} {result['key_name']}: {result['status'].upper()}")
        if result["error"]:
            log.info(f"   Error: {result['error']}")
    
    exit_code = 0 if working_keys > 0 else (1 if total_keys == 0 else 3)
    
    return {
        "status": summary["overall_status"],
        "exit_code": exit_code,
        "summary": summary,
        "results": results
    }

def main() -> int:
    """Main function - can run systematic test or original behavior"""
    import sys
    
    # Check if systematic testing is requested
    if "--systematic" in sys.argv:
        result = test_all_keys_systematically()
        return result["exit_code"]
    
    # Original behavior - stop at first working key
    keys = _gather_keys()
    if not keys:
        log.error("❌ No API keys found. Add GOOGLE_API_KEY and/or GOOGLE_API_KEY_0..9 to your .env.")
        return 1

    for idx, key in enumerate(keys):
        log.info(f"--- Trying key #{idx} ---")
        if try_key(key):
            return 0

    log.error("❌ All keys tried; none succeeded.")
    return 3

if __name__ == "__main__":
    sys.exit(main())