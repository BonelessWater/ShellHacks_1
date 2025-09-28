#!/usr/bin/env python3
"""Check GOOGLE_API_KEY* environment variables by making a minimal
request to Google's Generative Language REST endpoint.

This script does NOT print or store raw keys. It reports the variable
name and whether the key authenticated (OK) or failed (FAIL).
"""
import os
import json
import sys
import urllib.request
import urllib.error

TEST_MODEL = "models/gemini-2.5-flash"
ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={key}"


def test_key(key: str) -> (bool, str):
    url = ENDPOINT_TEMPLATE.format(model=TEST_MODEL, key=key)
    body = json.dumps({
        "contents": [{"parts": [{"text": "Hello"}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1
        }
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp_body = resp.read().decode("utf-8")
            try:
                j = json.loads(resp_body)
                # success if we get some generated text or candidates
                if isinstance(j, dict) and ("candidates" in j or "output" in j or "candidates" in j.get("candidates", {})):
                    return True, "OK"
                # Some responses include 'candidates' or 'output', treat any 200 as OK
                return True, "OK"
            except Exception:
                return True, "OK"
    except urllib.error.HTTPError as e:
        try:
            payload = e.read().decode('utf-8')
        except Exception:
            payload = ''
        return False, f"HTTP {e.code} - {str(e)} {payload[:200]}"
    except Exception as e:
        return False, str(e)


def find_keys():
    names = []
    # Common names: GOOGLE_API_KEY, GOOGLE_API_KEY_0..9, GOOGLE_API_KEY_1..9
    if os.getenv("GOOGLE_API_KEY"):
        names.append("GOOGLE_API_KEY")
    for i in range(0, 10):
        n = f"GOOGLE_API_KEY_{i}"
        if os.getenv(n):
            names.append(n)
    for i in range(1, 10):
        n = f"GOOGLE_API_KEY{i}"
        if os.getenv(n) and n not in names:
            names.append(n)
    return names


def main():
    found = find_keys()
    results = {}
    if not found:
        print("No GOOGLE_API_KEY* environment variables found")
        return 1

    for name in found:
        key = os.getenv(name)
        print(f"Testing {name}...")
        ok, info = test_key(key)
        results[name] = {"ok": bool(ok), "info": info}
        print(f"  {name}: {'OK' if ok else 'FAIL'} - {info}")

    # Write summary (do not store raw keys)
    try:
        with open('.key_check.json', 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2)
    except Exception:
        pass

    # Print concise summary
    ok_count = sum(1 for v in results.values() if v['ok'])
    print(f"\nSummary: {ok_count}/{len(results)} keys OK")
    return 0


if __name__ == '__main__':
    sys.exit(main())
