#!/usr/bin/env python3
"""
Enhanced Gemini LLM module with multi-key support and improved error handling
"""

import logging
import os
import random
import re
import time
from typing import Optional

# Make heavy external imports optional so the module can be imported in
# lightweight test/CI environments without those packages installed.
import types

try:
    import dspy
except Exception:  # pragma: no cover - fallback for missing dependency
    # Provide a minimal fallback so classes that inherit dspy.LM can be
    # defined without the real package present.
    class _DspyFallbackLM:
        pass

    dspy = types.SimpleNamespace(LM=_DspyFallbackLM)


def _import_genai():
    """Try to import google.generativeai; return module or None."""
    try:
        import google.generativeai as genai_mod  # type: ignore

        return genai_mod
    except Exception:
        return None

# Configuration constants
MODEL_CANDIDATES = [
    "models/gemini-2.5-flash",  # fast/cheaper
    "models/gemini-2.5-pro",  # stronger
]


MIN_DELAY_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0

log = logging.getLogger("fraud_detection_llm")


class APIKeyManager:
    """Manages multiple Google API keys for load balancing"""

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.failed_keys = set()

        # Perform a quick validation pass to prioritize working keys (best-effort)
        if self.api_keys:
            try:
                self.prioritize_working_keys()
            except Exception:
                # don't fail initialization if validation encounters issues
                pass

        if not self.api_keys:
            raise RuntimeError(
                "No valid GOOGLE_API_KEY found. Please set GOOGLE_API_KEY_1, "
                "GOOGLE_API_KEY_2, or GOOGLE_API_KEY_3"
            )

        log.info(f"Loaded {len(self.api_keys)} API keys")

    def _load_api_keys(self) -> list:
        """Load all available API keys from environment"""
        keys = []

        # Look for common API key env vars: GOOGLE_API_KEY, GOOGLE_API_KEY_0..9,
        # GOOGLE_API_KEYNAME variants and keys with 'BIGQUERY' or similar suffixes.
        seen = set()

        # Direct canonical name
        v = os.getenv("GOOGLE_API_KEY")
        if v and v.strip():
            kv = v.strip()
            keys.append(kv)
            seen.add(kv)
            log.info("Found GOOGLE_API_KEY")

        # Numbered variants
        for i in range(0, 10):
            name = f"GOOGLE_API_KEY_{i}"
            k = os.getenv(name)
            if k and k.strip():
                kv = k.strip()
                if kv not in seen:
                    keys.append(kv)
                    seen.add(kv)
                    log.info(f"Found {name}")

        # Other variants like GOOGLE_API_KEY_<NAME> and any env var that looks
        # like an API key name. This permits user-created env vars such as
        # MY_PROJECT_GENAI_KEY, ILAN_GOOGLE_KEY, or PRODUCTION_GCP_API_KEY.
        flexible_keywords = ("API_KEY", "GOOGLE", "GENAI", "GCP", "KEY", "BIGQUERY", "DOCUMENT")
        for kname, kval in os.environ.items():
            if not kval or not kval.strip():
                continue
            up = kname.upper()
            # Skip ones we've already examined explicitly above
            if up in ("GOOGLE_API_KEY",) or up.startswith("GOOGLE_API_KEY_"):
                continue

            if any(kw in up for kw in flexible_keywords):
                kv = kval.strip()
                if kv not in seen:
                    keys.append(kv)
                    seen.add(kv)
                    log.info(f"Found {kname}")

        return keys

    def prioritize_working_keys(self, timeout: float = 8.0):
        """Validate keys and move known-working keys to the front of the list.

        This is helpful when env var names are unpredictable: run a quick
        validation pass and reorder so get_current_key() will find working
        keys faster.
        """
        working = []
        failed = []

        for key in list(self.api_keys):
            try:
                if self._validate_key(key):
                    working.append(key)
                else:
                    failed.append(key)
            except Exception:
                failed.append(key)

        # reorder: working keys first, then failed ones
        if working:
            self.api_keys = working + [k for k in self.api_keys if k not in working]
            log.info(f"Prioritized {len(working)} working API key(s)")

    def get_current_key(self) -> Optional[str]:
        """Get the current API key"""
        if not self.api_keys:
            return None
        # Find next candidate key that hasn't been validated as failed
        attempts = 0
        while attempts < len(self.api_keys):
            key = self.api_keys[self.current_key_index]

            if key in self.failed_keys:
                # rotate
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                attempts += 1
                continue

            # lazily validate key using a small test (best-effort)
            try:
                if self._validate_key(key):
                    log.debug(f"Using API key #{self.current_key_index + 1}")
                    return key
                else:
                    self.failed_keys.add(key)
                    log.warning(f"Validation failed for API key #{self.current_key_index + 1}")
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    attempts += 1
                    continue
            except Exception as e:
                log.warning(f"Error validating API key: {e}")
                self.failed_keys.add(key)
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                attempts += 1
                continue

        # All keys invalid or failed
        if self.failed_keys and len(self.failed_keys) >= len(self.api_keys):
            log.warning("All keys failed or invalid")
            return None

        return None

    def _validate_key(self, key: str) -> bool:
        """Best-effort validation: configure genai with the key and try to
        instantiate a small GenerativeModel. Returns True if appears valid.

        This is intentionally lightweight and may still fail under quota.
        """
        # First try the official SDK if available
        try:
            genai = _import_genai()
            if genai is not None:
                try:
                    genai.configure(api_key=key)
                    # Try to create a model object (no network call until generate)
                    try:
                        genai.GenerativeModel(MODEL_CANDIDATES[0])
                    except Exception:
                        # Some SDKs instantiate lazily; still consider this a pass
                        pass
                    return True
                except Exception as e:  # SDK-level failure
                    log.debug(f"SDK key validation failed: {e}")
                    return False

        except Exception:
            # If something odd happens importing the SDK, fall through to REST
            pass

        # Fallback: do a minimal REST call to the Generative Language endpoint
        try:
            import json
            import urllib.request
            import urllib.error

            # Use current Gemini API format (not the old text-bison endpoint)
            TEST_MODEL = "models/gemini-2.0-flash"
            ENDPOINT_TEMPLATE = (
                "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={key}"
            )

            url = ENDPOINT_TEMPLATE.format(model=TEST_MODEL, key=key)
            body = json.dumps({
                "contents": [{"parts": [{"text": "Hi"}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 1
                }
            }).encode("utf-8")
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=8) as resp:
                    # Any 200/2xx response likely indicates a usable key
                    return True
            except urllib.error.HTTPError as e:
                try:
                    payload = e.read().decode("utf-8")
                except Exception:
                    payload = ""
                log.debug(f"REST key validation HTTP error: {e.code} - {payload[:200]}")
                return False
            except Exception as e:
                log.debug(f"REST key validation error: {e}")
                return False
        except Exception as e:
            log.debug(f"Falling back to REST validation failed: {e}")
            return False

    def rotate_key(self) -> Optional[str]:
        """Rotate to the next API key"""
        if len(self.api_keys) <= 1:
            return self.get_current_key()

        self.current_key_index = (self.current_key_index + 1) % len(
            self.api_keys
        )
        log.info(
            f"Rotated to API key #{self.current_key_index + 1}"
        )
        return self.get_current_key()

    def mark_key_failed(self, key: str, error_msg: str = ""):
        """Mark a key as failed temporarily"""
        if key in self.api_keys:
            self.failed_keys.add(key)
            key_index = self.api_keys.index(key) + 1
            log.warning(
                f"Marked API key #{key_index} as failed: {error_msg[:100]}"
            )

            # Rotate to next key
            self.rotate_key()

    def get_available_count(self) -> int:
        """Get count of available (non-failed) keys"""
        return len(self.api_keys) - len(self.failed_keys)

    def reset_failures(self):
        """Reset all failure tracking"""
        self.failed_keys.clear()
        log.info("Reset all API key failures")


# Global API key manager
api_key_manager = APIKeyManager()


class MultiKeyGeminiLM(dspy.LM):
    """Enhanced Gemini LM with multiple API key support and intelligent failover"""

    def __init__(self):
        super().__init__(model="gemini-multi-key")
        self.model = None
        self.current_model_name = None
        self.last_call_time = 0
        self.call_count = 0
        self.initialized = False

        log.info("MultiKeyGeminiLM created - will initialize on first call")

    def _configure_api_key(self) -> bool:
        """Configure the API with current key"""
        api_key = api_key_manager.get_current_key()
        if not api_key:
            log.error("No available API keys")
            return False

        genai = _import_genai()
        if genai is not None:
            try:
                genai.configure(api_key=api_key)
                log.debug(
                    f"Configured API key #{api_key_manager.current_key_index + 1}"
                )
                return True
            except Exception as e:
                log.error(f"Failed to configure API key via SDK: {e}")
                api_key_manager.mark_key_failed(api_key, str(e))
                return False

        # SDK not available; rely on REST fallback for now. We'll still treat
        # the key as usable here and let later REST calls fail/rotate if needed.
        log.debug("google.generativeai SDK not available; using REST fallback")
        return True

    def _try_initialize_model(self) -> bool:
        """Try to initialize a working model with current API key"""
        if not self._configure_api_key():
            return False

        # If SDK available, try to create a lightweight model instance and
        # perform a tiny generation to ensure the model and key work. If SDK
        # is not available, attempt a minimal REST generateContent call.
        last_error = None
        genai = _import_genai()

        for candidate in MODEL_CANDIDATES:
            try:
                log.debug(f"Trying model: {candidate}")

                if genai is not None:
                    test_model = genai.GenerativeModel(candidate)

                    # Simple test with minimal tokens
                    test_response = test_model.generate_content(
                        "OK",
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.0, max_output_tokens=3
                        ),
                    )

                    if test_response and getattr(test_response, "text", None):
                        self.model = test_model
                        self.current_model_name = candidate
                        self.initialized = True
                        log.info(
                            f"Initialized: {candidate} with key "
                            f"#{api_key_manager.current_key_index + 1}"
                        )
                        return True
                else:
                    # REST fallback: use the same endpoint as validation but
                    # with a tiny payload.
                    import json
                    import urllib.request

                    TEST_MODEL = candidate
                    ENDPOINT_TEMPLATE = (
                        "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={key}"
                    )
                    url = ENDPOINT_TEMPLATE.format(model=TEST_MODEL, key=api_key_manager.get_current_key())
                    body = json.dumps({
                        "contents": [{"parts": [{"text": "OK"}]}],
                        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1},
                    }).encode("utf-8")
                    req = urllib.request.Request(url, data=body, method="POST")
                    req.add_header("Content-Type", "application/json")
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        # any 2xx implies model works for this key
                        if resp.status >= 200 and resp.status < 300:
                            self.current_model_name = candidate
                            self.initialized = True
                            log.info(
                                f"Initialized (REST): {candidate} with key "
                                f"#{api_key_manager.current_key_index + 1}"
                            )
                            return True

            except Exception as e:
                error_msg = str(e)
                last_error = e

                if "429" in error_msg or "quota" in error_msg.lower():
                    log.warning(f"Quota exceeded for {candidate}")
                    # Mark this key as failed and try next one
                    current_key = api_key_manager.get_current_key()
                    api_key_manager.mark_key_failed(current_key, error_msg)

                    # Try with next key
                    if api_key_manager.get_available_count() > 0:
                        time.sleep(1)  # Brief pause
                        return self._try_initialize_model()  # Recursive retry
                    else:
                        log.error("All API keys exhausted")
                        return False

                elif "404" in error_msg:
                    log.debug(f"Model not found: {candidate}")
                else:
                    log.debug(f"Failed to initialize {candidate}: {error_msg[:100]}")

                time.sleep(0.5)
                continue

        log.error(f"Could not initialize any model. Last error: {last_error}")
        return False

    def _handle_rate_limit_error(
        self, error_msg: str, attempt: int
    ) -> Optional[float]:
        """Handle rate limiting with key rotation"""

        # Mark current key as failed
        current_key = api_key_manager.get_current_key()
        api_key_manager.mark_key_failed(current_key, error_msg)

        # Try to rotate to next available key
        if api_key_manager.get_available_count() > 0:
            log.info("Rate limited, rotating to next API key...")
            if self._try_initialize_model():
                return 1.0  # Short delay before retrying with new key

        # If no more keys available, calculate backoff delay
        delay = BASE_RETRY_DELAY**attempt + random.uniform(0, 1)

        # Try to extract retry delay from error message
        if "retry in" in error_msg:
            try:
                delay_match = re.search(r"retry in (\d+\.?\d*)", error_msg)
                if delay_match:
                    extracted_delay = float(delay_match.group(1))
                    delay = max(delay, extracted_delay)
            except Exception:
                pass

        return delay

    def _rate_limit_delay(self):
        """Add delay to respect rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time

        if time_since_last < MIN_DELAY_BETWEEN_CALLS:
            sleep_time = MIN_DELAY_BETWEEN_CALLS - time_since_last
            log.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_call_time = time.time()

    def _smart_key_rotation(self):
        """Intelligently rotate keys based on usage"""
        self.call_count += 1

        # Rotate every 10 calls to distribute load
        if self.call_count % 10 == 0 and api_key_manager.get_available_count() > 1:
            log.info(
                f"Proactive key rotation after {self.call_count} calls"
            )
            old_key = api_key_manager.get_current_key()
            api_key_manager.rotate_key()

            # Reinitialize with new key
            if api_key_manager.get_current_key() != old_key:
                self.initialized = False
                self._try_initialize_model()

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Initialize if needed
        if not self.initialized:
            if not self._try_initialize_model():
                log.error("No working Gemini model available")
                return "Error: No working model available"

        # Process input
        if messages and not prompt:
            if isinstance(messages, list):
                prompt = "\n".join(
                    [
                        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                        for msg in messages
                    ]
                )
            else:
                prompt = str(messages)

        if not prompt:
            return "Error: No prompt provided"

        # Rate limiting and smart rotation
        self._rate_limit_delay()
        self._smart_key_rotation()

        # Retry logic with key rotation
        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=(
                        _import_genai().types.GenerationConfig(
                            temperature=0.1, max_output_tokens=500
                        )
                        if _import_genai() is not None
                        else None
                    ),
                )

                if response and response.text:
                    result = response.text.strip()
                    log.debug(
                        f"Success with {self.current_model_name} (key "
                        f"#{api_key_manager.current_key_index + 1})"
                    )
                    return result
                else:
                    log.warning("Empty response from Gemini")
                    return "Error: Empty response"

            except Exception as e:
                error_msg = str(e)

                if "429" in error_msg or "quota" in error_msg.lower():
                    retry_delay = self._handle_rate_limit_error(
                        error_msg, attempt
                    )

                    if retry_delay and attempt < MAX_RETRIES - 1:
                        log.warning(
                            f"Retrying in {retry_delay:.1f}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Reset failures and try one more time
                        if attempt == MAX_RETRIES - 1:
                            api_key_manager.reset_failures()
                            if self._try_initialize_model():
                                continue

                        log.error("Rate limit exceeded on all keys")
                        return "Error: Rate limit exceeded"

                elif "404" in error_msg:
                    log.error(f"Model not found: {self.current_model_name}")
                    return "Error: Model not available"

                else:
                    log.error(f"API error: {error_msg}")
                    if attempt < MAX_RETRIES - 1:
                        delay = 1 + (attempt * 0.5)
                        log.info(f"Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    return f"Error: {error_msg[:100]}"

        return "Error: Max retries exceeded"


# Global LM instance functions
def get_configured_lm():
    """Get a configured LM instance"""
    try:
        return MultiKeyGeminiLM()
    except Exception as e:
        log.error(f"Failed to create LM: {e}")
        return None


def reset_api_keys():
    """Reset API key failure tracking"""
    api_key_manager.reset_failures()
    log.info("API key failures reset")


def rotate_api_key():
    """Manually rotate to next API key"""
    old_key = api_key_manager.current_key_index + 1
    api_key_manager.rotate_key()
    new_key = api_key_manager.current_key_index + 1
    log.info(f"Manually rotated from key #{old_key} to key #{new_key}")


def get_api_status():
    """Get API key status information"""
    return {
        "total_keys": len(api_key_manager.api_keys),
        "available_keys": api_key_manager.get_available_count(),
        "current_key": api_key_manager.current_key_index + 1,
        "failed_keys": len(api_key_manager.failed_keys),
    }
