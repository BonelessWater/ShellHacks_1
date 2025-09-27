#!/usr/bin/env python3
"""
Enhanced Gemini LLM module with multi-key support and improved error handling
"""

import time
import random
import re
import logging
from typing import Optional
import os

import google.generativeai as genai
import dspy

# Configuration constants
MODEL_CANDIDATES = [
    "models/gemini-2.5-flash",  # fast/cheaper
    "models/gemini-2.5-pro",    # stronger
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
        
        if not self.api_keys:
            raise RuntimeError("No valid GOOGLE_API_KEY found. Please set GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, or GOOGLE_API_KEY_3")
        
        log.info(f"Loaded {len(self.api_keys)} API keys")
    
    def _load_api_keys(self) -> list:
        """Load all available API keys from environment"""
        keys = []
        
        # Check for numbered keys first
        for i in range(0, 1):
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if key and key.strip():
                keys.append(key.strip())
                log.info(f"Found GOOGLE_API_KEY_{i}")
        
        # Fallback to original key name
        fallback_key = os.getenv("GOOGLE_API_KEY")
        if fallback_key and fallback_key.strip() and fallback_key not in keys:
            keys.append(fallback_key.strip())
            log.info("Found GOOGLE_API_KEY (fallback)")
        
        return keys
    
    def get_current_key(self) -> Optional[str]:
        """Get the current API key"""
        if not self.api_keys:
            return None
        
        # Find next working key
        attempts = 0
        while attempts < len(self.api_keys):
            key = self.api_keys[self.current_key_index]
            
            if key not in self.failed_keys:
                log.debug(f"Using API key #{self.current_key_index + 1}")
                return key
            
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            attempts += 1
        
        # All keys failed, reset and try again
        if self.failed_keys:
            log.warning("All keys failed, resetting failure tracking")
            self.failed_keys.clear()
            return self.api_keys[self.current_key_index]
        
        return None
    
    def rotate_key(self) -> Optional[str]:
        """Rotate to the next API key"""
        if len(self.api_keys) <= 1:
            return self.get_current_key()
        
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        log.info(f"Rotated to API key #{self.current_key_index + 1}")
        return self.get_current_key()
    
    def mark_key_failed(self, key: str, error_msg: str = ""):
        """Mark a key as failed temporarily"""
        if key in self.api_keys:
            self.failed_keys.add(key)
            key_index = self.api_keys.index(key) + 1
            log.warning(f"Marked API key #{key_index} as failed: {error_msg[:100]}")
            
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
        
        try:
            genai.configure(api_key=api_key)
            log.debug(f"Configured API key #{api_key_manager.current_key_index + 1}")
            return True
        except Exception as e:
            log.error(f"Failed to configure API key: {e}")
            api_key_manager.mark_key_failed(api_key, str(e))
            return False
    
    def _try_initialize_model(self) -> bool:
        """Try to initialize a working model with current API key"""
        if not self._configure_api_key():
            return False
        
        last_error = None
        for candidate in MODEL_CANDIDATES:
            try:
                log.debug(f"Trying model: {candidate}")
                test_model = genai.GenerativeModel(candidate)
                
                # Simple test with minimal tokens
                test_response = test_model.generate_content(
                    "OK",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=3
                    )
                )
                
                if test_response and test_response.text:
                    self.model = test_model
                    self.current_model_name = candidate
                    self.initialized = True
                    log.info(f"Initialized: {candidate} with key #{api_key_manager.current_key_index + 1}")
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
                        time.sleep(1)  # Brief pause before trying next key
                        return self._try_initialize_model()  # Recursive retry with new key
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
    
    def _handle_rate_limit_error(self, error_msg: str, attempt: int) -> Optional[float]:
        """Handle rate limiting with key rotation"""
        
        # Mark current key as failed
        current_key = api_key_manager.get_current_key()
        api_key_manager.mark_key_failed(current_key, error_msg)
        
        # Try to rotate to next available key
        if api_key_manager.get_available_count() > 0:
            log.info(f"Rate limited, rotating to next API key...")
            if self._try_initialize_model():
                return 1.0  # Short delay before retrying with new key
        
        # If no more keys available, calculate backoff delay
        delay = BASE_RETRY_DELAY ** attempt + random.uniform(0, 1)
        
        # Try to extract retry delay from error message
        if "retry in" in error_msg:
            try:
                delay_match = re.search(r'retry in (\d+\.?\d*)', error_msg)
                if delay_match:
                    extracted_delay = float(delay_match.group(1))
                    delay = max(delay, extracted_delay)
            except:
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
            log.info(f"Proactive key rotation after {self.call_count} calls")
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
                prompt = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages])
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
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=500,
                    )
                )
                
                if response and response.text:
                    result = response.text.strip()
                    log.debug(f"Success with {self.current_model_name} (key #{api_key_manager.current_key_index + 1})")
                    return result
                else:
                    log.warning("Empty response from Gemini")
                    return "Error: Empty response"
                    
            except Exception as e:
                error_msg = str(e)
                
                if "429" in error_msg or "quota" in error_msg.lower():
                    retry_delay = self._handle_rate_limit_error(error_msg, attempt)
                    
                    if retry_delay and attempt < MAX_RETRIES - 1:
                        log.warning(f"Retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
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
        "failed_keys": len(api_key_manager.failed_keys)
    }