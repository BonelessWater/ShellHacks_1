#!/usr/bin/env python3
"""
Configuration module for Invoice Fraud Detection System
Manages multiple Google API keys for load balancing
"""

import os
import logging
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("fraud_detection_config")

class APIKeyManager:
    """Manages multiple Google API keys for load balancing"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.failed_keys = set()
        
        if not self.api_keys:
            raise RuntimeError("No valid GOOGLE_API_KEY_0 found. Please set GOOGLE_API_KEY_0")
        
        log.info(f"Loaded {len(self.api_keys)} API keys")
    
    def _load_api_keys(self) -> List[str]:
        """Load only GOOGLE_API_KEY_0 from environment"""
        keys = []

        # Only check for GOOGLE_API_KEY_0
        key = os.getenv("GOOGLE_API_KEY_0")
        if key and key.strip():
            keys.append(key.strip())
            log.info("Found GOOGLE_API_KEY_0")

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

# Global instance
api_key_manager = APIKeyManager()

# Constants
MODEL_CANDIDATES = [
    "models/gemini-2.5-flash",
]

APPROVED_VENDORS = {
    "ACME Corp", 
    "Beta Industries", 
    "Delta LLC", 
    "Gamma Tech"
}

SUSPICIOUS_KEYWORDS = [
    "gift", "cash", "tip", "bonus", "personal", "entertainment"
]

# Thresholds
HIGH_VALUE_THRESHOLD = 1000.0
HIGH_QUANTITY_THRESHOLD = 100
TOTAL_MISMATCH_TOLERANCE = 0.01

# Rate limiting
MIN_DELAY_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0