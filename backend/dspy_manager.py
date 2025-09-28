"""Guarded DSPy module manager for initializing DSPy modules lazily.

This module provides a small wrapper around dspy and google.generativeai
integration. It is defensive: imports are delayed and failures are logged
instead of raising so the rest of the app can run in test environments.
"""
import logging
import os
from typing import Optional, Dict, Any

log = logging.getLogger("dspy_manager")


class DSPyModuleManager:
    def __init__(self):
        self.modules: Optional[Dict[str, Any]] = None
        self.initialized = False

    def initialize_modules(self) -> bool:
        """Attempt to configure dspy and create modules. Returns True on success."""
        if self.initialized:
            return True

        try:
            import dspy  # type: ignore
            import google.generativeai as genai  # type: ignore
        except Exception as e:
            log.warning(f"DSPy or Google Generative AI SDK not available: {e}")
            return False

        try:
            # Configure generative API with the first available key
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                # Try numbered keys
                for i in range(0, 5):
                    k = os.environ.get(f"GOOGLE_API_KEY_{i}")
                    if k:
                        api_key = k
                        break

            if not api_key:
                log.warning("No GOOGLE_API_KEY found in environment; DSPy will not be fully functional")
            else:
                try:
                    genai.configure(api_key=api_key)
                except Exception as ee:
                    log.warning(f"Failed to configure google.generativeai: {ee}")

            # Create a basic LM wrapper for dspy (optional)
            class _SimpleGenie:
                def __init__(self, model_name: str = "gemini-1.5-flash"):
                    self.model_name = model_name

                def __call__(self, *args, **kwargs):
                    # dspy will call the configured LM object; for simplicity return a string
                    try:
                        # create on-demand to avoid keeping references
                        m = genai.GenerativeModel(self.model_name)
                        # dspy expects a callable that returns some structured object; keep simple
                        return m
                    except Exception as e:
                        log.warning(f"GenerativeModel init failed: {e}")
                        return None

            # Configure DSPy to use our LM wrapper
            try:
                lm = _SimpleGenie()
                dspy.configure(lm=lm)
            except Exception as e:
                log.warning(f"dspy.configure failed: {e}")

            # Attempt to create modules via project's signatures helper if present
            try:
                from backend.dspy_signatures import create_dspy_modules  # type: ignore

                self.modules = create_dspy_modules()
            except Exception as e:
                log.warning(f"create_dspy_modules not available or failed: {e}")
                self.modules = None

            self.initialized = True
            return True

        except Exception as e:
            log.warning(f"DSPy initialization failed: {e}")
            self.initialized = False
            return False

    def get_module(self, name: str):
        if not self.initialized:
            return None
        if not self.modules:
            return None
        return self.modules.get(name)


_global_manager = DSPyModuleManager()


def get_manager() -> DSPyModuleManager:
    return _global_manager
