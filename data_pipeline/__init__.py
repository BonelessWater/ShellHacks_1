# Shim package to maintain backwards compatibility with imports that expect a top-level `data_pipeline`
# It re-exports the backend.data_pipeline package.
from backend import data_pipeline as _dp

# Expose the subpackage modules
__all__ = [name for name in _dp.__all__] if hasattr(_dp, "__all__") else []

# Optionally attach attributes for tests that import specific modules
for attr in dir(_dp):
    if not attr.startswith("_"):
        globals()[attr] = getattr(_dp, attr)
