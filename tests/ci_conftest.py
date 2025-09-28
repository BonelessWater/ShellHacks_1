import os
import pytest

NO_NETWORK = os.environ.get("NO_NETWORK") == "1"

if NO_NETWORK:
    try:
        import requests

        class _BlockedRequests:
            def __getattr__(self, item):
                def _block(*args, **kwargs):
                    raise RuntimeError("Network calls are disabled in CI (NO_NETWORK=1)")

                return _block

        # Monkeypatch requests module globally
        requests.request = _BlockedRequests()
        requests.get = _BlockedRequests()
        requests.post = _BlockedRequests()
    except Exception:
        pass

    # Also block urllib
    try:
        import urllib.request as _urllib_request

        def _blocked_urlopen(*args, **kwargs):
            raise RuntimeError("Network calls are disabled in CI (NO_NETWORK=1)")

        _urllib_request.urlopen = _blocked_urlopen
    except Exception:
        pass


@pytest.fixture(autouse=True)
def ensure_no_network_during_tests():
    if NO_NETWORK:
        # If a test intentionally needs network, it must unset NO_NETWORK or mock fetchers
        yield
    else:
        yield
