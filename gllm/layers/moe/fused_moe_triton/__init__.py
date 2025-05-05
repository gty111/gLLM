from typing import Optional, Dict, Any
from contextlib import contextmanager

_config: Optional[Dict[str, Any]] = None

@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> Optional[Dict[str, Any]]:
    return _config