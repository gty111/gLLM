import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frontend_reports_the_package_version():
    setup = re.search(r'version="([^"]+)"', (ROOT / "setup.py").read_text())[1]
    cargo = re.search(r'^version = "([^"]+)"', (ROOT / "rust/frontend/Cargo.toml").read_text(), re.M)[1]
    assert cargo == setup
