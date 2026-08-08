import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from utils.config import load_config  # noqa: E402


@pytest.fixture
def tiny_cfg():
    return load_config(ROOT / "configs" / "vqvae_tiny_cpu.yaml")
