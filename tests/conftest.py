import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from flood.config import load_region  # noqa: E402


@pytest.fixture
def region(tmp_path):
    """The Lagos region with outputs redirected to a temporary folder and no raw data."""
    return load_region("lagos", data_dir=tmp_path / "data", output_dir=tmp_path / "out")
