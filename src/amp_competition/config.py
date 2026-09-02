"""Load YAML run configs from the repository `configs/` directory."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(name: str = "default.yaml") -> dict[str, Any]:
    path = Path(name)
    if not path.is_file():
        path = CONFIGS_DIR / name
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping, got {type(data).__name__}")
    return data
