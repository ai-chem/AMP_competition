"""Load YAML run configs, pin RNG seeds, and snapshot launch parameters."""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_SEED = 42


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; overlay lists and scalars replace."""
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_path(name: str | Path) -> Path:
    path = Path(name)
    if path.is_file():
        return path.resolve()
    candidate = CONFIGS_DIR / path
    if candidate.is_file():
        return candidate.resolve()
    if candidate.with_suffix(".yaml").is_file():
        return candidate.with_suffix(".yaml").resolve()
    raise FileNotFoundError(f"Config not found: {name}")


def load_config(name: str | Path = "default.yaml") -> dict[str, Any]:
    """Load a YAML mapping. ``extends: other.yaml`` merges the parent first."""

    def _load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        if path in stack:
            cycle = " -> ".join(str(item) for item in (*stack, path))
            raise ValueError(f"Config extends cycle: {cycle}")
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config {path} must be a mapping, got {type(data).__name__}")
        parent = data.pop("extends", None)
        if parent is None:
            return data
        parent_path = _resolve_config_path(parent)
        return deep_merge(_load(parent_path, (*stack, path)), data)

    return _load(_resolve_config_path(name), ())


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch. Call before model load / sampling."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except (TypeError, RuntimeError):
        pass


def _git_info() -> dict[str, Any]:
    def _git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()

    try:
        return {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {}


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import peft

        info["peft"] = peft.__version__
    except ImportError:
        pass
    return info


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def save_run(
    command: str,
    config: dict[str, Any],
    *,
    out_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``run.json`` + resolved YAML into ``out_dir`` and ``runs/<id>/``."""
    started = datetime.now(timezone.utc)
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{command}"
    payload = {
        "run_id": run_id,
        "command": command,
        "status": "running",
        "started_at": started.isoformat(),
        "seed": int(config.get("seed", DEFAULT_SEED)),
        "deterministic": bool(config.get("deterministic", True)),
        "config": _jsonable(config),
        "git": _git_info(),
        "env": _env_info(),
        "extra": _jsonable(extra or {}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = RUNS_DIR / run_id
    archive.mkdir(parents=True, exist_ok=True)
    for directory in (out_dir, archive):
        (directory / "run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_yaml(config, directory / "config.resolved.yaml")
    payload["_paths"] = {
        "out_dir": str(out_dir),
        "archive": str(archive),
        "run_json": str(out_dir / "run.json"),
    }
    return payload


def finish_run(payload: dict[str, Any], extra: dict[str, Any] | None = None, *, status: str = "completed") -> None:
    """Update saved run.json files with final status and extra fields."""
    payload["status"] = status
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    if extra:
        merged = dict(payload.get("extra") or {})
        merged.update(_jsonable(extra))
        payload["extra"] = merged
    paths = payload.get("_paths", {})
    dump = {key: value for key, value in payload.items() if not key.startswith("_")}
    text = json.dumps(dump, indent=2) + "\n"
    for key in ("out_dir", "archive"):
        directory = paths.get(key)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / "run.json").write_text(text, encoding="utf-8")
