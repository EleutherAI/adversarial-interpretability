from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


DEFAULT_SUBDIRS: List[str] = [
    "logs",
    "plots",
    "artifacts",
    "samples",
    "metrics",
]


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_for_path() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _run(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _git_metadata(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    md: Dict[str, Any] = {}
    md["commit_short"] = _run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
    md["commit_full"] = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    md["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    md["is_dirty"] = bool(status) if status is not None else None
    md["remote_origin_url"] = _run(["git", "config", "--get", "remote.origin.url"], cwd=repo_root)
    return md


def _safe_env() -> Dict[str, str]:
    # Capture full environment; redact a few common secrets by name pattern.
    redacted_keys = {"AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "HF_TOKEN", "WANDB_API_KEY"}
    env: Dict[str, str] = {}
    for k, v in os.environ.items():
        env[k] = "<REDACTED>" if k in redacted_keys else v
    return env


def _optional_cuda_metadata() -> Dict[str, Any]:
    try:
        import torch  # type: ignore

        return {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "num_devices": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception:
        return {
            "torch_cuda_available": None,
            "torch_cuda_version": None,
            "num_devices": None,
        }


def _coerce(obj: Any) -> Any:
    # Convert Paths and Namespaces and other simple containers to JSON/YAML-safe types
    if isinstance(obj, Path):
        return str(obj)
    # argparse.Namespace duck-typing: has __dict__
    if hasattr(obj, "__dict__") and not isinstance(obj, dict):
        return {k: _coerce(v) for k, v in vars(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [ _coerce(v) for v in obj ]
    return obj


def _pip_freeze() -> Optional[List[str]]:
    text = _run([sys.executable, "-m", "pip", "freeze"])
    if text is None:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines


def ensure_run_dir(run_dir: Path | str, create_subdirs: bool = True, subdirs: Optional[List[str]] = None) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    if create_subdirs:
        for name in (subdirs or DEFAULT_SUBDIRS):
            (path / name).mkdir(parents=True, exist_ok=True)
    return path


def start_run(base_dir: Path | str, run_prefix: Optional[str] = None, include_git_hash: bool = True,
              create_subdirs: bool = True, subdirs: Optional[List[str]] = None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if run_prefix is None:
        run_prefix = Path(sys.argv[0]).stem or "run"

    suffix = _timestamp_for_path()
    short = _git_metadata().get("commit_short") if include_git_hash else None
    name = f"{run_prefix}-{suffix}" + (f"-{short}" if short else "")
    run_path = base / name
    return ensure_run_dir(run_path, create_subdirs=create_subdirs, subdirs=subdirs)


def write_config_yaml(run_dir: Path | str, args: Any, filename: str = "config.yaml") -> Path:
    run_path = Path(run_dir)
    config_path = run_path / filename
    payload = _coerce(args)
    with open(config_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return config_path


def capture_metadata(run_dir: Path | str, extra: Optional[Dict[str, Any]] = None, filename: str = "metadata.json") -> Path:
    run_path = Path(run_dir)
    meta_path = run_path / filename

    repo_root = Path(__file__).resolve().parents[2]

    metadata: Dict[str, Any] = {
        "timestamp_utc": _now_utc_str(),
        "command": " ".join(sys.argv),
        "working_dir": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "env": _safe_env(),
        "git": _git_metadata(repo_root),
        "cuda": _optional_cuda_metadata(),
        "repo_root": str(repo_root),
        "pip_freeze": _pip_freeze(),
    }
    if extra:
        metadata.update(_coerce(extra))

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


