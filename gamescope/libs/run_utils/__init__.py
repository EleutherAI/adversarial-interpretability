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
import getpass

import yaml
from contextlib import contextmanager


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
    # Try pip freeze first
    text = _run([sys.executable, "-m", "pip", "freeze"])
    if text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines
    # Fallback to importlib.metadata if pip is not available in the runtime
    try:
        try:
            import importlib.metadata as im  # type: ignore
        except Exception:  # pragma: no cover
            import importlib_metadata as im  # type: ignore
        pkgs = [f"{d.metadata['Name']}=={d.version}" for d in im.distributions() if d.metadata.get('Name')]
        pkgs = sorted({p for p in pkgs})
        return pkgs
    except Exception:
        return None


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


def write_config_yaml(
    run_dir: Path | str,
    command: str,
    args: Any,
    filename: str = "config.yaml",
) -> Path:
    """Write a configuration YAML capturing the exact command and arguments.

    Behavior:
      - If args is a simple mapping/namespace, writes {command: ..., args: {...}}
      - If args is a mapping that already includes structured sections (e.g., from an orchestrator),
        it preserves the structure and just ensures 'command' is present/updated.
    """
    run_path = Path(run_dir)
    config_path = run_path / filename
    coerced = _coerce(args)

    if isinstance(coerced, Mapping):
        payload: Dict[str, Any] = dict(coerced)
        payload["command"] = command
    else:
        payload = {"command": command, "args": coerced}

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
    # Append a lightweight record to the shared runs index for discoverability
    try:
        _append_runs_index_event(run_path, event="start", timestamp=metadata.get("timestamp_utc"), status="running")
    except Exception:
        pass
    return meta_path


def _derive_results_root(run_path: Path) -> Path:
    # Expecting results/<env>/<experiment>/<run>
    return run_path.parents[3] if len(run_path.parents) >= 4 else run_path.parents[-1]


def _append_runs_index_event(run_path: Path, event: str, timestamp: Optional[str], status: Optional[str], exit_reason: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    results_root = _derive_results_root(run_path)
    index_dir = Path(results_root) / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    argv0 = (sys.argv[1] if (len(sys.argv) > 1 and sys.argv[0].endswith("python")) else sys.argv[0])
    script_name = Path(argv0).stem
    record: Dict[str, Any] = {
        "event": event,
        "timestamp_utc": timestamp or _now_utc_str(),
        "run_dir": str(run_path),
        "run_basename": run_path.name,
        "experiment_name": run_path.parent.name,
        "env_name": run_path.parent.parent.name if len(run_path.parents) >= 2 else None,
        "results_root": str(results_root),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "pid": os.getpid(),
        "git_commit_short": _git_metadata().get("commit_short"),
        "script_name": script_name,
        "status": status,
        "exit_reason": exit_reason,
    }
    if extra:
        record.update(_coerce(extra))

    index_path = index_dir / "runs_index.jsonl"
    line = json.dumps(record, ensure_ascii=False)
    fd = os.open(str(index_path), os.O_CREAT | os.O_APPEND | os.O_WRONLY)
    try:
        os.write(fd, (line + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def mark_status(run_dir: Path | str, status: str, exit_reason: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write a status.json in the run dir and append a terminal event to the runs index.

    status: "success" | "failed" | custom
    exit_reason: e.g., "exception", "system_exit", "normal"
    """
    run_path = Path(run_dir)
    status_path = run_path / "status.json"
    payload: Dict[str, Any] = {
        "timestamp_utc": _now_utc_str(),
        "status": status,
    }
    if exit_reason:
        payload["exit_reason"] = exit_reason
    if extra:
        payload.update(_coerce(extra))
    with open(status_path, "w") as f:
        json.dump(payload, f, indent=2)

    try:
        _append_runs_index_event(run_path, event="end", timestamp=payload["timestamp_utc"], status=status, exit_reason=exit_reason, extra=extra)
    except Exception:
        pass
    return status_path


@contextmanager
def run_context(base_dir: Path | str, run_prefix: Optional[str] = None, config_args: Any = None, config_filename: str = "config.yaml", create_subdirs: bool = True, subdirs: Optional[List[str]] = None):
    """Context manager that sets up a run directory and marks status on exit.

    Usage:
        with run_context(base_dir=results_dir, run_prefix="qwen_bc_eval", config_args=vars(args)) as run_dir:
            ...
    """
    run_dir = start_run(base_dir=base_dir, run_prefix=run_prefix, create_subdirs=create_subdirs, subdirs=subdirs)
    write_config_yaml(run_dir, f"{sys.executable} " + " ".join(sys.argv), config_args or {}, filename=config_filename)
    capture_metadata(run_dir)
    try:
        yield run_dir
    except SystemExit as e:
        mark_status(run_dir, status="failed" if int(getattr(e, "code", 1) or 1) != 0 else "success", exit_reason="system_exit")
        raise
    except Exception:
        mark_status(run_dir, status="failed", exit_reason="exception")
        raise
    else:
        mark_status(run_dir, status="success", exit_reason="normal")


def _resolve_run_dir_from_path(path: Path) -> Optional[Path]:
    cur = path.resolve()
    if cur.is_file():
        cur = cur.parent
    for ancestor in [cur] + list(cur.parents):
        if (ancestor / "metadata.json").exists():
            return ancestor
    return None


def mark_artifact_used(artifact_path: Path | str, consumer_run_dir: Optional[Path | str] = None, reason: Optional[str] = None) -> None:
    """Record that an artifact produced by a run was used elsewhere.

    Writes a line to <producer_run_dir>/artifacts/USED_BY.jsonl and appends an
    'artifact_used' event to the runs index for the producer run.
    """
    art_path = Path(artifact_path)
    producer_run = _resolve_run_dir_from_path(art_path)
    if producer_run is None:
        return
    artifacts_dir = producer_run / "artifacts"
    try:
        rel = str(art_path.resolve().relative_to(artifacts_dir)) if art_path.resolve().is_relative_to(artifacts_dir) else art_path.name
    except Exception:
        rel = art_path.name

    used_by_path = artifacts_dir / "USED_BY.jsonl"
    record: Dict[str, Any] = {
        "when": _now_utc_str(),
        "artifact_relpath": rel,
        "consumer_run_dir": str(consumer_run_dir) if consumer_run_dir else os.environ.get("RUN_DIR"),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
    }
    if reason:
        record["reason"] = reason

    used_by_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    fd = os.open(str(used_by_path), os.O_CREAT | os.O_APPEND | os.O_WRONLY)
    try:
        os.write(fd, (line + "\n").encode("utf-8"))
    finally:
        os.close(fd)

    try:
        _append_runs_index_event(producer_run, event="artifact_used", timestamp=None, status=None, exit_reason=None, extra={
            "artifact_relpath": rel,
            "consumer_run_dir": record.get("consumer_run_dir"),
            "reason": reason,
        })
    except Exception:
        pass


