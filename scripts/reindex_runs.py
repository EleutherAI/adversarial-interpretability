#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY)
    try:
        os.write(fd, (line + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def _iter_metadata_files(results_root: Path) -> Iterable[Path]:
    # Scan depth up to results/<env>/<experiment>/<run>/metadata.json
    for env_dir in results_root.iterdir():
        if not env_dir.is_dir() or env_dir.name == "index":
            continue
        for experiment_dir in env_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
            for run_dir in experiment_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                meta = run_dir / "metadata.json"
                if meta.exists():
                    yield meta


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def reindex(results_root: Path, dry_run: bool = False) -> int:
    index_dir = results_root / "index"
    index_path = index_dir / "runs_index.jsonl"
    count = 0
    for meta_path in _iter_metadata_files(results_root):
        run_dir = meta_path.parent
        env_name = run_dir.parent.parent.name if len(run_dir.parents) >= 2 else None
        experiment_name = run_dir.parent.name
        data = _load_json(meta_path) or {}
        timestamp_utc = data.get("timestamp_utc")
        hostname = data.get("hostname")
        command = data.get("command")
        script_name = None
        if isinstance(command, str) and command:
            first = command.split()[0]
            script_name = Path(first).stem
        git = data.get("git") or {}
        commit_short = git.get("commit_short")
        record = {
            "event": "backfill",
            "timestamp_utc": timestamp_utc,
            "run_dir": str(run_dir),
            "run_basename": run_dir.name,
            "experiment_name": experiment_name,
            "env_name": env_name,
            "results_root": str(results_root),
            "hostname": hostname,
            "user": None,
            "pid": None,
            "git_commit_short": commit_short,
            "script_name": script_name,
            "status": "unknown",
        }
        if dry_run:
            print(json.dumps(record))
        else:
            _append_jsonl(index_path, record)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill runs index from existing runs")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    count = reindex(args.results_root, dry_run=args.dry_run)
    print(f"Indexed {count} runs from {args.results_root}")


if __name__ == "__main__":
    main()


