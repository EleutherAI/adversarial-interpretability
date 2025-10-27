#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


INDEX_RELATIVE_PATH = Path("index/runs_index.jsonl")


@dataclass
class RunRecord:
    run_dir: Path
    run_basename: str
    env_name: Optional[str]
    experiment_name: Optional[str]
    timestamp: Optional[datetime]
    script_name: Optional[str]
    hostname: Optional[str]
    user: Optional[str]
    status: Optional[str]
    duration_seconds: Optional[float] = None
    usage_count: int = 0


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Expecting e.g. 2025-10-23T05:09:14Z
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _read_index(index_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not index_path.exists():
        return records
    with open(index_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def _to_run_records(raw: Iterable[Dict[str, Any]]) -> List[RunRecord]:
    by_run_dir: Dict[str, Dict[str, Any]] = {}
    start_times: Dict[str, datetime] = {}
    end_times: Dict[str, datetime] = {}
    usage_counts: Dict[str, int] = {}
    for r in raw:
        rd = r.get("run_dir")
        if not rd:
            continue
        event = r.get("event")
        ts = _parse_timestamp(r.get("timestamp_utc"))
        if event == "start" and ts is not None:
            start_times[rd] = ts
        elif event == "end" and ts is not None:
            end_times[rd] = ts
        elif event == "artifact_used":
            usage_counts[rd] = usage_counts.get(rd, 0) + 1
        # Keep the last record as the summary for fields like status/script
        by_run_dir[rd] = r
    out: List[RunRecord] = []
    for r in by_run_dir.values():
        rd = r.get("run_dir")
        start_ts = start_times.get(rd)
        end_ts = end_times.get(rd)
        duration = (end_ts - start_ts).total_seconds() if (start_ts and end_ts) else None
        out.append(
            RunRecord(
                run_dir=Path(r.get("run_dir")),
                run_basename=str(r.get("run_basename") or Path(r.get("run_dir")).name),
                env_name=r.get("env_name"),
                experiment_name=r.get("experiment_name"),
                timestamp=_parse_timestamp(r.get("timestamp_utc")),
                script_name=r.get("script_name"),
                hostname=r.get("hostname"),
                user=r.get("user"),
                status=r.get("status"),
                duration_seconds=duration,
                usage_count=usage_counts.get(rd, 0),
            )
        )
    return out


def _has_any_files(directory: Path) -> bool:
    if not directory.exists():
        return False
    for _root, _dirs, files in os.walk(directory):
        if files:
            return True
    return False


def _is_non_junk(run: RunRecord) -> bool:
    # Heuristic: non-junk if there are artifacts OR metrics present.
    artifacts = run.run_dir / "artifacts"
    metrics = run.run_dir / "metrics"
    if _has_any_files(artifacts):
        return True
    if _has_any_files(metrics):
        return True
    return False


def _derive_script_name_fallback(run: RunRecord) -> Optional[str]:
    if run.script_name:
        return run.script_name
    # Fallback: look for config.yaml or metadata.json and infer script
    meta_path = run.run_dir / "metadata.json"
    try:
        if meta_path.exists():
            data = json.loads(meta_path.read_text())
            cmd = data.get("command")
            if isinstance(cmd, str) and cmd:
                first = cmd.split()[0]
                return Path(first).stem
    except Exception:
        pass
    # As a last resort, use experiment name
    return run.experiment_name or "unknown"


def _format_run_line(run: RunRecord) -> str:
    ts = run.timestamp.isoformat() if run.timestamp else ""
    path_str = str(run.run_dir)
    env_ex = "/".join([x for x in [run.env_name, run.experiment_name, run.run_basename] if x])
    dur = f"  dur={int(run.duration_seconds)}s" if run.duration_seconds is not None else ""
    uses = f"  used={run.usage_count}" if run.usage_count else ""
    return f"{ts}  {env_ex}{dur}{uses}  {path_str}"


def list_default(results_root: Path, name_contains: Optional[str] = None, env_contains: Optional[str] = None) -> int:
    index_path = results_root / INDEX_RELATIVE_PATH
    raw = _read_index(index_path)
    runs = _to_run_records(raw)
    # Filter non-junk
    runs = [r for r in runs if _is_non_junk(r)]
    # Optional filters
    if name_contains:
        sub = name_contains.lower()
        runs = [r for r in runs if sub in r.run_basename.lower()]
    if env_contains:
        sube = env_contains.lower()
        runs = [r for r in runs if (r.env_name or "").lower().find(sube) != -1]
    # Group by script
    grouped: Dict[str, List[RunRecord]] = defaultdict(list)
    for r in runs:
        key = _derive_script_name_fallback(r) or "unknown"
        grouped[key].append(r)
    # Sort each group by timestamp desc (fallback to mtime)
    def sort_key(r: RunRecord) -> Tuple[float, str]:
        if r.timestamp is not None:
            return (r.timestamp.timestamp(), r.run_basename)
        try:
            return (r.run_dir.stat().st_mtime, r.run_basename)
        except Exception:
            return (0.0, r.run_basename)

    for key in sorted(grouped.keys()):
        group = sorted(grouped[key], key=sort_key, reverse=True)
        print(f"\n# {key} ({len(group)})")
        for r in group:
            print(_format_run_line(r))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and list experiment runs")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Path to the results directory (index expected at <results-root>/index)",
    )
    parser.add_argument("--contains", type=str, default=None, help="Substring to match in run name")
    parser.add_argument("--env", type=str, default=None, help="Substring to match in environment name")
    args = parser.parse_args()

    # Default behavior: non-junk grouped by script, newest first, with optional filters
    code = list_default(args.results_root, name_contains=args.contains, env_contains=args.env)
    raise SystemExit(code)


if __name__ == "__main__":
    main()


