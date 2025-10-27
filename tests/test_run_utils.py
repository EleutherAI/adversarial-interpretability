from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_dummy_run_success(tmp_path: Path) -> None:
    results = tmp_path / "results"
    cmd = [sys.executable, str(SCRIPTS / "dummy_run.py"), "--results-dir", str(results), "--write-artifact"]
    proc = _run(cmd, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr

    # Index should exist
    index = results / "index" / "runs_index.jsonl"
    assert index.exists()
    lines = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    assert any(r.get("event") == "start" for r in lines)
    assert any(r.get("event") == "end" and r.get("status") == "success" for r in lines)

    # Find non-junk listing should show at least one line under dummy_run
    proc2 = _run([sys.executable, str(SCRIPTS / "find_run.py"), "--results-root", str(tmp_path)], cwd=REPO_ROOT)
    assert proc2.returncode == 0
    assert "# dummy_run" in proc2.stdout


def test_dummy_run_failure(tmp_path: Path) -> None:
    results = tmp_path / "results"
    cmd = [sys.executable, str(SCRIPTS / "dummy_run.py"), "--results-dir", str(results), "--fail"]
    proc = _run(cmd, cwd=REPO_ROOT)
    assert proc.returncode != 0

    index = results / "index" / "runs_index.jsonl"
    assert index.exists()
    lines = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    assert any(r.get("event") == "end" and r.get("status") == "failed" and r.get("exit_reason") == "exception" for r in lines)


