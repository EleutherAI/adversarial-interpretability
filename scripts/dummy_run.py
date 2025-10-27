#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gamescope.libs.run_utils import run_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy run script to exercise run_context and indexing")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--env", type=str, default="dummy_env")
    parser.add_argument("--experiment", type=str, default="dummy_experiment")
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()

    base_dir = args.results_dir / args.env / args.experiment
    with run_context(base_dir=base_dir, run_prefix="dummy_run", config_args=vars(args)) as run_dir:
        raise RuntimeError("intentional failure for testing")
        if args.write_artifact:
            art = Path(run_dir) / "artifacts" / "dummy.txt"
            art.parent.mkdir(parents=True, exist_ok=True)
            art.write_text("dummy artifact\n", encoding="utf-8")
        if args.fail:
            raise RuntimeError("intentional failure for testing")


if __name__ == "__main__":
    main()


