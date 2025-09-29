from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from libs.run_utils import capture_metadata, ensure_run_dir, start_run, write_config_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic config runner for experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config containing 'command' and 'args'")
    parser.add_argument("--results_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--run_prefix", type=str, default="experiment")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # Required fields: command (str) and args (list[str] or dict)
    command: str = str(cfg.get("command"))
    if not command:
        raise ValueError("config must include 'command'")

    run_dir = start_run(base_dir=Path(args.results_dir), run_prefix=str(args.run_prefix))

    # Write the full config (including resolved run_dir) into the run folder
    enriched = dict(cfg)
    enriched["run_dir"] = str(run_dir)
    write_config_yaml(run_dir, enriched)
    capture_metadata(run_dir, extra={"config_runner": True})

    # Build argument list
    cmdline = [sys.executable] if command.endswith(".py") and " " not in command else []
    if cmdline:
        cmdline.append(command)
    else:
        # raw shell-like command; split minimal by space
        cmdline = command.split()

    # Expand args
    args_section = cfg.get("args", {})
    if isinstance(args_section, dict):
        for k, v in args_section.items():
            flag = f"--{k.replace('_','-')}"
            if isinstance(v, bool):
                if v:
                    cmdline.append(flag)
            else:
                cmdline.extend([flag, str(v)])
    elif isinstance(args_section, list):
        cmdline.extend([str(x) for x in args_section])
    else:
        raise ValueError("'args' must be a list or mapping")

    # Ensure subprocess knows the run_dir
    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir)

    logs_dir = Path(run_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "config_runner.log", "w") as log:
        ret = subprocess.call(cmdline, env=env, stdout=log, stderr=subprocess.STDOUT)
    if ret != 0:
        raise SystemExit(ret)


if __name__ == "__main__":
    main()


