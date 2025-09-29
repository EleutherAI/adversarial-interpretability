from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from libs.run_utils import capture_metadata, start_run, write_config_yaml


def wait_for_vllm_ready(base_url: str, timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    health_url = base_url.rstrip("/") + "/v1/models"
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: PERF203
            last_err = str(e)
        time.sleep(1)
    raise RuntimeError(f"vLLM did not become ready: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch vLLM and run Qwen BC eval with shared args")
    # Shared
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--device", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--results_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "results" / "chess_probe"))

    # vLLM args (common subset)
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=2048)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_extra_args", type=str, nargs=argparse.REMAINDER, default=[])

    # Eval args
    parser.add_argument("--num_eval_data", type=int, default=200)
    parser.add_argument("--save_jsonl", action="store_true")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)

    args = parser.parse_args()

    # Create the run directory up front and record combined config
    run_dir = start_run(base_dir=Path(args.results_dir), run_prefix="qwen_bc_eval")

    combined_config: Dict[str, Any] = {
        "orchestrator": {
            "model_name_or_path": args.model_name_or_path,
            "device": args.device,
        },
        "vllm": {
            "port": int(args.vllm_port),
            "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
            "max_model_len": int(args.vllm_max_model_len),
            "dtype": args.vllm_dtype,
            "extra_args": list(args.vllm_extra_args) if args.vllm_extra_args else [],
        },
        "eval": {
            "num_eval_data": int(args.num_eval_data),
            "save_jsonl": bool(args.save_jsonl),
            "dataset_path": args.dataset_path,
            "lora_adapter_path": args.lora_adapter_path,
            "system_prompt": args.system_prompt,
        },
    }
    write_config_yaml(run_dir, combined_config)
    capture_metadata(run_dir, extra={"orchestration": True})

    # Launch vLLM serve
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    port = int(args.vllm_port)
    base_url = f"http://127.0.0.1:{port}"

    vllm_cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model_name_or_path,
        "--port",
        str(port),
        "--dtype",
        args.vllm_dtype,
        "--gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--max-model-len",
        str(args.vllm_max_model_len),
    ]
    if args.vllm_extra_args:
        vllm_cmd.extend(args.vllm_extra_args)

    logs_dir = Path(run_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    vllm_log_path = logs_dir / "vllm_serve.log"
    with open(vllm_log_path, "w") as vllm_log:
        vllm_proc = subprocess.Popen(vllm_cmd, env=env, stdout=vllm_log, stderr=subprocess.STDOUT)

    try:
        wait_for_vllm_ready(base_url, timeout_s=180)

        # Run eval, pointing to the same run_dir, so it writes metrics inside
        eval_cmd: List[str] = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "environments" / "chess_probe" / "scripts" / "eval_qwen_bc.py"),
            "--model_name_or_path",
            args.model_name_or_path,
            "--num_eval_data",
            str(args.num_eval_data),
            "--device",
            "cuda",
            "--run_dir",
            str(run_dir),
        ]
        if args.save_jsonl:
            eval_cmd.append("--save_jsonl")
        if args.dataset_path:
            eval_cmd.extend(["--dataset_path", args.dataset_path])
        if args.lora_adapter_path:
            eval_cmd.extend(["--lora_adapter_path", args.lora_adapter_path])
        if args.system_prompt:
            eval_cmd.extend(["--system_prompt", args.system_prompt])

        eval_log_path = logs_dir / "eval.log"
        with open(eval_log_path, "w") as eval_log:
            ret = subprocess.call(eval_cmd, stdout=eval_log, stderr=subprocess.STDOUT)
        if ret != 0:
            raise RuntimeError(f"eval script exited with code {ret}")
    finally:
        # Graceful shutdown of vLLM
        try:
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=10)
            except Exception:
                vllm_proc.kill()
        except Exception:
            pass


if __name__ == "__main__":
    main()


