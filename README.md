# Monorepo Guide

Scope
- Shared infrastructure for adversarial interpretability across competitive games (chess, diplomacy, etc.).
- Common evals, visualization, data reports, model wrappers, and commands to run training jobs

Non-goals
- Forcing similar model architectures or training code across all experiments.
- Over-abstracting before concrete use cases exist.
- Standardised analysis for all experiements - just want some consistency in final presentation (plots/tables)

Directory layout
- docs/
- environments/
  - chess_probe/
- libs/
  - evals/
  - visualization/
- configs/
  - examples/
- scripts/

Shared libraries
- evals/: engine-eval delta, Elo, deception metrics (PR/recall), cost tracking.
- visualization/: plotting helpers and experiment dashboards.
- probes/: soft-token and residual injection modules with small, clear APIs.
- engines/: thin wrappers for Stockfish/Lc0 or other evaluators.

Runners
- TRL PPO (single-turn for chess-probe) with probe-only optimization.
- Verifier- or agent-tooling adapters for multi-step environments.

Results and experiment tracking
- Location: write all outputs under `results/<env>/<experiment_name>/<YYYYMMDD_HHMMSS-<run_id>>/`.
  - Example: `results/chess_probe/probe_ablation/20250115_142530-a1b2c3/`
- Contents inside a run directory:
  - `config.yaml` (or `.toml`): exact configuration used for the run (copied from `--config` or auto-dumped resolved config)
  - `metadata.json`: immutable run metadata
    - git commit, branch, dirty flag; user, host; Python/CUDA versions; random seeds
    - full invocation (command, args, `PYTHONPATH`), environment name, library versions (optionally `pip freeze`)
  - `logs/`: captured stdout/stderr/wandb
  - `plots/`: generated figures for quick inspection
  - `artifacts/`: model/probe checkpoints and large outputs (consider symlinks or pointer files if we need to store stuff elsewhere)
  - `samples/`: qualitative samples (games, traces, prompts/responses)
  - `metrics/`: summary metrics from experiment
- Script conventions (strongly recommended):
  - `--config path/to/config.yaml` and `--experiment-name <slug>`
  - `--output-dir results/` (default) so scripts create the full run path automatically
  - `--notes "short freeform note"` saved in `metadata.json`
  - On startup: create the run directory, copy the config, write `metadata.json`
  - During training/eval: append metrics to `metrics.jsonl`, write plots and artifacts under the run directory
- Remote trackers: optionally mirror metrics to W&B or MLflow, but the filesystem record above is the source of truth for reproducibility.

### Config runner

Run any experiment from a YAML file; a fresh run directory is created and the full config is recorded.

```bash
uv run python scripts/config_runner.py --config configs/examples/my_eval.yaml
```

YAML shape:

```yaml
command: environments/chess_probe/scripts/eval_qwen_bc.py
args:
  model_name_or_path: Qwen/Qwen3-8B-Base
  num_eval_data: 200
  results_dir: results/chess_probe
  save_jsonl: true
```

The runner injects `run_dir` for downstream scripts (available as `--run_dir` if supported, otherwise in env as RUN_DIR).



Add a new environment
1) Create environments/<env_name>/ with a README.md describing assumptions and dependencies.
2) Reuse libs/ components where possible; avoid environment-specific logic in libs/ (or, if you need evals and they'd be relevant to multiple experiments, create them in libs/).
3) Provide example configs under configs/examples/ to run your experiments.
4) Add/modify scripts under scripts/ to run your experiment and collect results.

Licensing
- Preserve third-party licenses and headers. See THIRD_PARTY_NOTICES.md.

## Setup with uv

- Install uv (Linux/macOS):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Verify installation:
```bash
uv --version
```

- Initial environment setup (once `pyproject.toml` exists at the repo root):
```bash
uv sync
```
This creates a local virtual environment (e.g., `.venv/`) and installs the base project dependencies.

- Run scripts using the synced environment:
```bash
uv run python scripts/your_script.py --help
```

- If you define optional extras for your environment, include them at run time:
```bash
uv run --with '.[your_extra]' python scripts/your_script.py ...
```

Notes
- `uv sync` is only needed after changing dependencies or on first setup. For ephemeral runs without a full sync, you may also use `uv run` which will resolve and execute in a temporary environment.