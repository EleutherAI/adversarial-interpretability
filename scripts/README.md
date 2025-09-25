# Scripts (Placeholder)

Intent
- Entry points for training, evaluation, and utilities; call into libs/ components.

Status
- Placeholder. Add minimal CLIs as environments land.

CLI conventions
- Common flags:
  - `--config path/to/config.yaml`
  - `--experiment-name <slug>`
  - `--output-dir results/` (default)
  - `--seed 123` and `--notes "short note"`
- On startup, scripts should:
  1) Create a run directory under `results/<env>/<experiment_name>/<timestamp-runid>/`
  2) Copy the resolved config to `config.yaml`
  3) Write `metadata.json` with git commit/dirty flag, env info, CLI args, seeds
  4) Start structured logging to `metrics.jsonl` and `logs/`
- During execution:
  - Append metrics as JSON lines with `step`, `split`, metric dict, elapsed seconds, ISO timestamp
  - Save checkpoints under `artifacts/` and plots under `plots/`
- On completion:
  - Write a final `summary.json` (best metrics, artifact paths)
