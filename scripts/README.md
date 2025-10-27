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

## find_run

List and locate experiment runs using the append-only filesystem index at `results/index/runs_index.jsonl`.

Usage:

```bash
uv run python scripts/find_run.py
```

Default (no args):
- Shows non-junk runs (heuristic: has artifacts or metrics) grouped by script and ordered by newest first.
- Displays per-run duration (start→end) and usage count (number of artifact_used events).

Options (coming soon):
- `--results-root PATH` — path to the results directory (index assumed at `<results-root>/index`).
- `--contains STR` — filter runs whose run directory name contains `STR` (case-insensitive).
- `--env STR` — filter runs whose environment name contains `STR` (case-insensitive).
- Additional filters (experiment, commit, config, status) will be added alongside `open_log.py`.

## reindex_runs

Backfill the runs index for existing runs so they show up in `find_run`.

Usage:

```bash
uv run python scripts/reindex_runs.py --results-root results
```

Flags:
- `--dry-run` — print records instead of writing to the index.
