# Chess Probe Environment

Overview
- Minimal environment for a probe that injects a soft token or residual into a frozen chess model.
- Based on DeepMind's searchless_chess (Apache-2.0). We vendor only what’s necessary and preserve licensing.

Plan
- Start with minimal engine wrappers and dataset loaders sufficient for probe training and eval.
- Prefer referencing searchless_chess as a submodule or copying the smallest necessary files with headers intact.

Suggested structure
- src/: probe wrappers and minimal integration with engine/model
- scripts/: training/eval entrypoints for TRL single-turn PPO
- data/: dataset instructions (links), not the data itself

Licensing
- See THIRD_PARTY_NOTICES.md for Apache-2.0 attribution and requirements.

References
- Chen, H., Vondrick, C., Mao, C. (2024). SelfIE: Self-Interpretation of Large Language Model Embeddings. arXiv. [arXiv:2403.10949v2](https://arxiv.org/pdf/2403.10949v2)
