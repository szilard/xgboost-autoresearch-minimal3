# Optimizing XGBoost Machine Learning Models with AI Agents

**TL;DR:** An AI coding agent (e.g. Claude Code, Codex) autonomously tunes an XGBoost model, and its gains are then checked on a held-out test set it never sees. The setup can be used to compare how well different agents/LLMs do ML research.

This is a follow-up to [xgboost-autoresearch](https://github.com/szilard/xgboost-autoresearch): a minimal, self-contained setup for a single run.

How a run works (the agent's instructions are in `program.md`):

- **Task:** predict whether a flight departs 15+ minutes late (airline data, balanced, 200K train rows from 2005 / 50K eval and 50K holdout rows from 2006), measured by AUC. The agent optimizes AUC on the eval set; the holdout set is scored only after the run.
- **Loop:** for a fixed 2-hour budget, the agent edits `train.py` (data prep, feature engineering, hyperparameters, model training), commits, runs it via `harness.py`, and keeps the commit if eval AUC improved or resets it otherwise. It also researches ideas on the web and logs every experiment in `results.tsv` and `research-log.md`.
- **Guardrails:** `harness.py` enforces time limits (1 min training, 5 min evaluation per run). Evaluation calls the agent's feature code on one row at a time, so features computed across rows (counts, group means) don't carry over to scoring. The holdout set and the scripts that score it are off-limits to the agent.
- **Ground truth:** after the run, you score every kept model on the holdout set to check that the eval AUC gains generalize (`groundtruth_all.tsv`, `auc_history.png`).

To run repeated trials with various agents/LLMs, use an orchestrator such as
[xgboost-autoresearch-minimal2-runs](https://github.com/szilard/xgboost-autoresearch-minimal2-runs).

Recommended machine: m8i.2xlarge (8 cores, 32 GB RAM). The per-run time limits depend on the hardware, so compare results only across runs on the same machine type.

See [README-autoresearch.md](README-autoresearch.md) for setup and details. Runs are archived in `results/`, one folder per run (none yet in this repo).
