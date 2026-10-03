# Optimizing XGBoost Machine Learning Models with AI Agents

**TL;DR:** An AI coding agent (e.g. Claude Code, Codex) autonomously tunes an XGBoost model, and its gains are then checked on a held-out test set it never sees. The setup can be used to compare how well different agents/LLMs do ML research.

This is a follow-up to [xgboost-autoresearch](https://github.com/szilard/xgboost-autoresearch): a minimal, self-contained setup for a single run. It is an adaptation of A. Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) project to XGBoost.

The idea: give an AI agent a small but real XGBoost training setup and let it experiment autonomously for a fixed time budget (2 hours). It modifies the code, trains, checks if the result improved, keeps or discards, and repeats. When the time is up it stops, and you get a log of experiments and (hopefully) a better model. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` markdown file that provides context to the AI agent and sets up your autonomous research org.

## How a run works

The agent's instructions are in `program.md`:

- **Task:** predict whether a flight departs 15+ minutes late (airline data, balanced, 200K train rows from 2005 / 50K eval and 50K holdout rows from 2006), measured by AUC. The agent optimizes AUC on the eval set; the holdout set is scored only after the run.
- **Loop:** for a fixed 2-hour budget, the agent edits `train.py` (data prep, feature engineering, hyperparameters, model training), commits, runs it via `harness.py`, and keeps the commit if eval AUC improved or resets it otherwise. It also researches ideas on the web and logs every experiment in `output/results.tsv` and `output/research-log.md`.
- **Guardrails:** `harness.py` enforces time limits (1 min training, 5 min evaluation per run). Evaluation calls the agent's feature code on one row at a time, so features computed across rows (counts, group means) don't carry over to scoring. The holdout set and the human-only scripts in `human/` (data prep, holdout scoring) are off-limits to the agent.
- **Ground truth:** after the run, you score every kept model on the holdout set to check that the eval AUC gains generalize (`output/groundtruth_all.tsv`, `output/auc_history.png`).

To run repeated trials with various agents/LLMs, use an orchestrator such as
[xgboost-autoresearch-minimal3-runs](https://github.com/szilard/xgboost-autoresearch-minimal3-runs).

Recommended machine: m8i.2xlarge (8 cores, 32 GB RAM). The per-run time limits depend on the hardware, so compare results only across runs on the same machine type.

## Quick start

```bash

# 1. Install dependencies
pip install pandas xgboost scikit-learn cloudpickle matplotlib --break-system-packages

# 2. Build train.csv (2005) and eval/holdout.csv (2006) from the airline data on S3
python3 human/prepare.py

# 3. Manually run a single training experiment (to verify everything works)
python3 train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode. (If `train.py` refuses because the experiment clock is running, a previous run left `output/` behind: archive or delete it.)

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permission prompts), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

After the setup the agent starts the clock (`python3 harness.py start`), runs experiments for 2 hours, then wraps up and stops the clock (`python3 harness.py stop`). If an agent forgets to stop, the clock keeps running and the report counts up to the moment you run it; run `python3 harness.py stop` yourself as soon as you notice, so the total stops growing.

## After the run

```bash
python3 harness.py report            # total time, split into XGBoost training / evaluation vs the AI
./human/run_groundtruth_all.sh       # holdout AUC of every kept experiment -> output/groundtruth_all.tsv
python3 human/plot_auc_history.py    # eval vs holdout AUC -> output/auc_history.png
```

To score a single commit on the holdout set from its saved artifact (no retraining): `python3 human/check_groundtruth.py [commit]`.

All of the run's outputs are in `output/` (`results.tsv`, `research-log.md`, `run.log`, `timing/`, `groundtruth_all.tsv`, `auc_history.png`). You (the human, not the agent) archive it with `mkdir -p results && mv output results/<run-name>` (optionally adding the final `train.py` and selected artifacts) and commit it there, or delete it, so the next run starts clean. Runs are archived in `results/`, one folder per run (none yet in this repo).

The timing works the same for any agent (Claude Code, Codex, ...): `harness.py` logs the wall-clock time of every run to `output/timing/runs.tsv`, and everything else between `start` and `stop` is the AI's time (token generation, tool calls, web research, API latency).

## Project structure

```
program.md             - agent instructions (edited and iterated on by the human)
train.py               - XGBoost training, the single code file the AI agent edits
harness.py             - experiment clock (start/status/stop/report), timed runs, saves the artifact,
                         row-by-row eval scoring (not modified by the AI agent)
human/                 - human-only tools, off-limits to the AI agent
  prepare.py             - builds the data splits, read straight from S3 (not stored locally)
  check_groundtruth.py   - holdout scoring of a saved artifact
  run_groundtruth_all.sh - holdout scoring of all kept experiments
  plot_auc_history.py    - plot of eval vs holdout AUC
data/                  - train/eval/holdout.csv splits, created by human/prepare.py (gitignored)
artifacts/             - saved model + prepare per commit, artifacts/<commit>.pkl (gitignored)
output/                - outputs of the current run: results.tsv, research-log.md, run.log, timing/ (experiment clock and
                         per-run timings), groundtruth_all.tsv, auc_history.png (not gitignored, the human moves it to
                         results/<run-name>/ or deletes it after a run)
results/               - archived runs, one folder per run (human only)
```

## Design choices

- **Single file to modify.** The only code the agent touches is `train.py`. This keeps the scope manageable and diffs reviewable.
- **Self-contained.** No external dependencies beyond XGBoost, pandas (training and data prep), scikit-learn, cloudpickle and matplotlib (plot). No distributed training, no complex configs. The only network access needed is `human/prepare.py` reading the source data from S3 and the agent's web research.
