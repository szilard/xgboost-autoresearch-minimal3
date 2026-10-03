# autoresearch XGBoost

an adaptation of A. Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) project to XGBoost

The idea: give an AI agent a small but real XGBoost training setup and let it experiment autonomously for a fixed time budget (2 hours). It modifies the code, trains, checks if the result improved, keeps or discards, and repeats. When the time is up it stops, and you get a log of experiments and (hopefully) a better model. The training code here is a small, single-file XGBoost setup. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` markdown file that provides context to the AI agents and sets up your autonomous research org. 

## How it works

The repo is deliberately kept small:

- **`human/prepare.py`** - builds train.csv from the 2005 airline data and eval/holdout.csv from the 2006 data, read straight from S3 (not stored locally). Human only; the AI agent must not read it.
- **`train.py`** - the single code file the agent edits (besides its own `output/results.tsv` and `output/research-log.md`). Contains the XGBoost model training. Everything is fair game that will lead to a model that generalizes on unseen data: data preparation, feature engineering, choosing hyperparameters, and model training. **This file is edited and iterated on by the agent**.
- **`program.md`** - baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.
- **`harness.py`** - runs and times the experiments (`python3 harness.py run`), keeps the 2-hour experiment clock (`start`/`status`/`stop`), saves the trained model and `prepare` to `artifacts/<commit>.pkl` (gitignored) and scores `eval.csv` row by row. Not modified by the AI agent.
- **`human/check_groundtruth.py`** - script to check the "ground truth" AUC on `holdout.csv` by the human, from the saved artifact (no retraining): `python3 human/check_groundtruth.py [commit]`. AI should not access this file.
- **`human/run_groundtruth_all.sh`** + **`human/plot_auc_history.py`** - after a run, score every kept experiment in `output/results.tsv` on the holdout set (`output/groundtruth_all.tsv`) and plot eval vs holdout AUC (`output/auc_history.png`). Human only.

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

All of the run's outputs are in `output/` (`results.tsv`, `research-log.md`, `run.log`, `timing/`, `groundtruth_all.tsv`, `auc_history.png`). Then you (the human, not the agent) archive it with `mkdir -p results && mv output results/<run-name>` (optionally adding the final `train.py` and selected artifacts) and commit it there, or delete it, so the next run starts clean.

The timing works the same for any agent (Claude Code, Codex, ...): `harness.py` logs the wall-clock time of every run to `output/timing/runs.tsv`, and everything else between `start` and `stop` is the AI's time (token generation, tool calls, web research, API latency).

## Project structure

```
train.py               - XGBoost training (AI agent modifies this)
harness.py             - experiment clock, timed runs, saves the artifact, row-by-row eval scoring
program.md             - agent instructions
human/                 - human-only tools, off-limits to the AI agent
  prepare.py             - builds the data splits
  check_groundtruth.py   - holdout scoring of a saved artifact
  run_groundtruth_all.sh - holdout scoring of all kept experiments
  plot_auc_history.py    - plot of eval vs holdout AUC
data/                  - train/eval/holdout.csv splits (gitignored)
artifacts/             - saved model + prepare per commit (gitignored)
output/                - outputs of the current run: results.tsv, research-log.md, run.log, timing/ (experiment clock and
                         per-run timings), groundtruth_all.tsv, auc_history.png (not gitignored, the human moves it to
                         results/<run-name>/ or deletes it after a run)
results/               - archived runs, one folder per run (human only)
```

## Design choices

- **Single file to modify.** The only code the agent touches is `train.py`. This keeps the scope manageable and diffs reviewable.
- **Self-contained.** No external dependencies beyond XGBoost, pandas (training and data prep), scikit-learn, cloudpickle and matplotlib (plot). No distributed training, no complex configs. The only network access needed is `human/prepare.py` reading the source data from S3 and the agent's web research.
