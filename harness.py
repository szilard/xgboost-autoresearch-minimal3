"""Save the trained artifact and score it row by row. Not modified by the agent.

The artifact is `{"model": model, "prepare": prepare}` pickled with cloudpickle,
which stores `prepare` by value together with every module-level lookup it uses
(cat_levels etc.), so it can be scored later without train.py or the training data.
Artifacts live in the gitignored artifacts/ folder, named by full commit hash.

It also keeps the experiment clock and times every run (output/timing/ folder, gitignored like
the rest of output/; the human archives or deletes it after a run):

    python3 harness.py start    # start the clock (at "go", after the setup)
    python3 harness.py run      # run train.py: timed, killed if training exceeds
                                # train_timeout_s or evaluation eval_timeout_s,
                                # refused once the time budget is used up
    python3 harness.py status   # elapsed / remaining time
    python3 harness.py stop     # stop the clock (the agent's last action)
    python3 harness.py report   # total time, split into XGBoost runs vs the AI
"""
import csv
import json
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
import multiprocessing as mp
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

repo_dir = Path(__file__).parent
data_dir = repo_dir / "data"
artifacts_dir = repo_dir / "artifacts"
timing_dir = repo_dir / "output" / "timing"
clock_file = timing_dir / "clock.json"
runs_file = timing_dir / "runs.tsv"
n_workers = os.cpu_count()
time_budget_s = 3600
train_timeout_s = 60     # train.py up to the save_and_evaluate() call
eval_timeout_s = 5 * 60  # save_and_evaluate(): saving the artifact + row-by-row eval
eval_marker = "Training done, evaluating..."


def git(*args):
    return subprocess.check_output(["git", *args], cwd=repo_dir, text=True).strip()


def find_artifact(commit):
    """Artifact path for a full or abbreviated commit hash."""
    matches = sorted(artifacts_dir.glob(f"{commit}*.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected 1 artifact for {commit}, found {len(matches)}")
    return matches[0]


def load_artifact(commit):
    with open(find_artifact(commit), "rb") as f:
        return pickle.load(f)


_prepare = None


def _prepare_rows(df):
    """prepare() each row on its own; runs in a worker process."""
    out = [_prepare(df.iloc[[i]]) for i in range(len(df))]
    return pd.concat([X for X, _ in out]), np.concatenate([y for _, y in out])


def score_by_row(artifact, df):
    """prepare() row by row across processes, then predict in one batch. Returns AUC."""
    global _prepare
    _prepare = artifact["prepare"]
    per_worker = -(-len(df) // n_workers)
    chunks = [df.iloc[i:i + per_worker] for i in range(0, len(df), per_worker)]
    # fork so the workers inherit _prepare as set above
    with mp.get_context("fork").Pool(n_workers) as pool:
        out = pool.map(_prepare_rows, chunks)
    X = pd.concat([X for X, _ in out])
    y = np.concatenate([y for _, y in out])
    y_prob = artifact["model"].predict_proba(X)[:, 1]
    return roc_auc_score(y, y_prob)


def save_and_evaluate(model, prepare):
    """Save the artifact for the current commit, then score eval.csv with the reloaded copy."""
    clock = read_clock()
    if "start" in clock and "stop" not in clock and os.environ.get("HARNESS_RUN") != "1":
        sys.exit("ERROR: the experiment clock is running, launch runs with `python3 harness.py run`")
    # tells `harness.py run` that training is over: switch from the training to the eval time limit
    print(eval_marker, flush=True)

    blob = cloudpickle.dumps({"model": model, "prepare": prepare})
    # score exactly what the holdout evaluation will load, not the in-memory objects
    artifact = pickle.loads(blob)

    if git("status", "--porcelain", "--", "train.py"):
        print("WARNING: train.py has uncommitted changes, artifact not saved")
    else:
        artifacts_dir.mkdir(exist_ok=True)
        path = artifacts_dir / f"{git('rev-parse', 'HEAD')}.pkl"
        path.write_bytes(blob)
        print(f"Artifact: {path.relative_to(repo_dir)} ({len(blob) / 1e6:.1f} MB)")

    eval_df = pd.read_csv(data_dir / "eval.csv")
    t0 = time.time()
    eval_auc = score_by_row(artifact, eval_df)
    print(f"Eval time: {time.time() - t0:.1f}s")
    print(f"Eval AUC: {eval_auc:.4f}")


def fmt(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def read_clock():
    return json.loads(clock_file.read_text()) if clock_file.exists() else {}


def elapsed(clock):
    return clock.get("stop", time.time()) - clock["start"]


def cmd_start():
    timing_dir.mkdir(parents=True, exist_ok=True)
    clock_file.write_text(json.dumps({"start": time.time()}))
    print(f"Clock started, time budget {fmt(time_budget_s)}")


def cmd_status():
    clock = read_clock()
    if "start" not in clock:
        print("Clock not started")
        return
    left = time_budget_s - elapsed(clock)
    print(f"Elapsed: {fmt(elapsed(clock))}, remaining: {fmt(max(left, 0))}")
    if "stop" in clock:
        print("Clock stopped")
    elif left <= 0:
        print("TIME IS UP: do not start new experiments; wrap up and run `python3 harness.py stop`")


def cmd_stop():
    clock = read_clock()
    if "start" not in clock:
        sys.exit("Clock not started")
    clock.setdefault("stop", time.time())
    clock_file.write_text(json.dumps(clock))
    print(f"Clock stopped after {fmt(elapsed(clock))}")


def cmd_run():
    """Run train.py as a timed subprocess, streaming its output, and log the timing."""
    clock = read_clock()
    if "start" not in clock:
        sys.exit("ERROR: clock not started, run `python3 harness.py start` first")
    if "stop" in clock or elapsed(clock) >= time_budget_s:
        print("TIME IS UP: do not start new experiments; wrap up and run `python3 harness.py stop`")
        sys.exit(3)

    t0 = time.time()
    # own process group, so a timeout also kills the row-scoring worker processes
    proc = subprocess.Popen(
        [sys.executable, "-u", "train.py"], cwd=repo_dir, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        env={**os.environ, "HARNESS_RUN": "1"},
    )
    timed_out = None  # phase ("training"/"eval") in which the run was killed

    def kill(phase):
        nonlocal timed_out
        timed_out = phase
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(train_timeout_s, kill, ["training"])
    timer.start()
    train_s = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if train_s is None and line.rstrip("\n") == eval_marker:
            train_s = time.time() - t0
            timer.cancel()
            timer = threading.Timer(eval_timeout_s, kill, ["eval"])
            timer.start()
    proc.wait()
    timer.cancel()
    run_s = time.time() - t0
    if train_s is None:
        train_s = run_s

    status = f"timeout-{timed_out}" if timed_out else "ok" if proc.returncode == 0 else "crash"
    new_file = not runs_file.exists()
    with open(runs_file, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new_file:
            w.writerow(["commit", "start", "end", "train_s", "eval_s", "status"])
        w.writerow([git("rev-parse", "--short=7", "HEAD"), f"{t0:.1f}", f"{t0 + run_s:.1f}",
                    f"{train_s:.1f}", f"{run_s - train_s:.1f}", status])
    if timed_out:
        limit = train_timeout_s if timed_out == "training" else eval_timeout_s
        print(f"TIMEOUT: {timed_out} killed after {limit}s")
    print(f"Run time: {run_s:.1f}s (training {train_s:.1f}s, eval {run_s - train_s:.1f}s, {status})")
    sys.exit(124 if timed_out else proc.returncode)


def cmd_report():
    """Total time since the clock started, split into train.py runs and the rest (the AI)."""
    clock = read_clock()
    if "start" not in clock:
        sys.exit("Clock not started")
    start = clock["start"]
    end = clock.get("stop", time.time())
    with open(runs_file) if runs_file.exists() else open(os.devnull) as f:
        runs = [r for r in csv.DictReader(f, delimiter="\t") if float(r["start"]) >= start]

    # union of the run intervals, in case the agent ran experiments in parallel
    runs_wall = 0.0
    cur_start = cur_end = None
    for s, e in sorted((float(r["start"]), min(float(r["end"]), end)) for r in runs):
        if cur_end is None or s > cur_end:
            runs_wall += (cur_end - cur_start) if cur_end is not None else 0.0
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    runs_wall += (cur_end - cur_start) if cur_end is not None else 0.0

    total = end - start
    train_s = sum(float(r["train_s"]) for r in runs)
    eval_s = sum(float(r["eval_s"]) for r in runs)
    run_s = train_s + eval_s
    counts = {k: sum(r["status"] == k for r in runs)
              for k in ("ok", "crash", "timeout-training", "timeout-eval")}

    pct = lambda x: f"{100 * x / total:5.1f}%" if total else ""
    print(f"Clock:        {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))} -> "
          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}"
          f"{'' if 'stop' in clock else ' (still running)'}")
    print(f"Total:        {fmt(total)}")
    print(f"XGBoost runs: {fmt(runs_wall)} {pct(runs_wall)}   "
          f"({len(runs)} runs: {counts['ok']} ok, {counts['crash']} crash, "
          f"{counts['timeout-training']} training timeout, {counts['timeout-eval']} eval timeout)")
    print(f"  training:   {fmt(train_s)} {pct(train_s)}   "
          f"(train.py up to evaluation: startup, data, features, fit)")
    print(f"  evaluation: {fmt(eval_s)} {pct(eval_s)}   (saving the artifact + row-by-row scoring of eval.csv)")
    print(f"AI:           {fmt(total - runs_wall)} {pct(total - runs_wall)}   "
          f"(everything else: token generation, tool calls, web research, API latency)")
    if run_s > runs_wall + 1:
        print(f"Note: runs overlapped; summed run time is {fmt(run_s)}")


if __name__ == "__main__":
    commands = {"start": cmd_start, "status": cmd_status, "stop": cmd_stop,
                "run": cmd_run, "report": cmd_report}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(f"usage: python3 harness.py {{{','.join(commands)}}}")
    commands[sys.argv[1]]()
