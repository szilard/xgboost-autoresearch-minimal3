# Ground truth: score a saved artifact on holdout.csv, row by row. Human only.
# Usage: python3 human/check_groundtruth.py [commit]   (default: HEAD)
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # harness.py is in the repo root
from harness import data_dir, git, load_artifact, score_by_row

commit = sys.argv[1] if len(sys.argv) > 1 else git("rev-parse", "HEAD")
artifact = load_artifact(commit)

holdout = pd.read_csv(data_dir / "holdout.csv")
t0 = time.time()
holdout_auc = score_by_row(artifact, holdout)
print(f"Holdout time: {time.time() - t0:.1f}s")
print(f"Holdout AUC: {holdout_auc:.4f}")
