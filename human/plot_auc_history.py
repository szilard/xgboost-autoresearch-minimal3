from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

output_dir = Path(__file__).parent.parent / "output"
df = pd.read_csv(output_dir / "holdout_scores.tsv", sep="\t")
df.insert(0, " n ", range(1, len(df) + 1))
df.columns = df.columns.str.strip()

for col in ["eval_auc", "holdout_auc"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

plt.figure(figsize=(10, 6))
keep = df["status"] == "keep"
discard = df["status"] == "discard"  # crashes (eval_auc 0) are not plotted
plt.plot(df.loc[discard, "n"], df.loc[discard, "eval_auc"], marker="o", color="lightgrey", linestyle="none", label="eval discard")
plt.plot(df.loc[keep, "n"], df.loc[keep, "eval_auc"], color="cornflowerblue", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(df.loc[keep, "n"], df.loc[keep, "eval_auc"], marker="o", color="steelblue", linestyle="none", label="eval keep", zorder=2)
plt.plot(df.loc[keep, "n"], df.loc[keep, "holdout_auc"], color="#d94040", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(df.loc[keep, "n"], df.loc[keep, "holdout_auc"], marker="o", color="#d94040", linestyle="none", label="holdout", zorder=2)
plt.xlabel("n")
plt.ylabel("AUC")
plt.ylim(0.67, 0.69)
plt.title("AUC vs n")
plt.grid(True, color="lightgrey", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "auc_history.png", dpi=150)
plt.show()

for col in ["eval_auc", "holdout_auc"]:
    idx = df[col].idxmax()
    print(f"{col}: max={df.loc[idx, col]:.4f} at n={df.loc[idx, 'n']}")
