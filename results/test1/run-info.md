# test1

First full 2-hour test run of this repo, done by hand (not through the orchestrator).

- **Date:** 2026-10-03 23:00:39 -> 2026-10-04 00:49:26 UTC
- **Repo version:** `b261f5f`, run on a throwaway copy with fresh git history (its first commit: `2e14489`), run branch `oct3`
- **Agent:** Claude Code general-purpose subagent, launched from a Claude Code session running Claude Fable 5.1 (subagents use the session's model by default; not confirmed from a session log)
- **Prompt:** the kickoff prompt from the README, plus the answers a human would give during setup (run tag approved, setup confirmed in advance, work the whole budget without asking)
- **Machine:** 8 cores, 30 GB RAM
- **Clock:** 1h48m47s of the 2-hour budget (the agent stopped about 11 minutes early); 56.5% in runs, 43.5% the agent's own time
- **Runs:** 146 (baseline + 145 experiments): 34 keep, 111 discard, 1 crash, no timeouts

| | Commit | Eval AUC | Holdout AUC |
|---|---|---|---|
| Baseline | `2e14489` | 0.6743 | 0.6725 |
| Final and best | `65129c7` | 0.6918 | 0.6874 |

## Caveats

- The run used the instructions as of `b261f5f`, before the strict keep rule (`884b806`) and the fixes of `316aa83`. Under the "simplicity criterion" in force then, 8 of the 34 kept commits have a slightly lower Eval AUC than the kept commit before them (by 0.0001-0.0006).
- Twice the agent logged a discard without resetting (`10cec75`, `661122e`), so the change stayed in later commits for a while. It found and corrected both; `661122e` remains in the branch history and is reverted by `8e0078d`.
- The agent ran some read-only scratch commands on `train.csv`, which the instructions did not yet explicitly allow.
- No leak check was run. By its own report the agent did not read `human/`, `data/holdout.csv` or `data/eval.csv`; this was not audited from its session log.
- The final `train.py` and the artifacts are not archived here.
