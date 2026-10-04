# test2

First 1-hour test run under the current rules (strict keep rule, 1-hour budget), done by hand (not through the orchestrator).

- **Date:** 2026-10-04 06:30:43 -> 07:29:30 UTC
- **Repo version:** `c7abc3d`, run on a throwaway copy without `results/` and with fresh git history (its first commit: `c0880c0`), run branch `oct4`
- **Agent:** Claude Code general-purpose subagent, launched from a Claude Code session running Claude Fable 5.1 (subagents use the session's model by default; not confirmed from a session log)
- **Prompt:** the kickoff prompt from the README, plus the answers a human would give during setup (run tag approved, setup confirmed in advance, work the whole budget without asking)
- **Machine:** 8 cores, 30 GB RAM
- **Clock:** 58m47s of the 1-hour budget; 76.7% in runs, 23.3% the agent's own time
- **Runs:** 63 (baseline + 61 experiments + 1 crashed attempt that was fixed and rerun): 17 keep, 45 discard, 1 crash, no timeouts

| | Commit | Eval AUC | Holdout AUC |
|---|---|---|---|
| Baseline | `c0880c0` | 0.6743 | 0.6725 |
| Final and best | `6080416` | 0.6879 | 0.6869 |

For comparison, `test1` at its 60-minute mark: Eval AUC 0.6899, holdout AUC 0.6868.

## Notes

- Keep rule: every kept commit has a strictly higher Eval AUC than the kept commit before it (checked from `results.tsv`).
- Resets: the run branch contains exactly the 17 kept commits, in the order of `results.tsv` (checked from the branch's git log).
- By its own report the agent corrected two lines of the final summary in `research-log.md` after stopping the clock, and did its every-10-experiments research in parallel with the next run instead of before it.
- No leak check was run. By its own report the agent did not read `human/`, `data/holdout.csv` or `data/eval.csv`; this was not audited from its session log.
- `train.py` here is the final version (commit `6080416`). The artifacts are not archived.
