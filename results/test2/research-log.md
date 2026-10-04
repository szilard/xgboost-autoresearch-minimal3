# Research log, run `oct4`

Task: predict `dep_delayed_15min` (2005 train, 2006 eval), metric Eval AUC from `python3 harness.py run`.
Each entry: experiment number, commit (short hash, same as in `output/results.tsv`), hypothesis, class, result, decision.

## Baseline (c0880c0), Eval AUC 0.6743, keep
Starter `train.py` as is: 100 trees, depth 6, lr 0.1, all six calendar/carrier/airport columns as XGBoost categoricals.

## Data notes (train.csv only)
- 200K rows, balanced, no missing values. Month/DayofMonth/DayOfWeek are strings like `c-11`.
- Delay rate rises almost monotonically with the scheduled hour: 0.19 at 05h to 0.65 at 20h, then falls. Strongest single signal.
- Month: 0.40 (Apr) to 0.59 (Jul, Dec). Day of week: 0.45 to 0.55. Carrier: 0.14 (HA) to 0.63 (AS).
- Origin/Dest have 283 levels each, 20 carriers.
- Train is 2005, eval is 2006: anything specific to a 2005 calendar date (one storm on one day) cannot transfer.
  DayofMonth x Month interactions are therefore suspect; weekday/hour/airport/carrier structure should transfer.

## Research before the first experiment
- XGBoost tuning guides (AWS SageMaker XGBoost tuning docs, apxml gradient boosting course): tune
  learning_rate together with n_estimators, then max_depth / min_child_weight, subsample / colsample_bytree,
  reg_alpha / reg_lambda; lower learning rate with more trees is the usual gain.
- Flight delay feature engineering write-ups (Erdos Institute project reports, arXiv 2601.00875): hour of
  scheduled departure, cyclical/ordinal time features, route, and K-fold target encoding of high-cardinality
  categoricals (airports, route, carrier) are the standard features.

## Experiments

### Exp 1 (374abf5), Eval AUC 0.6754, keep
- Class: exploration. Hypothesis: calendar columns as ordinal integers instead of categoricals, plus Hour and
  minutes since midnight, give smoother splits that transfer better to 2006 than categorical partitions.
- Result: 0.6743 -> 0.6754. Kept.

### Exp 2 (99a9b2f), Eval AUC 0.6735, discard
- Class: ablation. Hypothesis: DayofMonth only carries 2005-specific dates, dropping it should help on 2006.
- Result: 0.6754 -> 0.6735. Wrong: DayofMonth carries transferable signal (holidays, month position). Reset to 374abf5.

### Exp 3 (aa082be), Eval AUC 0.6817, keep
- Class: exploration (hyperparameters). Hypothesis (tuning guides): lower learning rate with more trees plus
  row/column subsampling and min_child_weight regularise and should transfer better across years.
- Result: 0.6754 -> 0.6817. Kept. Training 3.5s, far inside the 60s limit.

### Exp 4 (5b54f60), Eval AUC 0.6814, discard
- Class: exploration (encoding). Source: target encoding advice in the flight delay write-ups; XGBoost
  categorical tutorial (optimal partitioning already sorts categories by gradient statistics).
- Hypothesis: smoothed target encodings (m=20) of carrier/origin/dest next to the categoricals help.
- Result: 0.6817 -> 0.6814. Redundant with XGBoost's own categorical partitioning. Reset to aa082be.

### Exp 5 (f1086d4), Eval AUC 0.6774, discard
- Class: exploration (calendar features). Hypothesis: DayofMonth helped in exp 2, presumably via holidays, so an
  explicit signed distance to the nearest US travel holiday (pure calendar lookup keyed by month/day/weekday,
  valid for any year) should transfer to 2006 better than raw Month x DayofMonth.
- Result: 0.6817 -> 0.6774, a clear drop. HolDist is a near-continuous date index, so it makes it easier for the
  trees to isolate individual 2005 dates (weather events) that do not repeat in 2006. Lesson: features that
  identify the exact date more sharply hurt. Reset to aa082be.

### Exp 6 (58c5ef0), Eval AUC 0.6742, discard
- Class: exploration (hyperparameters). Hypothesis: deeper trees (9) capture airport x hour x carrier interactions.
- Result: 0.6817 -> 0.6742. More capacity overfits 2005. Next: go the other way (shallower). Reset to aa082be.

### Exp 7 (764f7c0), Eval AUC 0.6845, keep
- Class: follow-up to exp 6. Hypothesis: shallower trees (4) limit the 2005-specific interactions.
- Result: 0.6817 -> 0.6845. Kept. Less capacity generalises better across the year shift.

### Exp 8 (492c0c4), Eval AUC 0.6838, discard
- Class: follow-up to exp 7. Hypothesis: if 4 beats 6, 3 may beat 4.
- Result: 0.6845 -> 0.6838. Depth 4 is the sweet spot at 500 trees / lr 0.03. Reset to 764f7c0.

### Exp 9 (d28759b), Eval AUC 0.6799, discard
- Class: exploration (encoding). Hypothesis: a fixed global ordering (smoothed target encoding) of the 283 airports
  is a more regularised representation than free categorical partitions.
- Result: 0.6845 -> 0.6799. The categorical partitions are worth keeping: airports group differently by context
  (hour, month), which a single target-rate ordering cannot express. Reset to 764f7c0.

### Exp 10 (05fdb9a), Eval AUC 0.6834, discard
- Class: follow-up (hyperparameters). Hypothesis: at depth 4 the model may be under-boosted at 500 x 0.03.
- Result: 0.6845 -> 0.6834. More boosting rounds overfit 2005 as well. Reset to 764f7c0.

## Synthesis after 10 experiments
- What helps: less capacity. Depth 6 -> 4 (+0.003), lower learning rate with subsampling (+0.006), ordinal calendar
  columns (+0.001).
- What hurts: anything that lets the model pin down 2005 more precisely: depth 9 (-0.0075), 1000 trees (-0.001),
  holiday distance (a date index, -0.004). Target encodings add nothing next to, and are worse than, the
  categorical partitions for airports.
- Theory: train (2005) -> eval (2006) shift dominates. The transferable structure is hour of day, airport, carrier,
  weekday and season; date-level detail is 2005 weather. The best model is a heavily regularised, low-order one.
- Next: research regularisation knobs that act on categorical splits and on interaction order (max_cat_threshold,
  max_cat_to_onehot, gamma, reg_lambda, colsample_bynode/bylevel, interaction constraints), then try explicit
  low-order interaction features that should transfer (carrier x origin hub, origin x hour).

### Exp 11 (82956e0), Eval AUC 0.6842, discard
- Class: follow-up to exp 10. Hypothesis: if 1000 trees is worse than 500, fewer (300) may be better.
- Result: 0.6845 -> 0.6842. 500 trees at lr 0.03 is about the optimum for depth 4. Reset to 764f7c0.

## Research after 10 experiments
- XGBoost parameter docs (xgboost.readthedocs.io/en/stable/parameter.html): regularisation knobs not yet tried:
  gamma, reg_lambda/reg_alpha, colsample_bylevel/bynode, max_cat_threshold (max categories considered per
  split), max_cat_to_onehot, interaction_constraints, monotone_constraints, DART (rate_drop).
- HyperTime (arXiv 2305.18421) and related work on temporal distribution shift: under year-to-year shift the
  configurations that are best in-distribution are not the most robust; favour conservative settings.
- Plan: (a) interaction constraints so Month and DayofMonth cannot be combined into an exact 2005 date,
  (b) max_cat_threshold / reg_lambda / gamma to regularise the airport partitions, (c) colsample_bynode.

### Exp 12 (a4364a3), Eval AUC 0.6803, discard
- Class: exploration (interaction constraints, XGBoost docs). Hypothesis: forbidding Month x DayofMonth stops the
  model from learning exact 2005 dates.
- Result: 0.6845 -> 0.6803. The opposite: the Month x DayofMonth interaction transfers to 2006 (holiday periods,
  within-season timing). This revises the theory from exp 5: date-level structure is useful; HolDist hurt for
  another reason (probably too fine a handle on single days). Reset to 764f7c0.

### Exp 13 (40461d8), Eval AUC 0.6842, discard
- Class: follow-up to exp 12. Hypothesis: since Month x DayofMonth transfers, a single DayOfYear axis lets depth-4
  trees isolate date windows (Christmas, Thanksgiving weeks) with fewer splits.
- Result: 0.6845 -> 0.6842. No gain; Month + DayofMonth are enough. Reset to 764f7c0.

### Exp 14 (5746222), Eval AUC 0.6851, keep
- Class: exploration (regularisation, XGBoost parameter docs). Hypothesis: stronger L2 on leaf weights shrinks
  small-sample leaves (rare airports), which are the least stable across years.
- Result: 0.6845 -> 0.6851. Kept.

### Exp 15 (0710d1c), Eval AUC 0.6856, keep
- Class: follow-up to exp 14. Hypothesis: the L2 trend continues.
- Result: 0.6851 -> 0.6856. Kept.

### Exp 16 (ef901f3), Eval AUC 0.6851, discard
- Class: follow-up to exp 15. Hypothesis: even stronger L2 (200).
- Result: 0.6856 -> 0.6851. Past the optimum; reg_lambda stays at 50. Reset to 0710d1c.

### Exp 17 (1544ce8), Eval AUC 0.6857, keep
- Class: exploration (regularisation). Hypothesis: with three redundant time columns (CRSDepTime, Hour,
  DepMinutes), stronger column subsampling decorrelates trees and gives the weaker features more say.
- Result: 0.6856 -> 0.6857. Marginal but higher, kept by the keep rule.

### Exp 18 (5f2a0c8), Eval AUC 0.6859, keep
- Class: exploration (regularisation). Hypothesis: larger minimum leaf weight (30, about 120 rows) blocks splits
  that carve out small groups of 2005 flights.
- Result: 0.6857 -> 0.6859. Kept.

### Exp 19 (e5f0fcb), Eval AUC 0.6854, discard
- Class: follow-up to exp 18. Hypothesis: the min_child_weight trend continues to 100.
- Result: 0.6859 -> 0.6854. Past the optimum; stays at 30. Reset to 5f2a0c8.

### Exp 20 (642436b), Eval AUC 0.6854, discard
- Class: exploration (categorical regularisation, XGBoost docs on max_cat_threshold). Hypothesis: allowing at most
  16 categories per partition split regularises the airport splits.
- Result: 0.6859 -> 0.6854. Restricting the partitions hurts, consistent with exp 9: rich airport groupings are
  real signal. Reset to 5f2a0c8.

## Synthesis after 20 experiments
- Best so far 0.6859 (5f2a0c8): ordinal calendar + Hour/DepMinutes, categorical carrier/origin/dest, depth 4,
  500 trees, lr 0.03, subsample 0.8, colsample_bytree 0.5, min_child_weight 30, reg_lambda 50.
- Helps: leaf-level regularisation (reg_lambda 50: +0.0011, min_child_weight 30: +0.0002, colsample 0.5: +0.0001),
  shallow trees. Each has an interior optimum (lambda 200, mcw 100 are worse).
- Hurts: restricting what the model can express about dates or airports (interaction constraint -0.004,
  max_cat_threshold 16 -0.0005, target encoding instead of categoricals -0.005); extra date handles
  (HolDist -0.004, DayOfYear -0.0003); more capacity (depth 9, 1000 trees).
- Theory: signal is in hour x airport x carrier x date structure, and the model needs flexible categorical
  partitions but strongly shrunk leaves. Hyperparameter gains are now ~0.0002 per step, so the next gains must
  come from new information: interaction categoricals (route, carrier x origin), or variance reduction.
- Next: research the Kaggle mlcourse.ai flight delays competition (same columns) for features that worked.

### Exp 21 (789d363), Eval AUC 0.6851, discard
- Class: follow-up to exp 20. Hypothesis: if restricting partitions hurts, unrestricted ones (283) may help.
- Result: 0.6859 -> 0.6851. The default 64 is better than both 16 and 283. Reset to 5f2a0c8.

## Research after 20 experiments
- Searched for write-ups of the Kaggle mlcourse.ai flight delays competition (same columns) and flight delay
  feature engineering projects (gabrielefirriolo flight-delay-prediction, Medium "Using machine learning to
  predict flight delays"): recurring features are route (origin-dest), carrier x origin, time slots / hour, and
  historical delay rates per route / airline / time band.
- Plan: add interaction categoricals one at a time: Route, then Carrier x Origin; unseen levels become NaN.

### Exp 22 (3157d0f), Eval AUC 0.6823, discard
- Class: exploration (interaction feature, from the flight delay write-ups). Hypothesis: a Route categorical gives
  route-specific delay levels that origin and dest alone cannot.
- Result: 0.6859 -> 0.6823, and eval got slower (48s). ~4000 thin levels: the partitions fit 2005 noise.
  Reset to 5f2a0c8.

### Exp 23 (23e3abb), Eval AUC 0.6843, discard
- Class: exploration (interaction feature). Hypothesis: Carrier x Origin identifies hubs / carrier operations at
  an airport, with far fewer levels than Route.
- Result: 0.6859 -> 0.6843. Interaction categoricals hurt across the year shift. Reset to 5f2a0c8.

### Exp 24 (ee47978), Eval AUC 0.6856, discard
- Class: ablation/simplification. Hypothesis: CRSDepTime, Hour and DepMinutes are redundant; one is enough.
- Result: 0.6859 -> 0.6856. Lower, so discarded under the strict keep rule despite being simpler. With
  colsample_bytree 0.5 the redundant copies raise the chance that each tree sees the time of day. Reset to 5f2a0c8.

### Exp 25 (971b01c), Eval AUC 0.6860, keep
- Class: exploration (variance reduction, XGBoost docs on num_parallel_tree / boosted random forests).
  Hypothesis: averaging 4 subsampled trees per round reduces the variance of each boosting step.
- Result: 0.6859 -> 0.6860. Marginal but higher, kept. Training 9.9s.

### Exp 26 (7134c8d), Eval AUC 0.6858, discard
- Class: follow-up to exp 25. Hypothesis: with 4 trees per round, stronger row subsampling (0.5) decorrelates them.
- Result: 0.6860 -> 0.6858. Reset to 971b01c.

### Exp 27 (6f6babd), Eval AUC 0.6851, discard
- Class: follow-up to exp 25 (random-forest style sampling per node, XGBoost RF docs).
- Result: 0.6860 -> 0.6851. Per-tree column sampling is better than per-node here. Reset to 971b01c.

### Exp 28 (f2d3f7b), Eval AUC 0.6863, keep
- Class: follow-up (re-tune depth after regularising). Hypothesis: depth 4 was chosen with weak regularisation;
  with reg_lambda 50 and min_child_weight 30 a deeper tree may now be affordable.
- Result: 0.6860 -> 0.6863. Kept.

### Exp 29 (618ad9b), Eval AUC 0.6864, keep
- Class: follow-up to exp 28. Hypothesis: depth 6.
- Result: 0.6863 -> 0.6864. Marginal but higher, kept. Training 13.6s.

### Exp 30 (fa8d26e), Eval AUC 0.6861, discard
- Class: follow-up to exp 29. Hypothesis: deeper trees may want more L2 (200).
- Result: 0.6864 -> 0.6861. reg_lambda 50 remains the optimum. Reset to 618ad9b.

## Synthesis after 30 experiments
- Best 0.6864 (618ad9b): depth 6, 500 rounds x 4 parallel trees, lr 0.03, subsample 0.8, colsample_bytree 0.5,
  min_child_weight 30, reg_lambda 50; features: Distance, CRSDepTime, Hour, DepMinutes, ordinal Month /
  DayofMonth / DayOfWeek, categorical carrier / origin / dest.
- New since exp 20: every added feature hurt (Route -0.0036, Carrier x Origin -0.0016), dropping redundant time
  columns hurt slightly (-0.0003). num_parallel_tree 4 (+0.0001) and re-tuned depth 5-6 under strong
  regularisation (+0.0004) helped a little. Depth and L2 interact: depth 6 was bad with lambda 1, fine with 50.
- The curve is flat: steps are now 0.0001-0.0003, comparable to what a different seed would move. The model
  family is close to its ceiling on these 8 raw columns under the 2005 -> 2006 shift.
- Next: remaining regularisers from the XGBoost "Notes on Parameter Tuning" page (gamma, max_bin), DART
  (dropout, XGBoost DART tutorial: reduces the train/test gap), then lower learning rate with more rounds.

## Research after 30 experiments
- XGBoost "Notes on Parameter Tuning" (xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html): two ways
  to control overfitting: model complexity (max_depth, min_child_weight, gamma, max_cat_threshold) and
  randomness (subsample, colsample_*), plus lower eta with more rounds.
- XGBoost DART tutorial + r-bloggers "DART: Dropout Regularization in Boosting Ensembles": dropping ~10% of trees
  per round narrows the train/test gap; training is slower (no prediction buffer).

### Exp 31 (a7db5cd), Eval AUC 0.6863, discard
- Class: exploration (regularisation, XGBoost parameter tuning notes). Hypothesis: gamma 1.0 prunes weak splits.
- Result: 0.6864 -> 0.6863. No gain. Reset to 618ad9b.

### Exp 32 (16ab378), Eval AUC 0.6859, discard
- Class: exploration (robustness to unseen levels). Hypothesis: 2006 has carriers/airports absent from 2005; they
  become NaN and follow an arbitrary default direction because train has no NaN. Masking 5% of carriers and 2%
  of origins/dests in training teaches a sensible default.
- Result: 0.6864 -> 0.6859. No gain; the lost information costs more than the default direction gains.
  Reset to 618ad9b.

### Exp 33 (29ab027), Eval AUC 0.6864, discard
- Class: exploration (encoding). Hypothesis: Month / DayOfWeek also as categoricals lets one split group
  non-adjacent months (Jun-Aug + Dec) or weekdays.
- Result: 0.6864 -> 0.6864, exactly equal, but more code and slower eval: discarded by the keep rule.
  Reset to 618ad9b. Three discards in a row within 0.001: plateau, research before the next change.

### Exp 34 (33668aa), Eval AUC 0.6862, discard
- Class: follow-up (tuning guides: lower eta, more rounds). Hypothesis: halving the learning rate at the same
  total shrinkage smooths the fit.
- Result: 0.6864 -> 0.6862, training 26s. No gain. Reset to 618ad9b.

## Research at the plateau (after exp 33)
- KDnuggets "7 XGBoost tricks", "How to rank 10% in your first Kaggle competition", Toptal "Ensemble methods":
  when features and single-model tuning stop helping, the remaining gains come from variance reduction: seed
  averaging / bagging and blending models that differ in structure (depth, growth policy, feature view).
- XGBoost docs: grow_policy=lossguide with max_leaves gives leaf-wise (LightGBM-style) trees, a structurally
  different way to spend the same capacity.
- Plan: lossguide trees; then a blend of structurally different XGBoost models behind one predict_proba.

### Exp 35 (f2eb67f), Eval AUC 0.6865, keep
- Class: exploration (tree structure, XGBoost docs on grow_policy). Hypothesis: leaf-wise growth with 32 leaves
  spends capacity where the gain is (hour x hub airports) instead of on full depth-6 levels.
- Result: 0.6864 -> 0.6865. Marginal but higher, kept.

### Exp 36 (489212b), Eval AUC 0.6868, keep
- Class: follow-up to exp 35. Hypothesis: fewer leaves (20) but deeper allowed paths (10): narrow, deep trees.
- Result: 0.6865 -> 0.6868. Kept.

### Exp 37 (8868f47), Eval AUC 0.6864, discard
- Class: follow-up to exp 36. Hypothesis: even fewer leaves (12).
- Result: 0.6868 -> 0.6864. 20 leaves is the optimum of 12 / 20 / 32. Reset to 489212b.

### Exp 38 (ea3215d), Eval AUC 0.6861, discard
- Class: exploration (blending, from the plateau research). Hypothesis: averaging the current model with a second
  one that sees the calendar as categoricals adds diversity.
- Result: 0.6868 -> 0.6861, run time 65s. The categorical-calendar member is weaker (date partitions overfit
  2005) and drags the average down. Reset to 489212b.

### Exp 39 (fae2e84), Eval AUC 0.6868, discard
- Class: exploration (regularisation of numeric splits). Hypothesis: 64 bins coarsen the time-of-day thresholds.
- Result: 0.6868 -> 0.6868, exactly equal; one more parameter and no faster (12.1s both), so discarded by the
  keep rule. Reset to 489212b.

### Exp 40 (724d63d), Eval AUC 0.6856, discard
- Class: exploration (categorical handling, XGBoost categorical tutorial). Hypothesis: one-vs-rest splits for the
  20 carriers are more constrained than partitions and may transfer better.
- Result: 0.6868 -> 0.6856. Partitions are better for carriers too. Reset to 489212b.

## Synthesis after 40 experiments
- Best 0.6868 (489212b): leaf-wise trees (lossguide, 20 leaves, max depth 10), 500 rounds x 4 parallel trees,
  lr 0.03, subsample 0.8, colsample_bytree 0.5, min_child_weight 30, reg_lambda 50.
- Since exp 30: tree shape was the only lever that moved (lossguide 32 leaves +0.0001, 20 leaves +0.0003).
  gamma, max_bin, lower lr, masking, extra categorical views, the two-view blend and one-hot carriers did not help.
- Pattern over the whole run: (1) capacity must be small but well placed (few leaves, strong L2, large
  min_child_weight); (2) default categorical partitions on raw carrier/origin/dest are the best encoding, every
  alternative (TE, one-hot, restricted or unrestricted thresholds, interaction categoricals) is worse;
  (3) ordinal calendar beats categorical calendar; (4) extra features consistently hurt.
- Next: ablations (is carrier stable across years?), then re-tune min_child_weight / rounds for leaf-wise trees.

## Research after 40 experiments
- XGBoost parameter docs (feature_weights with colsample_*; colsample_by* are cumulative) and the missing-value
  write-ups (machinelearningmastery "Navigating missing data challenges with XGBoost"): unseen categories are
  treated as missing and follow the default direction learned in training; with no missing values in train
  that direction is not informed. Exp 32 already tested the masking remedy without success; an alternative is
  to ask whether the carrier column is stable at all across years (ablation).

### Exp 41 (5f2f9ce), Eval AUC 0.6814, discard
- Class: ablation. Hypothesis: carrier effects may drift between years (and 2006 has unseen carriers), so the
  model might be better without the column.
- Result: 0.6868 -> 0.6814. Carrier is a strong, transferable feature. Reset to 489212b.

### Exp 42 (658e979), Eval AUC 0.6870, keep
- Class: exploration (feature). Hypothesis: the minute within the hour (on-the-hour vs odd-minute schedules, hub
  banks) is information the trees cannot extract from DepMinutes thresholds.
- Result: 0.6868 -> 0.6870. Kept: the first added feature since exp 1 that did not hurt.

### Exp 43 (f207279), Eval AUC 0.6873, keep
- Class: follow-up to exp 42 (numeric interaction feature). Hypothesis: one axis for the weekly cycle,
  (DayOfWeek-1)*24 + Hour, lets a single split isolate Friday/Sunday evening peaks or Saturday lulls.
- Result: 0.6870 -> 0.6873. Kept.

### Exp 44 (f635f48), Eval AUC 0.6876, keep
- Class: follow-up to exp 43. Hypothesis: the daily delay curve changes with season (summer afternoon storms,
  winter mornings); (Month-1)*24 + Hour gives it one axis.
- Result: 0.6873 -> 0.6876. Kept. Numeric composite axes of two ordinal columns help where extra categoricals hurt.

### Exp 45 (35f5ae4), Eval AUC 0.6871, discard
- Class: follow-up to exp 44. Hypothesis: a Month x DayOfWeek axis (summer Fridays, December Sundays).
- Result: 0.6876 -> 0.6871. Month x weekday cells are close to individual 2005 dates (4-5 days each); hurts.
  Reset to f635f48.

### Exp 46 (eb6d4df), Eval AUC 0.6872, discard
- Class: follow-up to exp 43. Hypothesis: the same hour x weekday grid ordered hour-first makes "evening hours on
  Thu-Fri" contiguous.
- Result: 0.6876 -> 0.6872. Redundant with HourOfWeek; hurts. Reset to f635f48.

### Exp 47 (d10687e crash, e7c3ebb), Eval AUC 0.6875, discard
- Class: exploration (feature). Hypothesis: approximate scheduled arrival time (DepMinutes + 0.12 * Distance + 30)
  is a Distance x time-of-day axis (late arrivals, red-eyes).
- d10687e crashed after 1s (KeyError: ArrMinutes was computed before DepMinutes existed); fixed in e7c3ebb.
- Result: 0.6876 -> 0.6875. No gain. Reset to f635f48.

### Exp 48 (c7799dd), Eval AUC 0.6874, discard
- Class: follow-up (re-tune after adding features). Hypothesis: 13 partly redundant features want a lower
  colsample_bytree (0.4).
- Result: 0.6876 -> 0.6874. Reset to f635f48.

### Exp 49 (b23b304), Eval AUC 0.6873, discard
- Class: follow-up (re-tune rounds for leaf-wise trees). Hypothesis: 20-leaf trees may want more rounds (700).
- Result: 0.6876 -> 0.6873. Reset to f635f48.

### Exp 50 (a8f31b7), Eval AUC 0.6873, discard
- Class: follow-up to exp 49. Hypothesis: fewer rounds (350) if 700 is worse.
- Result: 0.6876 -> 0.6873. 500 rounds is the optimum again. Reset to f635f48.

## Synthesis after 50 experiments
- Best 0.6876 (f635f48). Since exp 40 the gains came from features for the first time: Minute of the hour
  (+0.0002), HourOfWeek (+0.0003), MonthHour (+0.0003): numeric composite axes that combine Hour with another
  ordinal calendar column. Composites that approach single dates (MonthDow) or duplicate an existing axis
  (HourDow) hurt, as did approximate arrival time, and re-tuning colsample / rounds around the optimum.
- Theory now: the transferable signal is the daily delay curve and how it shifts by weekday and season; giving
  the trees a direct axis for those two interactions is worth more than any encoding of airports or carriers.
- Next (little time left): simplification (CRSDepTime is now exactly Hour*100 + Minute), then stop.

### Exp 51 (ec4b350), Eval AUC 0.6873, discard
- Class: ablation/simplification. Hypothesis: CRSDepTime is exactly Hour*100 + Minute, so it can go.
- Result: 0.6876 -> 0.6873. Lower, discarded (same pattern as exp 24: with colsample_bytree 0.5 the redundant
  time columns act as a higher sampling weight for time of day). Reset to f635f48.

## Research after 50 experiments (also a plateau: exp 48-51 within 0.001)
- Flight delay papers (ITM Conferences ICCWCS 2022 gradient boosting comparison, arXiv 1911.01605, CMU capstone
  poster): feature lists are the same raw columns plus scheduled arrival time / elapsed time (tried as
  ArrMinutes, exp 47) and cyclic encodings (irrelevant for trees). Nothing new to add on the feature side.
- Remaining principled lever from the plateau research: more variance reduction (more parallel trees per round).

### Exp 52 (dd5fe6e), Eval AUC 0.6873, discard
- Class: follow-up to exp 25 (variance reduction). Hypothesis: 8 parallel trees per round instead of 4.
- Result: 0.6876 -> 0.6873, training 24s. No gain; differences of 0.0003 here are seed-level noise.
  Reset to f635f48.

### Exp 53 (1cab6f4), Eval AUC 0.6869, discard
- Class: follow-up to exp 43/44 (composite axis). Hypothesis: carrier (ranked by train delay rate) x Hour gives
  each carrier's daily curve one axis.
- Result: 0.6876 -> 0.6869. Composites built on a target-rate ordering hurt, like the target encodings.
  Reset to f635f48.

### Exp 54 (e339794), Eval AUC 0.6876, discard
- Class: follow-up (re-tune for leaf-wise trees). Hypothesis: with only 20 leaves, min_child_weight 15 may do.
- Result: 0.6876 -> 0.6876, exactly equal, not simpler or faster: discarded by the keep rule. Reset to f635f48.

### Exp 55 (92e246e), Eval AUC 0.6875, discard
- Class: follow-up to exp 54. Hypothesis: the other direction, min_child_weight 60.
- Result: 0.6876 -> 0.6875. min_child_weight is flat between 15 and 60. Reset to f635f48.

### Exp 56 (1dffe72), Eval AUC 0.6871, discard
- Class: follow-up to exp 44. Hypothesis: a finer season x hour axis (half-months).
- Result: 0.6876 -> 0.6871. Finer than a month is too close to 2005 dates. Reset to f635f48.

### Exp 57 (6080416), Eval AUC 0.6879, keep
- Class: follow-up (re-tune L2 for 20-leaf trees with 13 features). Hypothesis: the leaf-wise trees are already
  small, so less L2 (25 instead of 50) may be enough.
- Result: 0.6876 -> 0.6879. Kept.

### Exp 58 (bf1e46a), Eval AUC 0.6877, discard
- Class: follow-up to exp 57. Hypothesis: even less L2 (12).
- Result: 0.6879 -> 0.6877. reg_lambda 25 stays. Reset to 6080416.

### Exp 59 (9d1c66f), Eval AUC 0.6877, discard
- Class: follow-up (re-tune leaves with the lighter L2). Hypothesis: 24 leaves.
- Result: 0.6879 -> 0.6877. Reset to 6080416.

### Exp 60 (366c60f), Eval AUC 0.6876, discard
- Class: follow-up (re-tune row sampling). Hypothesis: subsample 0.7.
- Result: 0.6879 -> 0.6876. Reset to 6080416.

### Exp 61 (a9846e6), Eval AUC 0.6878, discard
- Class: follow-up to exp 60. Hypothesis: the other direction, subsample 0.9.
- Result: 0.6879 -> 0.6878. subsample 0.8 stays. Reset to 6080416. Less than 2 minutes left: wrap up.

## Final summary
- Best Eval AUC 0.6879 at commit 6080416 (baseline 0.6743, +0.0136). 61 experiments after the baseline
  (63 result rows incl. the baseline and one crash row): 16 kept, 45 discarded, 1 crash that was fixed and rerun.
- Final model: XGBoost, leaf-wise trees (grow_policy lossguide, max_leaves 20, max_depth 10), 500 rounds x 4
  parallel trees, lr 0.03, subsample 0.8, colsample_bytree 0.5, min_child_weight 30, reg_lambda 25.
  Features: Distance, CRSDepTime, Hour, Minute, DepMinutes, ordinal Month / DayofMonth / DayOfWeek,
  HourOfWeek, MonthHour, categorical UniqueCarrier / Origin / Dest.
- What worked: less and better-placed capacity (lr 0.03 + subsampling +0.006, depth 6 -> 4 +0.003, strong L2
  +0.001, leaf-wise 20-leaf trees +0.0004); ordinal calendar columns (+0.001); numeric composite axes with Hour
  (Minute, HourOfWeek, MonthHour, +0.0008 together).
- What did not: every alternative encoding of carrier/airports (target encoding, one-hot, max_cat_threshold up or
  down), interaction categoricals (Route, Carrier x Origin), date handles (holiday distance, DayOfYear,
  MonthDow, HalfMonthHour), forbidding Month x DayofMonth, blending with a categorical-calendar model, masking
  for unseen levels, more rounds / lower lr, more parallel trees. DART was researched but not run (training time).
- Caveat: the last ~0.001 was gained in steps of 0.0001-0.0003 selected on eval.csv; steps of that size are
  within seed noise, so expect part of it not to carry over to the holdout set.
- Next: feature_weights for column sampling instead of redundant time columns; airport geography / time zone
  lookups (would need external data); a seed-averaged blend of the final configuration.
