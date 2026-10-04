# Research log - run `oct3`

Branch `oct3`, branched from 2e14489. Results table: `output/results.tsv` (same commit hashes).
Task: predict dep_delayed_15min (AUC) - train 200K rows from 2005, eval 50K rows from 2006 (temporal shift).

## Baseline - 2e14489 - Eval AUC 0.6743 (keep)
Starter train.py unchanged: 100 trees, depth 6, lr 0.1, native categoricals for Month/DayofMonth/DayOfWeek/Carrier/Origin/Dest,
numeric CRSDepTime and Distance. Training 0.6s, eval 30s.

## Research before exp 1
Sources: XGBoost docs "Categorical Data" tutorial (xgboost.readthedocs.io/en/stable/tutorials/categorical.html: partition-based splits,
max_cat_to_onehot, max_cat_threshold to limit overfitting on high-cardinality cats); SageMaker "Tune an XGBoost model" + generic tuning
guides (lower learning_rate with more trees, subsample/colsample_bytree 0.6-1, min_child_weight, reg_alpha/lambda); flight-delay write-ups
(time slots, cyclic/ordinal calendar encodings, route features, hub indicators, historical delay rates per route/carrier/time band).
Own notes: train is 2005, eval 2006 -> anything that memorises specific 2005 dates (storms) will not transfer; regularisation and
features that are stable year over year (hour of day, carrier, airport, route) should matter most. No count features (forbidden).

## Exp 1 - dfbb64f - 0.6754 (keep) [exploration]
Hypothesis: calendar columns as ordinal ints (instead of categorical partitions) + Hour = CRSDepTime//100 generalise better.
Result: +0.0011 vs baseline and simpler/faster prepare. Keep.

## Exp 2 - d408615 - 0.6772 (keep) [exploration]
Hypothesis: slower learning with more trees and row/column subsampling (standard tuning-guide advice) beats 100 trees at lr 0.1.
Result: +0.0018. Keep. Training still only 4s, lots of headroom under the 60s limit.

## Exp 3 - 9af3313 - 0.6805 (keep) [exploration]
Hypothesis: partition splits on 283-level Origin/Dest overfit 2005; a single smoothed delay-rate number per category (m=20 towards the
global rate, fitted on train, looked up in prepare) is a more regularised encoding.
Result: +0.0033, and eval is faster (18s). Keep.

## Exp 4 - 4b5fcdc - 0.6705 (discard) [follow-up]
Hypothesis: delay rates of route / carrier-origin / carrier-dest add information beyond the three marginal encodings.
Result: -0.0100. With ~4300 routes (~47 rows each) the in-sample encoding leaks each row's own label, the model trusts it too much.
Next: same features but encoded out-of-fold for the training rows (cross-fitting, as in sklearn TargetEncoder / CatBoost ordered stats).

## Exp 5 - 555a9a9 - 0.6794 (discard) [follow-up]
Hypothesis: with out-of-fold encodings for the training rows the pair encodings stop leaking and help.
Result: leakage fixed (0.6705 -> 0.6794) but still -0.0011 vs exp 3 and more code. Discard. Unclear whether OOF itself or the pairs cost
the 0.001 -> next run isolates it: OOF with the three marginal encodings only.

## Exp 6 - 97b40cf - 0.6777 (discard) [ablation]
Hypothesis: isolates the OOF effect on the three marginal encodings.
Result: 0.6777 < 0.6805 (in-sample). So for the low-cardinality marginals the in-sample encoding is better: it is an exact ordinal code of
the category (sorted by delay rate), the trees can still single out an airport/carrier and interact it with hour; OOF blurs that identity.
And the pairs were worth +0.0017 on top of OOF marginals. Next: in-sample marginals + OOF pairs.

## Exp 7 - 8dab9dc - 0.6817 (keep) [follow-up]
Hypothesis: best of both - exact (in-sample) ordinal codes for the 3 marginals, out-of-fold delay rates for the 3 high-cardinality pairs.
Result: +0.0012 vs exp 3. Keep (modest gain for ~15 lines; watch it). Training took 47.8s because of a row-wise string join in fit_te -
must be fixed (string concat) in the next commit, it is a pure speed refactor.

## Exp 8 - 021b693 - 0.6782 (discard) [ablation/simplification]
Hypothesis: DayofMonth mostly lets the trees memorise specific 2005 dates (storms) that do not repeat in 2006; dropping it should not hurt.
Result: -0.0035, wrong. Day of month carries transferable signal (fixed-date holidays / holiday periods, end-of-month patterns).
Discard, but the speed refactor (training 47.8s -> 4s) is carried into the next commit.
Next: make the calendar signal explicit - days to the nearest US holiday, which needs the year (inferable from date + weekday).

## Exp 9 - cdd9e1a - 0.6791 (discard) [exploration]
Research: searched flight-delay feature-engineering papers for holiday features (arXiv 2601.00875 "Prediction of airport on-time
performance", Promet "impact of short-term features"): they stress historical delay rates of airline/route and calendar flags; nothing
specific on holiday distance. Hypothesis: signed days to the nearest major US holiday (year inferred from date+weekday, 2005 vs 2006)
makes the transferable part of the date signal explicit (moving holidays: Thanksgiving, Easter, Memorial/Labor day).
Result: -0.0026 vs 0.6817. Discard. Surprising; the feature also lets trees address exact 2005 dates in a second way -> more date overfit?

## Exp 10 - be18fc7 - 0.6817 (keep) [simplification]
The speed refactor alone (string concat instead of row-wise join): identical AUC 0.6817, training 4s instead of 48s. Keep.

## Synthesis after 10 experiments
- Helps: ordinal calendar ints (+0.001), slower learning/more trees with subsampling (+0.002), carrier/origin/dest as in-sample
  smoothed delay-rate codes instead of native categoricals (+0.003), out-of-fold delay rates of route/carrier-origin/carrier-dest (+0.001).
- Hurts: in-sample encodings of high-cardinality pairs (label leak, -0.010), OOF for the low-cardinality marginals (-0.003),
  dropping DayofMonth (-0.0035), holiday distance (-0.0026).
- Theory: the 2005 -> 2006 shift caps the AUC around 0.68; what transfers are stable structures (hour of day, airport/carrier/route
  propensities, fixed dates). Changes of +-0.001 are close to the noise of a 50K eval set (SE ~0.002), so prefer larger, principled moves.
- Next: the model-complexity axis has not been explored at all (depth, trees, regularisation) - with a temporal shift a simpler model
  may transfer better. Then interactions with hour (origin x hour, carrier x hour delay rates), minute-of-day features.

## Exp 11 - ca48e1c - 0.6820 (keep) [exploration]
Research: looked for temporal-shift advice for GBMs (HyperTime arXiv 2305.18421; "Training for the Future" NeurIPS 2021; TDS "biggest
weakness of boosting trees"): robust/regularised configs transfer better across time. Hypothesis: depth 4 transfers at least as well as 6.
Result: 0.6820 vs 0.6817: equal, simpler model -> keep.

## Exp 12 - 32082f0 - 0.6802 (discard) [follow-up]
Hypothesis: depth 4 with 600 trees may underfit; 1500 trees. Result: -0.0018. More capacity fits 2005 specifics. Try the other direction.

## Exp 13 - db5bdd9 - 0.6810 (discard) [follow-up]
300 trees: -0.0010. So 300/600/1500 trees = 0.6810/0.6820/0.6802: flat optimum around 600 at lr 0.05, depth 4. Hyperparameter axis is
flat (+-0.001); real gains have to come from features. First calibrate the seed noise so that I know what a real difference is.

## Exp 14 - 6d59e7f - 0.6825 (discard) [noise calibration]
Same config as ca48e1c, model random_state 7 instead of 42: 0.6825 vs 0.6820. Seed noise ~0.0005, so the -0.003 effects seen for
dropping DayofMonth / adding HolidayDist are real, and +-0.001 moves are not. Not kept (picking a seed on eval would be overfitting eval).
Next: is the DayofMonth signal a holiday signal? Drop DayofMonth, add HolidayDist clipped to +-8 days (cannot address arbitrary dates).

## Exp 15 - 8df8f7e - 0.6832 (keep) [follow-up to exp 8/9]
Hypothesis: the transferable part of DayofMonth is the holiday calendar; a clipped (+-8) holiday distance keeps that and removes the
ability to memorise arbitrary 2005 dates.
Result: +0.0012 vs 0.6820 (about 2x seed noise). Keep, although the calendar code is ~15 lines: it is the principled representation
(DayofMonth alone was worth +0.0035, this recovers it and more without the raw date).

## Exp 16 - 41c1a08 - 0.6838 (discard) [exploration]
Hypothesis: delay profiles by hour differ per airport/carrier (congestion banks); OOF delay rates of origin-hour, dest-hour, carrier-hour.
Result: +0.0006, within seed noise, 6 more lines and eval 39s instead of 28s. Discard: trees already interact Hour with the ordinal codes.
Next: ablation of Month (2005 monthly delay levels may not repeat in 2006).

## Exp 17 - 823c523 - 0.6831 (keep) [ablation/simplification]
Hypothesis: 2005 monthly delay levels do not repeat in 2006. Result: 0.6831 vs 0.6832 - Month carries no net transferable signal.
Keep (simplification): calendar features are now DayOfWeek + clipped HolidayDist only.
Insight: season-of-2005 effects are year-specific. Follow-up idea: if drift matters this much, late-2005 rows should be more
representative of 2006 than early-2005 rows -> recency sample weights.

## Exp 18 - a4153d6 - 0.6803 (discard) [exploration]
Hypothesis: late-2005 rows are closer to 2006 -> sample_weight = month. Result: -0.0028. The loss of effective sample size (and the
over-weighting of autumn/winter patterns) outweighs any recency benefit. Discard.
Next: ablation - are the OOF pair encodings still needed at depth 4 without Month? If not, ~20 lines of OOF machinery can go.

## Exp 19 - d361d1d - 0.6825 (keep) [ablation/simplification]
Without the three OOF pair encodings: 0.6825 vs 0.6831, a difference at seed-noise level. Keep as a simplification; the next commit
deletes the now unused OOF/pair code (~20 lines) and must reproduce 0.6825.

## Exp 20 - df864be - 0.6830 (keep) [simplification]
Same feature set as d361d1d with the OOF/pair code deleted and prepare() written compactly (column order changed, hence 0.6830 instead
of 0.6825 through colsample randomness). Eval now takes 17s. Keep.

## Synthesis after 20 experiments
- Current best-kept: df864be 0.6830 (the high-water mark was 0.6838 with 6 OOF pair encodings, within noise of this much simpler model).
- Features in: CRSDepTime, Distance, Hour, DayOfWeek, clipped HolidayDist, delay-rate codes of carrier/origin/dest. 600 trees, depth 4.
- What transfers 2005 -> 2006: time of day, weekday, holiday proximity, carrier/airport identity. What does not: Month (no net signal),
  raw dates, anything that needs many trees or deep trees, recency weighting.
- Seed noise is ~0.0005; pair/hour-interaction encodings add at most ~0.0005-0.001 each for a lot of code.
- Hyperparameters are flat around the current point (300/600/1500 trees, depth 4/6).
- Next directions: (a) regularisation that targets drift: monotone constraints on the delay-rate codes, larger min_child_weight/lambda;
  (b) richer time-of-day representation (minutes since midnight, minute within hour); (c) research known solutions for this very
  dataset format (mlcourse.ai "flight delays" Kaggle in-class competition).

## Exp 21 - f85c901 - 0.6750 (discard) [exploration]
Research: mlcourse.ai flight-delays / benchm-ml searches gave nothing concrete; XGBoost docs "Monotonic Constraints" tutorial.
Hypothesis: forcing predictions to be monotone in the 2005 delay rate of carrier/origin/dest regularises against drift.
Result: -0.0080. Clearly wrong: the codes are used as category identities (airport x hour, airport x weekday interactions), not just
as a monotone risk score. Discard.

## Exp 22 - cda6926 - 0.6833 (discard) [exploration]
Hypothesis: larger leaves (min_child_weight 50 ~ 200 rows) transfer better. Result: +0.0003 = noise. Discard, the model axis stays flat.
Next: simplification - Hour is redundant with CRSDepTime for a tree model; drop it.

## Exp 23 - 63e23d0 - 0.6829 (keep) [simplification]
Dropping Hour: 0.6829 vs 0.6830, equal -> keep the simpler feature set (7 features).
Next: variance reduction instead of more capacity - boosted random forest (num_parallel_tree=4, XGBoost docs "Random Forests in XGBoost").

## Exp 24 - 138866d - 0.6832 (discard) [exploration]
Boosted random forest with 4 parallel trees: +0.0003 = noise. Model variance is not the bottleneck. Discard.
Next: how local is the holiday signal? clip HolidayDist at +-4 instead of +-8.

## Exp 25 - 09c9b46 - 0.6837 (keep) [follow-up]
Clip +-4: +0.0008, same complexity -> keep. The holiday signal is local; days further away only add 2005-date memorisation.
Next: one more step in the same direction (+-2) to see where it turns.

## Exp 26 - 8b09a46 - 0.6834 (discard) [follow-up]
Clip +-2: 0.6834 vs 0.6837 at +-4. Flat/slightly worse; stay at +-4 and stop tuning this knob.
Next: categories unseen in 2005 (new carriers/airports in 2006) currently become NaN, and since train has no NaN the trees' default
direction for them is arbitrary. Encode unseen categories as the global delay rate (an "average" carrier/airport) instead.

## Exp 27 - dfba61b - 0.6836 (discard) [exploration]
Unseen categories -> global rate instead of NaN: 0.6836 vs 0.6837, no effect (and slower eval). Discard.
Next: re-test capacity. More trees / deeper trees hurt earlier, but then the model still had Month and DayofMonth to memorise 2005
dates with. With only transferable features left, depth 6 may now pay off (interactions airport x time x weekday).

## Exp 28 - f47f7da - 0.6847 (keep) [follow-up]
Depth 6 without Month/DayofMonth: +0.0010 (at exp 11, with the date columns, depth 6 = depth 4). Keep. Capacity is useful again once
the features that invite year-specific memorisation are gone. Next: depth 8.

## Exp 29 - 1209fab - 0.6816 (discard) [follow-up]
Depth 8: -0.0031. Depth 6 is the sweet spot at 600 trees / lr 0.05. Discard.
Next: time-of-day detail - minute within the hour (flights scheduled on the hour/half hour vs odd minutes), which trees cannot get
from CRSDepTime without many splits.

## Exp 30 - 10cec75 - 0.6849 (discard) [exploration]
Minute within the hour: +0.0002 = noise. Discard.

## Synthesis after 30 experiments
- Best kept: f47f7da 0.6847 - 7 features (CRSDepTime, Distance, DayOfWeek, HolidayDist clipped +-4, delay-rate codes of
  carrier/origin/dest), 600 trees, depth 6, lr 0.05, subsample/colsample 0.8.
- Pattern of the last 10: every removal of year-specific or redundant inputs was free or positive (Month, Hour, pair encodings,
  wider holiday window), and once they were gone depth 6 beat depth 4 (+0.001). Additions (parallel trees, minute, unseen-category
  fill, min_child_weight, monotone constraints) gave nothing or hurt.
- Theory: the eval gap is dominated by 2005-specific day-level shocks (weather days). The model cannot use them in 2006, but they
  are also noise in the training labels that blurs the estimates of the stable effects.
- Next: (a) absorb the 2005 day-level shock during training with a "daily delay rate of 2005" feature that is set to the neutral
  global rate for non-2005 rows (nuisance-variable idea: fit with the fixed effect, predict at the reference level);
  (b) colsample/lr variants at depth 6; (c) airport-level descriptors fitted on train (mean distance, mean departure time).

## Exp 31 - e39b811 - 0.6793 (discard) [exploration]
Research: searched for "fit with a time fixed effect, predict at the reference level" in the drift literature (GI loss NeurIPS 2021,
transfer-learning stacking for temporal drift in clinical models); no direct recipe for GBMs, tried the plain version.
Hypothesis: a "2005 daily delay rate" feature absorbs day-level shocks in training; other years get the neutral global rate.
Result: -0.0054. Predicting at an artificial "average day" is not the same as marginalising over days (the model interacts the day
rate with hour/airport, and a value of exactly the prior is an unusual region). Discard; too clever.
Next: the holiday list has 11 holidays, so +-4 days flags ~30% of the year - partly still a date memoriser. Restrict it to the six
big travel holidays (New Year, Memorial Day, July 4, Labor Day, Thanksgiving, Christmas).

## Exp 32 - 620899b - 0.6838 (discard) [follow-up]
Six big holidays only: -0.0009. The minor ones (MLK, Presidents, Easter, Columbus, Veterans) carry some signal too. Discard.
Next: let each holiday have its own effect - add the index of the nearest holiday (when within the +-4 window, else -1). Without Month
the model currently has to pool e.g. the day before Thanksgiving with the day before Columbus Day.

## Exp 33 - 274bb5a - 0.6863 (keep) [follow-up]
HolidayId (index of the nearest holiday when within +-4 days, else -1): +0.0016, 3x seed noise. Keep. Each holiday has its own delay
pattern (Thanksgiving/Christmas vs Columbus Day) and that pattern repeats in 2006.
Wart: the `% 12` maps both "previous Christmas" and "next New Year" to id 0, so Dec 29-31 get id 0 instead of New Year's id 1 (the
code comment is wrong). Consistent across years, so harmless, but the next commit cleans it (next New Year -> same id as New Year).

## Exp 34 - 9dd2084 - 0.6861 (keep) [simplification]
Clean id mapping: 0.6861 vs 0.6863, equal -> keep the cleaner code.
Next: with per-holiday ids the window may be wider again (+-7): a holiday-specific pattern a week out (pre-Christmas week,
Thanksgiving week) is not a pooled date memoriser any more.

## Exp 35 - 5cbb8dc - 0.6848 (discard) [follow-up]
Window +-7: -0.0013. Even per holiday, days further out are mostly 2005 noise. Stay at +-4.
Next: a second "view" of the airports. The delay-rate code orders airports by risk only; mean flight distance per origin/dest (fitted on
train) orders them by type (regional feeder vs long-haul hub), giving the trees another way to group similar airports.

## Exp 36 - 7cda060 - 0.6872 (keep) [exploration]
Mean flight distance per origin/dest: +0.0011 for 4 lines. Keep. A second ordering of the airports helps the trees group them.
Next: same descriptor for the carrier (regional vs mainline carriers) - zero extra lines.

## Exp 37 - eb3b4b4 - 0.6870 (discard) [follow-up]
Mean flight distance per carrier too: 0.6870 vs 0.6872, no gain. Discard (carriers are few; the code already separates them).

## Exp 38 - 864d583 - 0.6875 (discard) [exploration]
Research: lat/lon of airports are a standard input in flight-delay models (arXiv 2601.00875; SciTePress 2015 paper 55877); no file may
be read, so recovered a 2-D map from train itself: shortest paths over the route-distance graph + classical MDS (Isomap, Tenenbaum
2000; MathWorks "Classical Multidimensional Scaling"). A refined least-squares version reproduced route distances to 3 miles RMS.
Result: +0.0003 = noise for 12 lines and 4 more lookups (eval 28s). Discard: geography adds nothing beyond the airport codes.
Next: scheduled arrival time proxy = departure minutes + flight time from Distance (late-evening arrivals, turn-around pressure);
a sum of two features that trees approximate poorly.

## Exp 39 - 6cef6b8 - 0.6874 (discard) [exploration]
Approximate scheduled arrival minutes: +0.0002 = noise. Discard.
Plateau (exp 37-39 all within 0.0003 of 0.6872) -> research before the next change.

## Exp 40 - 736a6ef - 0.6871 (discard) [exploration]
Research (plateau): XGBoost parameter docs (grow_policy/max_leaves, colsample_bylevel/bynode, gamma, DART rate_drop/skip_drop) and the
DART tutorial (dropout of trees against over-specialisation; slower training).
Leaf-wise growth with 40 leaves: 0.6871 vs 0.6872, equal with more parameters. Discard.

## Synthesis after 40 experiments
- Best kept: 7cda060 0.6872. Features: CRSDepTime, Distance, DayOfWeek, HolidayDist (+-4), HolidayId, delay-rate codes of
  carrier/origin/dest, mean flight distance of origin/dest. 600 trees, depth 6, lr 0.05.
- Gains of the last 10: per-holiday id (+0.0016), airport mean distance (+0.0011). Everything else (geography by MDS, arrival time,
  minute, carrier mean distance, leaf-wise trees, day-rate absorption) was noise or negative.
- Theory unchanged: only calendar structure tied to holidays and stable airport/carrier properties transfer to 2006; model shape is
  flat. Label noise from 2005 weather days is the remaining lever on the training side.
- Next: (a) down-weight training rows from extreme 2005 days (their labels say little about structural effects) - a training-only
  change, inference untouched; (b) DART as a different regulariser; (c) other target-free airport descriptors.

## Exp 41 - 366031b crash, 214a9dd - 0.6840 (discard) [exploration]
Hypothesis: give the 2005 day level to the booster as a training-only offset (base_margin = logit of the day's delay rate, 0 inside
holiday windows); the trees then model deviations from the day level and inference (no offset) uses the structural part only.
First commit crashed (NameError: used calendar() from the discarded exp 31), fixed in 214a9dd.
Result: -0.0032. The offset also absorbs transferable day-level structure - above all the weekday effect - which the model then
never learns. Follow-up: take the weekday level out of the offset (offset = logit(day rate) - logit(weekday rate)).

## Exp 42 - 981f794 - 0.6878 (discard) [follow-up]
Offset = logit(day rate) - logit(weekday rate) on non-holiday days: 0.6878 vs 0.6872. The weekday fix recovers the loss of exp 41 but
the net gain (+0.0006) is seed-noise sized for 8 lines of training-only machinery. Discard. Day shocks are noise in the labels but
with ~550 rows/day x 365 days they average out well enough already.
Next: DART (dropout of trees) as a different regulariser - 300 trees at lr 0.1 to stay inside the 60s training limit.

## Exp 43 - 57bdccb - 0.6836 (discard) [exploration]
DART (XGBoost DART tutorial; Rashmi & Gilad-Bachrach 2015): -0.0036 and training 20s. Discard; the model is not over-specialised in the
sense DART fixes.
Next: Month had no net signal (exp 17), but a coarse meteorological season (DJF/MAM/JJA/SON) might keep the climatology (summer
thunderstorms, winter) without letting the trees localise 2005 weather episodes.

## Exp 44 - 14c9f0b - 0.6833 (discard) [exploration]
Coarse season: -0.0039. Even at quarter resolution the 2005 time-of-year pattern misleads in 2006. Time-of-year information is
harmful unless tied to a holiday. Discard.
Next: position of the flight within its route's daily schedule - CRSDepTime minus the route's median departure time (fitted on
train; the lookup example in program.md), plus the same relative to the origin's median.

## Exp 45 - 1851606 - 0.6868 (discard) [exploration]
Departure time relative to the route's median: 0.6868 vs 0.6872. No gain. Discard.
Next: carrier as a native categorical (20 levels, partition splits can form arbitrary carrier groups in one split) in addition to its
delay-rate code. The switch away from native categoricals in exp 3 was probably won on the 283-level airports, not on the carrier.

## Exp 46 - 5c2a4fa - 0.6836 (discard) [exploration]
Carrier as native categorical on top of its code: -0.0036. More flexible carrier grouping fits 2005 carrier specifics that changed in
2006 (mergers, hub changes). Discard. Follow-up in the opposite direction: is the carrier code itself still worth having? Ablate it.

## Exp 47 - e2cd01e - 0.6817 (discard) [ablation]
Without the carrier code: -0.0055. The carrier matters, but only in its restricted ordinal form (exp 46). Discard.
Insight: for categories, splits restricted to "contiguous in 2005 delay rate" transfer, arbitrary groupings do not.
Next: push that restriction further for all features with coarser histograms (max_bin 32): airports with similar rates share a bin,
departure time gets ~30-minute bins.

## Exp 48 - c3cca38 - 0.6871 (discard) [exploration]
max_bin 32: 0.6871 vs 0.6872, equal -> discard (no reason to carry the parameter).
Plateau again. Rather than more additions, test whether the sampling/regularisation parameters set in exp 2 are still needed on the
current feature set: remove subsample, colsample_bytree and min_child_weight (XGBoost defaults) - a simplification if equal.

## Exp 49 - 181b94c - 0.6854 (discard) [ablation/simplification]
Defaults for subsample/colsample_bytree/min_child_weight: -0.0018, so the randomisation does help. Discard.
Follow-up: go the other way - stronger randomisation (subsample 0.5, colsample_bytree 0.5).

## Exp 50 - 3e9183f - 0.6874 (discard) [follow-up]
subsample/colsample 0.5: 0.6874 vs 0.6872 = noise. 0.8/0.8 stays.

## Synthesis after 50 experiments
- Best kept: 7cda060 0.6872 (unchanged since exp 36). 14 experiments in a row without a keeper.
- What the last 10 say: (1) anything giving the model more freedom over carriers or time of year hurts by 0.003-0.004 (native
  categorical carrier, season, DART was simply worse); (2) removing the carrier code or the row/column sampling hurts as well;
  (3) target-free extras (route-median departure, arrival time, coarser bins, stronger sampling) are neutral; (4) absorbing 2005
  day shocks via an offset is neutral at best.
- Theory: the kept model sits at a broad optimum for these inputs; what is left is to restrict *which interactions* the trees may
  form, in the same spirit as the ordinal codes: interactions that are 2005-specific (holiday x carrier/airport) vs stable ones
  (time of day x airport).
- Next: XGBoost interaction_constraints (docs: "Feature Interaction Constraints").

## Exp 51 - b59ce66 - 0.6888 (keep) [exploration]
Research: XGBoost tutorial "Feature Interaction Constraints" (nested lists of feature names; overlapping groups; claimed benefit:
less noise, better generalisation by only allowing interactions that make sense).
Hypothesis: holiday x carrier/airport interactions are 2005 episodes (a storm at one hub on one Thanksgiving); allow holiday features
to interact only with weekday and time of day.
Result: +0.0016 (3x seed noise) for one parameter. Keep. First keeper since exp 36.
Next: stricter - holiday features interact only with each other.

## Exp 52 - 54ffd7f - 0.6887 (keep) [simplification]
Holiday features interact only with each other (purely additive holiday effect): 0.6887 vs 0.6888, equal with a more constrained
model -> keep.
Next: same idea for the carrier - carrier x airport interactions encode the 2005 hub structure, which changed in 2006. Let the carrier
interact with time of day / weekday / distance only.

## Exp 53 - 829bcc8 - 0.6883 (discard) [follow-up]
Carrier barred from interacting with airports: 0.6883 vs 0.6887, no gain -> discard. Carrier x airport is not the problem.
Next: time of year again, but purely additive: Month in its own constraint group. Season hurt in exp 44 when it could interact
with airports/carriers (2005 episodes); an additive monthly level is closer to climatology.

## Exp 54 - b4ba319 - 0.6860 (discard) [exploration]
Additive Month: -0.0027. The 2005 monthly levels themselves do not repeat, not just their interactions. Time of year is closed.
Next: origin x dest interactions (route specifics) - forbid them: origin-side and dest-side features in separate groups, both free
to interact with time, distance and carrier.

## Exp 55 - bf2a2fa - 0.6884 (discard) [follow-up]
No origin x dest interactions: 0.6884 vs 0.6887, equal, more config -> discard.
Next: the weekday. Put DayOfWeek into the calendar group only (it may interact with the holiday features but not with time of day,
carrier or airports): are weekday x airport/time interactions stable or 2005 noise?

## Exp 56 - 6e5d9ef - 0.6865 (discard) [follow-up]
Weekday barred from interacting with time of day / airports / carrier: -0.0022. Those interactions are stable and needed. Discard.
Next: revisit the holiday window now that the holiday effect is additive-only: +-7 (it hurt in exp 35 when holiday features could
interact with airports and carriers).

## Exp 57 - b68e1b4 - 0.6869 (discard) [follow-up]
Window +-7 again, now additive-only: -0.0018. +-4 stays; the holiday effect really is local.
Next: with the holiday features fenced off, re-check the learning rate / number of trees trade-off once: lr 0.025 with 1200 trees
(same total shrinkage budget, smoother fit).

## Exp 58 - b3b176c - 0.6891 (discard) [follow-up]
lr 0.025 x 1200 trees: +0.0004 = noise. Discard.
Thought: the holiday block has 11 holidays x 9 day offsets = 99 date cells, each estimated from ONE 2005 day, and the day-to-day
std of the delay rate is 0.12 - most of what the block learns per cell is 2005 weather. Pool it: let HolidayId and HolidayDist be
additive to each other too (f(holiday) + g(offset), 20 numbers instead of 99).

## Exp 59 - e47a3de - 0.6879 (discard) [follow-up]
f(holiday) + g(offset) instead of the full holiday x offset table: -0.0008. The specific days (Wednesday before Thanksgiving, Dec 26...)
matter; pooling loses them. Discard.
Next: retest the day-shock training offset (exp 42: +0.0006 before the constraints). With the holiday block now additive and the
flight block free of calendar features, removing the 2005 day level from the flight block's targets is a cleaner fit.

## Exp 60 - 1331269 - 0.6889 (discard) [follow-up]
Day-shock offset under the constraints: +0.0002 = noise for 8 lines. Discard for good.

## Synthesis after 60 experiments
- Best kept: 54ffd7f 0.6887. Same 10 features as at exp 36, plus interaction constraints that make the holiday block additive.
- Last 10: fencing off the holiday features was the one real gain (+0.0016). Other fences (carrier vs airports, origin vs dest)
  are neutral, fencing the weekday hurts (-0.002), any time-of-year feature hurts even additively (-0.003), pooling or widening the
  holiday block hurts (-0.001..-0.002), lr/trees and the day-shock offset are noise.
- Theory: stable = flight block (time of day x weekday x carrier x airports x distance) + an additive, local holiday table.
  2005-specific = time of year, holiday x flight interactions.
- Unused stable information: the *recurring scheduled flight*. A (carrier, origin, dest, hour) slot flies daily and many slots persist
  into 2006; chronic lateness of a slot (late in an aircraft rotation, tight turn-around) is a property the current features can only
  approximate. Next: out-of-fold delay rate of that slot (needs the OOF machinery again, so it must earn >= +0.002).

## Bookkeeping error found (affects exp 30-60) and Exp 61 - 98cd31f - 0.6878 (discard) [ablation]
While reading train.py I found `X["Minute"] = CRSDepTime % 100` still in the kept code. After exp 30 (10cec75, 0.6849 vs 0.6847) I
logged "discard" but forgot the `git reset`, so every commit from exp 31 on was built on top of 10cec75 and all AUCs of exp 31-60
include the Minute feature. The comparisons among them remain valid (same base), only the description of the base was wrong.
Since exp 51 Minute is not listed in any interaction-constraint group, i.e. it can only be used additively.
Exp 61 removes it to check: 0.6878 vs 0.6887 (-0.0009). So Minute is worth ~0.001 as an additive one-liner -> ablation discarded,
Minute stays, and the results.tsv row of 10cec75 is corrected to `keep` with a note. Best kept remains 54ffd7f (0.6887).
Next commit lists Minute explicitly as its own (additive) constraint group so the code says what the model does.

## Exp 62 - 60b940b - 0.6887 (keep) [cleanup]
Minute listed explicitly as its own constraint group: identical 0.6887 (confirms that unlisted features are additive-only). Keep.
Next: the recurring-flight idea from the synthesis: out-of-fold delay rate of the (carrier, origin, dest, hour) slot.

## Exp 63 - 3ac2143 - 0.6842 (discard) [exploration]
OOF delay rate of the (carrier, origin, dest, hour) slot: -0.0045. Within 2005 (which is what OOF measures) the slot rate is a strong
feature, so the trees lean on it; across years slots are re-timed and re-assigned, so it is much weaker in 2006 than the model
believes. Discard. General lesson: a feature's weight is calibrated on within-year predictiveness; fine-grained identities
(slot, route, exact date) are over-trusted.
Next (same lesson, other direction): the flight block can still pin down individual flights through the exact HHMM of CRSDepTime
combined with the airport codes. Replace CRSDepTime by the hour in the flight block (Minute stays as the additive feature).

## Exp 64 - 5449ccd - 0.6884 (discard) [follow-up]
CRSDepTime at hour resolution: 0.6884 vs 0.6887, equal, not simpler -> discard.
Next: L2 on leaf weights (reg_lambda 30). It shrinks small leaves (fine-grained cells = the over-trusted part) much more than large
ones, which is the kind of regularisation the slot/route lesson asks for; min_child_weight (exp 22) only forbids small leaves.

## Exp 65 - fe32d3f - 0.6896 (keep) [exploration]
reg_lambda 30: +0.0009 (2x seed noise) for one parameter. Keep. Shrinking small leaves helps. Next: push further (reg_lambda 100).

## Exp 66 - 1d1869b - 0.6901 (keep) [follow-up]
reg_lambda 100: +0.0005 more, same complexity, consistent trend. Keep. Next: 300.

## Exp 67 - e706812 - 0.6895 (discard) [follow-up]
reg_lambda 300: 0.6895, past the optimum. Stay at 100.
Next: with small leaves heavily shrunk, deeper trees may be safe now (depth 8 lost 0.003 at lambda 1).

## Exp 68 - 8fa5bbc - 0.6894 (discard) [follow-up]
Depth 8 with lambda 100: 0.6894 vs 0.6901. Much less damage than at lambda 1 (-0.003) but no gain. Depth 6 stays.
Next: lambda 100 also damps every step, so the 600 trees may now stop short: 1000 trees.

## Exp 69 - 2eb7b1e - 0.6899 (discard) [follow-up]
1000 trees at lambda 100: 0.6899 vs 0.6901, equal. 600 stays.
Next: simplification - with lambda 100 doing the small-leaf control, is min_child_weight=5 still needed? Remove it (default 1).

## Exp 70 - 7217b71 - 0.6899 (keep) [simplification]
Default min_child_weight: 0.6899 vs 0.6901, equal -> keep the simpler parameter set.

## Synthesis after 70 experiments
- Best kept: 7217b71 0.6899 (1d1869b scored 0.6901 with min_child_weight=5; same within noise).
  Features: CRSDepTime, Minute (additive), Distance, DayOfWeek, HolidayDist+HolidayId (additive block), delay-rate codes of
  carrier/origin/dest, mean distance of origin/dest. 600 trees, depth 6, lr 0.05, subsample/colsample 0.8, reg_lambda 100.
- Last 10: strong L2 on leaf weights was the gain (+0.0014 from lambda 1 -> 100); the recurring-flight slot rate hurt badly
  (-0.0045); hour-resolution time, depth 8, more trees, day-shock offset: neutral.
- Consolidated theory: the model is calibrated on within-2005 predictiveness, and everything fine-grained (slots, routes, exact
  dates, small leaves) is over-trusted relative to how it transfers. What works is taking trust away from fine cells:
  ordinal codes instead of free categories, additive fences around calendar features, heavy leaf shrinkage.
- Next: other ways to take trust from fine cells: Distance as additive-only (Distance x origin code pins down routes),
  reg_alpha, max_delta_step; then re-check row/column sampling under the new regularisation.

## Exp 71 - 6f711a5 - 0.6896 (discard) [exploration]
Research: XGBoost parameter docs / regularised-objective notes (apxml ch.3; getML XGBoost predictor docs) on reg_alpha (L1, zeroes weak
leaves), max_delta_step (caps leaf steps), gamma; noisy-label literature says regularisation mitigates label noise.
Distance additive only: 0.6896 vs 0.6899, equal, one more config line -> discard.
Next: reg_alpha 10 on top of lambda 100 (L1 sets weak small leaves exactly to zero).

## Exp 72 - 7852e0d - 0.6902 (discard) [follow-up]
reg_alpha 10: +0.0003 = noise. Discard.
Next: episodic shocks at the carrier/airport level (hurricane months at an airport, a carrier's strike or bankruptcy month). Unlike
the national day shocks (exp 31/41/42) these are correlated with the model's own features. Training-only nuisance columns
"<cat>_shock" = (smoothed delay rate of the category in the row's month) - (its yearly rate); prepare() always returns 0 for them
(= a month like the category's average), additive-only so they just soak up the episode.

## Exp 73 - d552cc3 - 0.6847 (discard) [exploration]
Training-only category x month shock columns: -0.0052. Fourth failure of the "absorb the 2005 nuisance, predict at neutral" family
(exp 31, 41/42, 60, 73). Explaining 2005 better makes the rest of the model fit 2005 more sharply, not more transferably. Closed.
Next: ablations under the final regime (constraints + lambda 100) to see what can be deleted. First the two mean-distance features.

## Exp 74 - 84adf4b - 0.6889 (discard) [ablation]
Without the mean-distance features: -0.0010. They still earn their 4 lines. Ablation discarded.
Next: which weekday interactions are the useful ones (exp 56 showed fencing the weekday off completely costs 0.002)? Allow
weekday x time of day only: DayOfWeek in a group with CRSDepTime, out of the carrier/airport group.

## Exp 75 - ac6b8ab - 0.6885 (discard) [follow-up]
Weekday x time-of-day only: -0.0014; weekday x airport/carrier interactions are real and stable (business vs leisure airports).
Next: since the weekly pattern is that useful, make it cheap to address: WeekHour = (weekday-1)*24 + hour, one numeric axis on which
"Friday evening" or "Sunday afternoon" is a single interval (2 splits instead of 4).

## Exp 76 - 0ed9a9c - 0.6900 (discard) [exploration]
Hour of the week: +0.0001 = noise. Discard; the trees already build weekday x hour cells.
Next: another *ordering of the airports* (the mean-distance ordering was a keeper): how much more delayed an airport is late in the
day than early (smoothed delay rate for departures >= 15:00 minus < 15:00, fitted on train). Airport x time-of-day is the
interaction the flight block relies on; this gives it a direct axis ("delays build up here" vs "flat").

## Exp 77 - fa6b432 - 0.6892 (discard) [exploration]
Evening-gap ordering of airports: -0.0007. A target-based second view does not help (it is in-sample and fine-grained again).
Next: smoothing of the delay-rate codes, m 20 -> 200. It only changes the ordering of small airports (pulls them to the middle
instead of the extremes), i.e. takes trust from small categories.

## Exp 78 - de46f82 - 0.6902 (discard) [exploration]
m_smooth 200: +0.0003 = noise. Discard.
Next: drift-calibrated codes. Random-fold OOF codes lost 0.003 in exp 6, but random folds share the same episodes. Cross-fit over
*time* instead: rows of a quarter get carrier/airport delay rates fitted on the other three quarters, so in training the code is
exactly as informative as a rate from a different period is - which is the situation at prediction time (2006 vs 2005).

## Exp 79 - 61d8c8e - 0.6843 (discard) [exploration]
Codes cross-fitted over quarters: -0.0056, even worse than random folds. The codes must stay exact identities; what the model learns
*about* each airport/carrier from the labels is what transfers, not the 2005 rate as a number. Closed.
Next: main effects first. Stage 1 boosts a purely additive model (every feature alone, the holiday pair together) on all rows,
stage 2 continues from it with the interacting flight block. Main effects are then estimated on unfragmented data and interactions
only explain what is left (GA2M-style hierarchy), instead of being mixed from the first tree on.

## Exp 80 - 83d1052 - 0.6896 (discard) [exploration]
Additive-first two-stage boosting: 0.6896 vs 0.6899, equal with more code. Discard; plain boosting finds the same hierarchy.

## Synthesis after 80 experiments
- Best kept: 7217b71 0.6899, unchanged for 10 experiments.
- Last 10 were all neutral or negative: Distance additive, reg_alpha, hour-of-week, heavier code smoothing, two-stage boosting
  (neutral); evening-gap view, weekday restricted to time (-0.001); monthly shock columns, quarter-cross-fitted codes (-0.005).
- Firm conclusions: (1) codes must be exact identities - every blurred/cross-fitted variant loses 0.003-0.006; (2) nuisance
  absorption does not transfer; (3) target-based fine-grained extras are over-trusted because their in-sample or within-year value
  exceeds their cross-year value.
- One family not tried: target encodings that are leak-free *by construction* because they come from other rows - the "other role":
  for a departure of carrier C from airport O, the delay rate of C's flights *into* O (the inbound aircraft that will fly this
  departure), and for the destination D the rate of C's departures *from* D. No row contributes to its own encoding.

## Exp 81 - 3dbc519 - 0.6894 (discard) [exploration]
Other-role carrier-airport rates (inbound rate at the origin, outbound rate at the destination): -0.0005 = noise for 8 lines. Discard.
Even leak-free pair-level rates add nothing: carrier x airport structure of 2005 is either already reachable through the codes
or does not carry over.

Research (plateau, exp 80+): searched for write-ups of the mlcourse.ai "flight delays" in-class Kaggle competition, which uses this
exact data format (mlcourse.ai assignment 10; kashnitsky's CatBoost starter; marcoantnionemetala's notebook). Their XGBoost benchmark
on DepTime + Distance scores 0.68 on a same-year split; the tricks there (hour/minute, route and carrier-airport combinations fed to
CatBoost) are the ones already tried here, and under the 2005->2006 shift the combination features did not pay.
Next: simplification under the final regime - with reg_lambda 100, is the row/column sampling still needed?

## Exp 82 - 43f294e - 0.6882 (discard) [ablation/simplification]
No row/column sampling: -0.0017 (same as in exp 49). The sampling stays.
Next: the carrier is the feature with the strongest drift (exp 46/47). Make it additive-only: a carrier level without
carrier-specific time-of-day / airport / distance patterns (exp 53 only cut carrier x airports and was neutral).

## Exp 83 - cfc4318 - 0.6874 (discard) [follow-up]
Carrier additive-only: -0.0025. Carrier x time-of-day/distance/weekday patterns are needed (only carrier x airport is dispensable).
Next: gamma (min_split_loss) 2 - another small-cell regulariser next to lambda: do not even create splits with tiny gain.

## Exp 84 - 9e62427 - 0.6899 (discard) [follow-up]
gamma 2: identical 0.6899. Discard (no reason to carry it).
Next: noise calibration of the final regime (the first one, exp 14, was 50 experiments and several regime changes ago): same config,
random_state 7. Tells whether 0.6899 is typical or a lucky seed; not a candidate for keeping.

## Exp 85 - 73d0c3c - 0.6900 (discard) [noise calibration]
random_state 7: 0.6900 vs 0.6899. The result is not seed luck; seed noise in this regime is ~0.0002. Not kept (seed picking).
Next: time of year once more, but only as a *shape* modifier. Month/season levels of 2005 do not repeat (exp 44, 54), yet the
physics behind season x time-of-day x airport is stable (summer afternoon thunderstorms, winter mornings). Give the 2005 month
level to the booster as a training-only offset (so no tree can earn anything from the level) and let Season into the flight block.

## Exp 86 - e9a5b37 - 0.6882 (discard) [exploration]
Season in the flight block with the month level taken out by an offset: -0.0017. Better than plain Season (-0.004) but still
negative: even the seasonal *shape* of 2005 does not carry over. Time of year is closed for good.
Next: retest a decision taken in an older regime. Exp 32 (pooled holiday distance, no ids, free interactions) found that the five
minor holidays helped. With per-holiday ids and the additive fence each minor holiday now has its own 9 date cells learned from
single 2005 days - mostly noise if their true effect is small. Six big travel holidays only.

## Exp 87 - 45887f8 - 0.6888 (discard) [retest]
Six big holidays only under ids + fence: -0.0011, same as in exp 32. The minor holidays (three-day weekends: MLK, Presidents,
Columbus; Easter; Veterans) carry real, repeatable signal. The 11-holiday list stays.
Next: the one kept "airport type" descriptor is the mean flight distance. A hub differs from a spoke more clearly in the *spread*
of its flight distances (a spoke flies to one or two hubs, a hub everywhere): std of Distance per origin/dest, fitted on train.

## Exp 88 - a910864 - 0.6897 (discard) [exploration]
Std of flight distance per airport: -0.0002 = noise. Discard.

Next: the last untested form of the day-shock idea - sample weights instead of offsets/features. Rows from extreme 2005 days (very
high or very low national delay rate) say little about structure; weight = max(0.2, 1 - 4*|day rate - global rate|). Unlike the
nuisance variants this does not explain 2005 better, it only listens less to its unusual days.

## Exp 89 - c847b19 - 0.6880 (discard) [exploration]
Down-weighting extreme days: -0.0019. Wrong again about the 2005 "noise": in a pooled, balanced sample most delayed flights come
from bad days in 2006 as well, so what delayed flights look like on bad days (which airports, which hours) is exactly the
structure the model needs. One-run check of the mirror image: up-weight the extreme days (weight = 1 + 4*|deviation|).

## Exp 90 - 661122e - 0.6898 (discard) [follow-up]
Up-weighting extreme days: 0.6898, equal. Discard. Row weighting by day type has nothing to give in either direction.

## Synthesis after 90 experiments
- Best kept: 7217b71 0.6899 for 20 experiments now; confirmed not seed luck (0.6900 with another seed).
- Last 10: nothing kept. Neutral: two-stage boosting, other-role carrier-airport rates, gamma, distance spread, up-weighting.
  Negative: no sampling (-0.002), carrier additive-only (-0.0025), season as shape modifier (-0.002), six big holidays only (-0.001),
  down-weighting extreme days (-0.002).
- The picture is stable: every component of the kept model is needed (ablations of carrier code, mean distance, Minute, minor
  holidays, sampling, weekday interactions all cost 0.001-0.005) and nothing I add moves it by more than seed noise.
- Remaining directions are low-probability: a few retests of early decisions under the final regime, and small structural variants
  of the constraint groups.

Research (exp 90+): schedule-only delay models in the literature (FlightSense arXiv 2605.07364: schedule-only baseline AUC 0.73 same
year, the big unmodelled part is reactionary delay along the aircraft rotation chain; "Delay-absorption capability" arXiv 2512.08197).
Rotation chains need tail numbers, which are not in the data. Closest schedule proxy: how far into the carrier's operating day at
that airport a departure is (later = more legs behind the aircraft = more accumulated delay).
Next: DepAfterFirst = CRSDepTime minus the 5%-quantile departure time of the (carrier, origin) pair, fitted on train.

## Exp 91 - 4fa9b71 - 0.6889 (discard) [exploration]
DepAfterFirst (rotation proxy): -0.0010. Discard; a (carrier, origin)-level lookup is fine-grained 2005 schedule detail again.
Next: ablation/simplification of the year inference. Use the 2005 holiday table for every row (drops is_2006 and the 2006 table).
If the moving holidays (Thanksgiving, Easter, MLK, Memorial, Labor...) matter, this must cost something; if not, the code gets simpler.

## Exp 92 - 7b1d91d - 0.6904 (keep) [ablation/simplification]
2005 holiday table for every row: 0.6904 vs 0.6899. The year inference is NOT needed: aligning to the 2006 dates of the moving
holidays (mostly a one-day shift; Easter three weeks) buys nothing. Keep - the code loses the 2006 table and the weekday trick.
This corrects my reading of exp 15/33: the block works as "date windows around the 2005 holiday dates" and is robust to +-1 day
of misalignment; it is not a precise holiday-alignment effect. Next commit removes the dead 2006 table (must reproduce 0.6904).

## Exp 93 - 0521e90 - 0.6904 (keep) [cleanup]
Single holiday table, dead code removed: identical 0.6904. Keep. New best-kept.
Next: with a fixed-date table, the 2005 Easter window (Mar 23-31) lands on ordinary days in 2006 (Easter 2006 is Apr 16). Its cells
can only be noise for another year -> drop Easter from the list.

## Exp 94 - 75f4cf9 - 0.6905 (keep) [simplification]
Without Easter: 0.6905 vs 0.6904, equal -> keep (10 holidays; the Easter window was the one that cannot line up across years).
Next: ablation of Distance, never tested on its own. Origin/dest codes plus their mean distances may already cover it.

## Exp 95 - de1bd6f - 0.6895 (discard) [ablation]
Without Distance: -0.0010. It stays.
Next: exp 82 removed both sampling parameters (-0.0017). Which one carries it? Remove subsample only (keep colsample_bytree 0.8).

## Exp 96 - 8166843 - 0.6898 (discard) [ablation/simplification]
No row subsampling: -0.0007 (and training 6.6s instead of 2.8s). Both sampling parameters contribute; they stay.
Next: simplification that follows from exp 92. With a fixed-date table the two holiday features (which holiday, signed distance)
just index a date. Replace them by ONE feature: the day of year if it lies within 4 days of a holiday, else -1. Less code (no ids,
no signed distances), same information.

## Exp 97 - 28679bf - 0.6908 (keep) [simplification]
One HolidayDay feature instead of id + distance: 0.6908 vs 0.6905, equal or better with less code. Keep. New best-kept.
The calendar part of the model is now simply: weekday (free to interact) + an additive effect for each date inside a holiday window.
Next: another simplification - the m-estimate smoothing of the delay-rate codes. Trees only use the order of the codes and
m 20 vs 200 made no difference (exp 78); plain group means would be a one-liner.

## Exp 98 - 1e4c32b - 0.6904 (discard) [simplification]
Unsmoothed group means: 0.6904 vs 0.6908 (-0.0004, noise level) and 4 lines shorter. Judgment call: discarded anyway. Without
smoothing an airport with one or two training rows gets its own label as its code (0 or 1), a small target leak that the m-estimate
prevents; the saving does not justify carrying a known defect, and the number did not improve.
Next: plateau check of the window width in the new single-feature form: +-3 (fewer date cells).

## Exp 99 - 749ab52 - 0.6907 (discard) [plateau check]
Windows +-3: 0.6907 vs 0.6908, equal. +-4 stays (plateau: +-3 = +-4 > +-7).
Next: exp 92 showed that a one-day misalignment of the windows does not matter, so neighbouring dates must have similar effects.
Then single-day cells (each learned from one 2005 day) are noisier than needed: pool pairs of days (day of year // 2).

## Exp 100 - fb3444a - 0.6900 (discard) [follow-up]
Two-day cells: -0.0008. Single dates stay.

## Synthesis after 100 experiments
- Best kept: 28679bf 0.6908. Features: CRSDepTime, Distance, Minute (additive), DayOfWeek, HolidayDay (additive; day of year inside
  +-4-day windows around 10 fixed 2005 holiday dates, else -1), delay-rate codes of carrier/origin/dest (m=20), mean distance of
  origin/dest. XGBoost: 600 trees, depth 6, lr 0.05, subsample/colsample 0.8, reg_lambda 100, interaction constraints.
- The last 10 gained +0.0009 purely by *removing* things: the year inference and 2006 holiday table (exp 92), Easter (94), the
  id/distance pair in favour of one date feature (97). Ablations that cost: Distance (-0.001), subsample (-0.0007), 2-day cells.
- Lesson: my "holiday alignment" story was more elaborate than what the data supports; the working mechanism is simply an additive
  per-date effect restricted to holiday windows. Ablating one's own explanations is as productive as adding features.
- Remaining ideas are small: column sampling per node instead of per tree, and a final check of nearby depths.

Next: additive fences were the pattern behind three gains (holiday dates, Minute, and indirectly lambda). One block not tried as
additive-only: the destination. A departure delay is decided at the origin; destination x time-of-day / carrier cells may be
mostly 2005 detail. Destination features (Dest_te, Dest_mean_dist) in their own group.

## Exp 101 - d15ea7f - 0.6890 (discard) [exploration]
Destination additive-only: -0.0018. Destination x time/carrier/origin-side interactions are real. Discard.
Next: with interactions confirmed useful for every flight feature, one more look at tree size between the tested 6 and 8: depth 7.

## Exp 102 - 2d9ad42 - 0.6901 (discard) [plateau check]
Depth 7: -0.0007. Depth 6 stays (6 > 7 > 8, and 4 < 6).
Next: column sampling per node instead of per tree (colsample_bynode 0.8, colsample_bytree off). Per-tree sampling decides which
*block* a tree can belong to; per-node sampling decorrelates splits inside the flight block, where the over-trust lives.

## Exp 103 - 178b448 - 0.6905 (discard) [exploration]
colsample_bynode instead of bytree: 0.6905 vs 0.6908, equal. Discard.

Research: Szilard Pafka's benchm-ml (github.com/szilard/benchm-ml), the benchmark this airline data comes from (train 2005-2006, test
2007): random forests and *very deep, slow* boosting (depth 16, lr 0.01, 1000 trees) are the most accurate there (AUC 0.72-0.73 at
100K rows), better than depth 6 / lr 0.1 / 300 trees. I only ever tried depth 8 at lr 0.05, where each deep tree takes a big step.
Next: the deep-and-slow regime within the 60s limit: depth 12, lr 0.02, 400 trees (lambda 100 and sampling unchanged).

## Exp 104 - 1aa4747 - 0.6903 (discard) [exploration]
Depth 12, lr 0.02, 400 trees: 0.6903 vs 0.6908 - about equal, and training is only 3.8s. Not kept (no gain), but the deep-slow
regime is clearly viable under lambda 100. Follow-up with the benchmark's actual setting: depth 16, lr 0.01, 1000 trees.

## Exp 105 - 8fcba57 - 0.6890 (discard) [follow-up]
Depth 16, lr 0.01, 1000 trees: -0.0018 and 20s training. The benchmark's deep regime does not help here (one training year, a
year of shift, 200K rows). Both deep-slow commits discarded; back to 28679bf.
Next: ablate my own explanation once more. Is the *selection* of holiday windows what matters, or would an additive effect for
every date do (no holiday list at all)? HolidayDay := day of year for all days.

## Exp 106 - 1f0559f - 0.6833 (discard) [ablation]
Every date with its own additive effect: -0.0075. So the restriction to holiday windows is the point: per-date effects of 2005
transfer near holidays and are harmful everywhere else. This part of the explanation holds.
Next: the complementary ablation - no date feature at all - to put a number on what the holiday windows are worth in the final model.

## Exp 107 - c778224 - 0.6861 (discard) [ablation]
No date feature: 0.6861. So: holiday-window dates +0.0047, every date -0.0028 relative to no date information. Ablation discarded.
Next: seed check of the current best (random_state 7) - are the +0.0009 from the last simplifications (0.6899 -> 0.6908) real?

## Exp 108 - e937122 - 0.6910 (discard) [noise calibration]
random_state 7 on the current best: 0.6910 vs 0.6908. The +0.0009 from the simplifications is real (0.690 -> 0.691 on both seeds).
Next: is the holiday block over-trusted? Each date cell is learned from a single 2005 day (holiday effect + that day's weather).
Because of the constraints, trees that split on HolidayDay contain nothing else, so their leaves can be scaled after training:
multiply the holiday trees by 0.6 (exploration; hacky booster-JSON edit, only worth keeping if the gain is clear).

## Exp 109 - e0d2431 - 0.6911 (discard) [exploration]
105 of the 600 trees are holiday trees; scaling their leaves by 0.6: 0.6911 vs 0.6908 = noise. The holiday block is neither over-
nor under-trusted in a way AUC can see. Discard (hack without gain).
Next: the three fences that were each neutral on their own (carrier x airports, origin x dest, Distance interactions; exp 53, 55,
71) combined under lambda 100: every entity (carrier, origin, dest) interacts only with the schedule (weekday, time of day),
Distance additive. If equal, the model is "entity x schedule" only - a much smaller hypothesis space.

## Exp 110 - 215d917 - 0.6884 (discard) [follow-up]
Entity x schedule only: -0.0024. The three individually neutral fences are not neutral together - the model needs *some* route to
entity x entity structure (carrier x origin, origin x dest via distance). Discard.

## Synthesis after 110 experiments
- Best kept: 28679bf 0.6908 (0.6910 with another seed).
- Last 10: nothing kept. Informative ablations: no date feature 0.6861, every date 0.6833, holiday windows 0.6908 - the window
  selection is the whole effect. Depth 7/12/16 and deep-slow boosting (benchm-ml regime) do not beat depth 6. Scaling the
  holiday trees, per-node column sampling: noise. Additive-only destination and entity-x-schedule-only constraints hurt.
- The model is at a broad optimum in every direction I can think of; what remains untested is the pair-level information
  (route, carrier-origin, carrier-dest) in *additive-only* form - in free-interaction form it gave +0.0006..0.0012 early on.

Research: scikit-learn "Target Encoder's Internal Cross fitting" example and MetricGate "Pitfalls of target encoding" / "CatBoost
ordered encoding": for high-cardinality keys the encoding of the training rows must be cross-fitted, with smoothing for the long tail.
Next: pair-level delay rates (route, carrier-origin, carrier-dest), 5-fold cross-fitted for training rows, each as its own
additive-only constraint group (so they cannot be combined into ever finer cells). Needs >= +0.002 to justify ~15 lines.

## Exp 111 - 0f2bb58 - 0.6896 (discard) [exploration]
Additive-only OOF pair rates: -0.0012 with 15 more lines and slower eval. Pair-level 2005 delay rates do not help in any form
(free, additive, leak-free other-role, slot). Closed.
Next: plateau re-check of reg_lambda in the final feature set (it was tuned at exp 65-67, before the calendar simplifications): 50.

## Exp 112 - 3b6b8fd - 0.6905 and Exp 113 - 7d79b0b - 0.6911 (both discard) [plateau check]
reg_lambda 50 / 100 / 200 = 0.6905 / 0.6908 / 0.6911: flat within noise. 100 stays (no reason to chase +0.0003).
Next: Minute works as an additive effect. Does it do better as part of a full-resolution time-of-day main effect, i.e. allowed to
interact with CRSDepTime only (minute patterns that differ by hour: bank structures)? Group [Minute, CRSDepTime].

## Exp 114 - d1f0c23 - 0.6907 (discard) [follow-up]
Minute x CRSDepTime allowed: 0.6907 vs 0.6908, equal. The plain additive Minute stays.
Next: simplification - the two mean-distance features together are worth 0.001 (exp 74). Is the destination one needed?
Keep only Origin_mean_dist.

## Exp 115 - 7fec6d2 - 0.6899 (discard) [ablation]
Origin mean distance only: -0.0009. The destination's mean distance is the useful half (or both are); it stays.
Next: Minute-style additive main effect for the hour of day: an additive copy of the hour (own constraint group). The time-of-day
main effect is the strongest signal; estimated on its own it is not fragmented by the entity splits of the flight block.
(The two-stage variant in exp 80 forced *all* main effects first and was neutral; this adds just the one.)

## Exp 116 - a961572 - 0.6911 (discard) [exploration]
Additive Hour: +0.0003 = noise. Discard.
Next: combine the near-misses, as program.md suggests when ideas run out. Several changes were each +0.0003..0.0004 and all point
the same way (smoother, more shrunk): reg_lambda 200, lr 0.025 with 1200 trees, m_smooth 200, additive Hour. Together they should
show +0.001 if the individual effects are real and not noise.

## Exp 117 - a3ba877 - 0.6917 (keep) [combination of near-misses]
All four together: 0.6917 vs 0.6908, +0.0009 (4x seed noise): the small effects were mostly real and add up. Keep. New best.
Next: ablate the pieces one at a time from the combination to keep only what is needed, starting with the one that adds code
(additive Hour).

## Exp 118 - dd23bd8 - 0.6917 (keep) [ablation/simplification]
Without the additive Hour: 0.6917, identical -> keep the simpler version. Next: m_smooth back to 20.

## Exp 119 - ae5d32c - 0.6912 (discard) [ablation]
m_smooth 20 instead of 200: -0.0005. The heavier smoothing (small categories ordered near the middle) stays.
Next: lr 0.05 x 600 trees instead of 0.025 x 1200 (half the training time if equal).

## Exp 120 - 31418ab - 0.6913 (discard) [ablation]
lr 0.05 x 600 instead of 0.025 x 1200: -0.0004. Small, but same code and training is still under 5s -> the slower setting stays.

## Synthesis after 120 experiments
- Best kept: dd23bd8 0.6917. Features: CRSDepTime, Distance, Minute (additive), DayOfWeek, HolidayDay (additive; day of year inside
  +-4-day windows around 10 fixed holiday dates, else -1), delay-rate codes of carrier/origin/dest (m=200), mean distance of
  origin/dest. XGBoost: 1200 trees, depth 6, lr 0.025, subsample/colsample_bytree 0.8, reg_lambda 200, interaction constraints.
- Last 10: combining four near-misses gave +0.0009; ablation showed three of them carry it (lambda 200, m_smooth 200, slower
  learning ~ +0.0003..0.0005 each) and the additive Hour none. Additive pair rates, Minute x time, origin-only mean distance,
  entity-x-schedule constraints: neutral or negative.
- Small effects in the "smoother / more shrunk" direction are real and additive even when each is at noise level alone.
- Next (remaining minutes): one more step in the same direction (lambda/trees), then wrap up.

Research: shrinkage notes (apxml "GBM shrinkage / learning rate"; gbm package vignette citing Friedman 2001/2002): smaller learning
rates keep helping with decreasing marginal utility, and the optimal number of trees does not scale exactly with 1/eta.
Next: second round in the same direction with the remaining near-misses: lr 0.0125 x 2400 trees, m_smooth 500, reg_alpha 10
(exp 72: +0.0003). Training should stay around 10s.

## Exp 121 - 0530df7 - 0.6915 (discard) [follow-up]
Second round of "smoother": 0.6915 vs 0.6917. Saturated - diminishing returns as Friedman describes. Discard.
Next: seed check of the final best (random_state 7), for the record.

## Exp 122 - 85adae1 - 0.6916 (discard) [noise calibration]
random_state 7 on the final best: 0.6916 vs 0.6917. Stable. Not kept.
Next: with lambda 200 and the slower learning rate in place, a last look at depth 5 vs 6 (4 and 7/8 were both worse in earlier regimes).

## Exp 123 - 25cef5c - 0.6909 (discard) [plateau check]
Depth 5: -0.0008. Depth 6 is the optimum in every regime tested (4, 5 < 6 > 7, 8, 12, 16).
Next: Minute x carrier. Scheduling conventions for the minute of the hour are carrier-specific (some carriers publish :00/:05
grids, others odd minutes), so the meaning of a minute may depend on the carrier. Group [Minute, UniqueCarrier_te].

## Exp 124 - fcffadf - 0.6916 (discard) [exploration]
Minute x carrier: 0.6916, equal. The plain additive Minute stays.
Next: upper side of the lambda plateau in the final regime (50/100/200 rose slowly): reg_lambda 400.

## Exp 125 - 9c11824 - 0.6916 (discard) [plateau check]
reg_lambda 400: 0.6916, equal. 200 stays.
Next: ablate another of my explanations. I concluded that the delay-rate codes act as category *identities* (exp 6, 21, 79). If
that is all they are, an arbitrary order (alphabetical index) would do as well and the target encoding could be deleted.

## Exp 126 - 0a67a91 - 0.6895 (discard) [ablation]
Alphabetical codes: 0.6895, only -0.0022. So the codes are indeed mostly identities, and the delay-rate *order* adds about 0.002
on top (similar categories are neighbours, so one split forms a sensible group). Ablation discarded.
Next: if the order is worth 0.002, a cleaner order may be worth a bit more. The raw delay rate of an airport/carrier is confounded
by its schedule mix (mostly-evening airports look worse). Order by the mean *residual* after removing the hour-of-day delay rate.

## Exp 127 - 78fabeb - 0.6914 (discard) [exploration]
Hour-adjusted ordering of the codes: 0.6914 vs 0.6917, equal. The raw delay-rate order is as good. Discard.
Remaining time: leave-one-out ablations of the final model, to end with a clean attribution of what each ingredient is worth
(and to catch anything that has become dispensable). First: no interaction constraints at all.

## Exp 128 - 11b2616 - 0.6911 (discard) [ablation]
No interaction constraints: 0.6911 vs 0.6917, only -0.0006 (they were worth +0.0016 at lambda 1 with the id/distance pair).
The heavy leaf shrinkage has taken over most of their job. Kept anyway: 6 plain config lines, 3x seed noise, and they encode the
one structural finding (date effects must not interact with the flight features). A defensible alternative is to drop them.
Next: reg_lambda back to the default 1 in the final model.

## Exp 129 - 0bf13bd - 0.6901 (discard) [ablation]
Default reg_lambda: -0.0016. Stays at 200. Next: without Minute.

## Exp 130 - 485a82f - 0.6915 (keep) [ablation/simplification]
Without Minute: 0.6915 vs 0.6917 = seed noise. In the final regime (lambda 200, slow learning) the additive Minute is no longer
needed -> removed. Keep. (It was worth +0.0009 at lambda 100 / lr 0.05, exp 61.)

## Synthesis after 130 experiments
- Best kept: 485a82f 0.6915 (dd23bd8 with Minute scored 0.6917; same within noise, this one is simpler).
- Leave-one-out values in the final regime: holiday-window dates +0.0047, reg_lambda +0.0016, delay-rate order of the codes
  +0.0022 (vs alphabetical), Distance +0.0010, interaction constraints +0.0006, m_smooth 200 +0.0005, slow learning +0.0004,
  Minute +0.0002 (removed).
- Ingredients become dispensable as the regularisation gets stronger (constraints, Minute); worth re-checking the mean-distance
  features too.

## Exp 131 - a4a9483 - 0.6905 (discard) [ablation]
Without the mean-distance features: -0.0010. They stay.
Next: the constraints again, now that Minute is gone and the only fence left is the one around HolidayDay (exp 128 with Minute:
-0.0006). If the loss is at noise level now, they go.

## Exp 132 - e7f219c - 0.6912 (keep) [ablation/simplification]
No constraints at all: 0.6912 vs 0.6915, -0.0003 = noise level. Keep the simpler model: with reg_lambda 200 the fence around the
holiday dates is no longer needed. Note the two simplifications together (Minute, constraints) cost 0.0005 against dd23bd8
(0.6917): about 1.5 standard errors of a paired AUC difference on 50K rows, accepted for 8 fewer lines per the simplicity criterion.
dd23bd8 stays in the history (and in artifacts/) as the highest-scoring kept commit.
Next: row/column sampling in the final regime (was worth 0.0017 at lambda 100, exp 82).

## Exp 133 - 430c48d - 0.6893 (discard) [ablation]
No row/column sampling: -0.0019. Stays.

## Second bookkeeping error found (affects exp 91-133)
Printing train.py at e7f219c shows the block from exp 90 (661122e) still present:
`weight = 1 + 4 * |day rate - global rate|` and `model.fit(..., sample_weight=weight)`.
After exp 90 (0.6898 vs 0.6899) I logged "discard" but did not run `git reset`, the same slip as after exp 30. Exp 91 was committed
on top of 661122e, so ALL runs from exp 91 to exp 133 were trained with these sample weights.
What this changes:
- Every results.tsv row and log entry from exp 91 on describes its model without mentioning the weights; the kept states
  7b1d91d, 0521e90, 75f4cf9, 28679bf, a3ba877, dd23bd8, 485a82f, e7f219c all include them.
- The syntheses after 100/110/120/130 experiments and the leave-one-out values listed there are measured on a weighted model.
- Comparisons among exp 91-133 share the same base, so their keep/discard decisions hold relative to each other. Comparisons
  across the boundary (e.g. exp 92's 0.6904 vs 0.6899 of 7217b71) mix a weighted and an unweighted model; exp 90 itself measured
  that difference as -0.0001, so the conclusions drawn are unlikely to flip, but they were not measured cleanly.
- Whether the weights are worth anything in the final regime is untested. Next run removes them from e7f219c to decide.

## Exp 134 - 8e0078d - 0.6913 (keep) [ablation/simplification]
Day sample weights removed: 0.6913 vs 0.6912 with them. Equal, so they go (4 lines fewer) and from this commit on the code is
what the log says it is. results.tsv: 661122e stays `discard` with a note that it was never reset and its change lived in all
commits from exp 91 to exp 133 until removed here.
Audit after this second slip: compared `git log main..HEAD` with the statuses in results.tsv (output of the check is in the
session); the only commits on the branch not logged as keep should be 661122e, whose change is now reverted.

Next: the two borderline simplifications (Minute, exp 130; constraints, exp 132) were decided on the weighted model. Re-check the
closer one on the clean model: put the holiday fence back on 8e0078d.

## Exp 135 - 65129c7 - 0.6918 (keep) [re-check on the clean model]
Holiday fence back on the unweighted model: 0.6918 vs 0.6913, +0.0005. Three measurements of this fence in the late regime now
agree in sign (+0.0006 with Minute, +0.0003 on the weighted base, +0.0005 clean). Keep: it is the higher score and 5 plain config
lines. This reverses exp 132, whose -0.0003 was measured on the model that still carried the stray sample weights.
Next: the same re-check for Minute (additive group) on the clean model.

## Exp 136 - a245d0f - 0.6917 (discard) [re-check on the clean model]
Additive Minute back: 0.6917 vs 0.6918, equal -> it stays out (exp 130 confirmed on the clean model). Reset done and verified:
HEAD is 65129c7.

## Exp 137-139 - 3690ff5 / 72e1163 / 313fa3a (all discard) [leave-one-out on the clean final model 65129c7 = 0.6918]
The attribution numbers in the 130-experiment synthesis were measured on the model with the stray sample weights, so the main
ones are re-measured here on the clean final model (each run reset to 65129c7 by hash afterwards, HEAD verified):
- no HolidayDay: 0.6871 (-0.0047)   - default reg_lambda: 0.6900 (-0.0018)   - no mean-distance features: 0.6908 (-0.0010)
They agree with the earlier figures (-0.0047 / -0.0016 / -0.0010), so the weights did not distort the attribution.
Next, same procedure: Distance, row/column sampling, m_smooth 20.

## Exp 140-142 - 5a0f024 / f4ac096 / 8c58ac5 (all discard) [leave-one-out on the clean final model]
no Distance 0.6912 (-0.0006), no row/column sampling 0.6902 (-0.0016), m_smooth 20 0.6915 (-0.0003). All stay as they are.

## Exp 143 - bfd4b8b - 0.6915 (discard) [noise calibration]
random_state 7 on the final commit: 0.6915 vs 0.6918. Seed spread here is ~0.0003, larger than the 0.0001-0.0002 seen at
exp 85/108/122. Consequence: the final model is "0.6915-0.6918", and the +0.0005 of the holiday fence (exp 135) is only about
1.5x that spread - kept on three same-sign measurements, but it is the least certain ingredient.

# Final summary (run `oct3`)

**Result.** Final commit / branch tip: `65129c7`, Eval AUC **0.6918** (0.6915 with another seed), also the highest Eval AUC of
the run. Baseline `2e14489`: 0.6743. Gain +0.0175. results.tsv has 145 rows after the baseline: 144 completed runs + 1 crash
(144 experiments by this log's numbering, exp 41 has two rows); 33 kept commits.

**Final model.** Features: CRSDepTime, Distance, DayOfWeek (ordinal), HolidayDay (day of year if within 4 days of one of 10 fixed
US holiday dates, else -1), smoothed delay-rate codes of carrier/origin/dest (m=200, in-sample), mean flight distance of
origin/dest. XGBoost: 1200 trees, depth 6, lr 0.025, subsample 0.8, colsample_bytree 0.8, reg_lambda 200, interaction constraints
that keep HolidayDay additive. Training ~4.5s, eval ~19s.

**What each ingredient is worth (leave-one-out on the clean final model):** holiday-window dates +0.0047; reg_lambda 200 vs 1
+0.0018; row/column sampling +0.0016; mean-distance features +0.0010; Distance +0.0006; holiday fence +0.0005; m_smooth 200 vs 20
+0.0003. From earlier regimes (not re-measured clean): delay-rate codes instead of native categoricals +0.003 (exp 3), carrier
code +0.0055 (exp 47), delay-rate order vs alphabetical +0.0022 (exp 126, weighted model), depth 6 vs 4/5/7/8 +0.001.

**Exp 144 - 4a9d1be - 0.6916 (discard), run after this summary was first written.** lr 0.05 x 600 trees on the clean model:
0.6916 vs 0.6918, within the seed spread. So the "slow learning" gain of exp 120 (+0.0004, measured on the weighted model) is NOT
confirmed on the clean model; 0.025 x 1200 is kept only because it is not worse. Halving the trees is an equally good choice.

**What worked.** (1) Ordinal delay-rate codes instead of native categoricals. (2) Removing all time-of-year information except
per-date effects inside holiday windows: Month/season/every-date features cost 0.002-0.008, holiday windows gain 0.0047.
(3) Heavy L2 on leaf weights (the benefit of slower learning is within noise, see exp 144). (4) Ablating my own constructions: the 2006 holiday table with year inference,
Easter, the id/distance pair, Minute and the sample weights all turned out unnecessary.

**What did not.** Anything fine-grained and target-based (pair/slot/route rates in-sample, out-of-fold, additive-only or
leak-free: -0.001..-0.010); cross-fitted codes (-0.003..-0.006); nuisance absorption of 2005 shocks (day/month offsets, shock
columns: 0..-0.005); recency or day-type weights; native categorical carrier; monotone constraints (-0.008); DART; very deep
slow boosting; geography recovered by MDS, arrival time, rotation proxies (noise).

**Corrections to this log (two bookkeeping errors of mine, both a missed `git reset` after logging a discard).**
1. Exp 30 (`10cec75`, Minute): stayed in the code from exp 31 on; found at exp 61, row corrected to keep; removed for good at
   exp 130 / confirmed exp 136.
2. Exp 90 (`661122e`, day sample weights): stayed in the code for exp 91-133; found when printing train.py before the wrap-up;
   removed in exp 134 (`8e0078d`, 0.6913 vs 0.6912: no effect). All rows and syntheses for exp 91-133 describe models that also
   had these weights; the eight kept rows in that range are annotated in results.tsv. `661122e` is the one commit on the branch
   whose status is discard. One decision taken on the weighted model was reversed after re-measuring clean (the holiday fence,
   exp 132 -> exp 135). The attribution table above was re-measured on the clean model.
After the second slip every reset was done by hash and HEAD verified.

**Caveats.** ~140 keep/discard decisions were made on the same 50K eval rows, many at the 0.0003-0.001 level, so the eval AUC of
the final model is somewhat optimistic for the holdout; the large effects (codes, holiday windows, no time of year, lambda,
sampling) should transfer, the small ones (holiday fence, m_smooth, mean distance) may not.

**What I would try next.** A second training year would change the problem most (the holiday table and the codes each rest on
one year). Within these rules: a proper temporal validation inside 2005 to choose regularisation without touching eval; and
re-measuring the exp 91-133 near-misses on the clean model, which I did not have time to do.
