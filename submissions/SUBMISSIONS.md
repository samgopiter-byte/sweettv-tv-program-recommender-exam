# Kaggle submissions log (all submissions made, none omitted)

Competition: `sweettv-tv-program-recommender` (late submission). Description used for every submission:
`Александр Голанков, Alex_Gol777`. Scores are as returned by Kaggle.

| # | Kaggle ref | File | md5 | Method | Local score before submitting | Public | Private |
|---|---|---|---|---|---|---|---|
| 1 | 57530793 | `submitted_v1_ref57530793.csv` | `1478f62df8a681c9870681a5580053fa` | personal history x exposure + 0.25 x ALS | validation MAP@5 0.2855 / 0.3059 (own seconds-based truth) | 0.28940 | 0.26088 |
| 2 | 57531976 | `submission_blend_v2.csv` | `6554a8ca8f2faca92acc287e26fc07f1` | habit model (seconds) + personal history x exposure | 0.3069 / 0.3271 (same truth) | 0.30515 | 0.27332 |
| 3 | 57531974 | `submission_ltr_v2.csv` | `24c843c41dc791518f0fc8754e29ac47` | LightGBM LambdaRank over seconds-based scorers | 0.3145 / 0.3375 (same truth) | 0.32193 | 0.29401 |
| 4 | 57533998 | `submission_v3_ensemble.csv` | `d72d7ecc7df11b8f7e87061facc500d9` | ensemble of three re-rankers trained on the Kaggle-replica labels (committed airings >= 80%) | Kaggle-replica expected MAP@5 0.4246 (cross-fit) / 0.4237 (transfer), 12-week fold | 0.44705 | 0.43831 |
| 5 | 57534905 | `submission_v6_ensemble.csv` | `0e5de6ecab6cf6433ee7e6516a4c3440` | same ensemble, tuned (15 leaves, 200 rounds), new signals (commit priors, hour/channel habits) and the four course methods (ALS, AR, CBF, PPR) on committed airings as features | replica 0.4322 (cross-fit) / 0.4290 (transfer), 12-week fold; mean of 5 folds 0.4139 / 0.4134 | **0.45130** | **0.44313** |

## The five attempts behind submission 4 (selected by the local Kaggle replica, not by the leaderboard)

Replica = expected MAP@5 over random tie-breaks, template users that have a truth, truth rebuilt from the Data-page recipe
(airings watched >= 80%, programmes ranked by their number, exactly five per user). Folds: val12 (8-week history, 12-week
target, the test horizon), val, val2, val3, val5 (history 6-12 weeks, 8-week targets).

| Attempt | cross-fit by users, mean of 5 folds | cross-fit, val12 | transfer to an unseen fold (val12 / val / val2 mean) | transfer, val12 |
|---|---|---|---|---|
| 1. old v3 features, LambdaRank on replica labels | 0.3973 | 0.4176 | 0.3999 | 0.4169 |
| 2. all features, LambdaRank (deterministic top-5 labels) | 0.4064 | 0.4233 | 0.4072 | 0.4224 |
| 3. all features, Poisson regression of committed airings per week | 0.4054 | 0.4200 | 0.4044 | 0.4186 |
| 4. all features, graded LambdaRank on tie-aware soft labels | 0.4060 | 0.4236 | 0.4069 | 0.4220 |
| **5. ensemble of 2, 3 and 4 (per-user z-scored average)** | **0.4083** | **0.4246** | **0.4090** | **0.4237** |
| reference: popularity | 0.2668 | 0.2858 | | |
| reference: best single scorer (slot habit, pooled) | 0.3760 | 0.3848 | | |

Notes
* Submissions 1-3 were selected on a seconds-based validation truth that turned out to differ from the host's definition of
  the truth (Kaggle Data page); submission 4 uses the host's definition. Only the selection method changed, nothing was
  tuned on the Kaggle score: the five attempts were ranked by the local replica and only the winner was submitted.
* Replica calibration on the three earlier submissions (12-week fold): replica 0.281 / 0.290 / 0.310 vs the mean of
  Kaggle public and private 0.275 / 0.289 / 0.308. For submission 4 the replica (0.424) is below Kaggle (public 0.447,
  private 0.438): the test has 20 weeks of history instead of 8.
* Submission 1 was produced before the window-clipping fix of 2020-10-22; regenerating it with the fixed code reproduces
  the file byte for byte (no view crosses the test cutoff).
* Submission 5 was sent after the user's explicit confirmation, as the fifth attempt of 2026-10-22 (the teacher's guidance is
  about five a day; the Kaggle counter showed 96 of 100 left after four). It is the version that the repository code
  reproduces (#4 was produced by an earlier state of the code), so it is the designated final submission unless a later one
  beats it. The gain over #4 (+0.0043 public, +0.0048 private) matches the replica's prediction (+0.005).
* Designated final submission: currently #5 (highest public and private), to be marked "Selected" on the Submissions tab before
  the screenshot, which must show both the Public and the Private column. Further attempts planned for 2026-10-23.
