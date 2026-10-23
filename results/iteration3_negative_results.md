# Third iteration: what did not help (all numbers: Kaggle replica, expected MAP@5, five folds)

Scripts: `src/extra/` (`lite.py` builds a light context from the cached v6 features; run from that folder with the project's Python).
Reference: the submitted ensemble (LightGBM LambdaRank + soft labels + Poisson), mean MAP@5 0.4139.

## 1. More diverse ensemble (XGBoost, CatBoost) - `src/extra/div.py`

| model (5-fold cross-fit by users) | mean MAP@5 | val12 |
|---|---|---|
| LightGBM LambdaRank, deterministic labels | 0.4125 | 0.4301 |
| LightGBM LambdaRank, soft labels | 0.4131 | 0.4319 |
| LightGBM Poisson | 0.4088 | 0.4262 |
| XGBoost rank:ndcg | 0.4134 | 0.4312 |
| XGBoost rank:pairwise | 0.4136 | 0.4322 |
| CatBoost YetiRank (23 min for the cross-fit) | 0.4127 | 0.4302 |
| ensemble of the three LightGBM models (submitted) | 0.4139 | 0.4322 |
| ensemble of all six | **0.4155** | 0.4338 |

The best ensemble gains +0.0016 over the submitted one, which is inside the noise of the evaluation (about 1,100 users per fold).
Not adopted.

## 2. Where the model loses - `src/extra/diag.py`, `series_diag.py`

Mean over the five folds (users with a truth, LightGBM cross-fit):

| quantity | value |
|---|---|
| share of true programmes the user had already committed to in the history ("seen") | 0.41 |
| candidate coverage of seen / new true programmes | 1.00 / 0.31 |
| hit rate of the top-5 on seen / new true programmes | 0.71 / 0.04 |
| users whose truth has a tie at the 5th place | 0.64 |
| MAP@5 of users with / without a tie | 0.365 / 0.499 |
| MAP@5 by history activity, quartiles from the lightest to the heaviest | 0.244 / 0.390 / 0.476 / 0.534 |

Decomposition of the expected truth mass (a tied programme counts with probability places-left / tied-size):
seen and sure 43.3%, seen and tied 14.4%, new and sure 15.2%, new and tied 27.2%. So 42% of the mass is new programmes, but two thirds of that
sits in tie groups, i.e. is a random draw among one-off airings. Hypothesis "new programmes are new episodes of a series the user already
watched" (series key from the title): **0.0-0.1%** of the new true programmes - rejected (ids are almost one per title, 1.05 ids per key).

## 3. If the host broke ties by "earliest first airing in the window" - `src/extra/tie_potential.py`

A model that knows this rule (feature: days to the first airing of the candidate in the window) would score 0.4179 instead of 0.4168 on the
earliest-first truth (+0.001) and 0.4135 instead of 0.4125 on the random-tie expectation. Not worth a probing submission.

## Conclusion

The model is saturated with the available signals: the remaining loss is concentrated in new one-off programmes and in ties, both close to
random. Further modelling effort has a low expected return; the final model stays the submitted ensemble (submission 5).
