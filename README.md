# SWEET.TV TV-program Top-5 recommender

Individual final-exam project of the course **LLM4Rec** (HSE University, AVITO group).
Task: for each of 1,260 users predict the five TV programmes they will watch most in weeks 31-42 of 2020
([Kaggle competition `sweettv-tv-program-recommender`](https://www.kaggle.com/competitions/sweettv-tv-program-recommender),
metric MAP@5, late submissions).

## Kaggle results (all submissions that were made)

| # | File | Method | Public | Private |
|---|---|---|---|---|
| 1 | `submissions/submitted_v1_ref57530793.csv` | personal history x exposure + 0.25 x ALS | 0.28940 | 0.26088 |
| 2 | `submissions/submission_blend_v2.csv` | habit model (seconds) + personal history x exposure | 0.30515 | 0.27332 |
| 3 | `submissions/submission_ltr_v2.csv` | LightGBM LambdaRank over seconds-based scorers | 0.32193 | 0.29401 |
| 4 | `submissions/submission_v3_ensemble.csv` | ensemble of three re-rankers on the Kaggle-replica labels | 0.44705 | 0.43831 |
| 5 | `submissions/submission_v6_ensemble.csv` | same ensemble, tuned, plus ALS / association rules / CBF / PPR on committed airings | **0.45130** | **0.44313** |

Designated final submission: **#5**, `submissions/submission_v6_ensemble.csv`, md5 `0e5de6ecab6cf6433ee7e6516a4c3440`.
Details (Kaggle refs, md5, local scores, the five attempts behind #4 and #5): [`submissions/SUBMISSIONS.md`](submissions/SUBMISSIONS.md).
Submissions 1-3 were chosen on a seconds-based validation truth that turned out to differ from the host's definition; from #4 on the
selection uses the replica of the host's truth described below.

## How the Kaggle score is built, and the local replica

The Data tab states how the hidden truth was made: for every user and every airing the *screen time* (watched time / airing length) is
computed; only airings with screen time >= 0.8 count; per user the programmes are ranked by the number of such "committed" airings and exactly
five are kept (order matters, `tv_show_id = 0` removed). Public and private scores split the users 50/50. The metric is the standard
MAP@5 (AP normalised by min(5, |truth|), as on the teacher's slides).
Many users have ties at the 5th place and the host does not say how they were broken, so the replica
([`src/hs.py`](src/hs.py)) reports the **expected MAP@5 over random tie-breaks**, on the template users that have a truth.
Calibration on the first three submissions (12-week fold): replica 0.281 / 0.290 / 0.310 vs the mean of Kaggle public and private
0.275 / 0.289 / 0.308, same order ([`src/calibrate.py`](src/calibrate.py)). Models were selected with the replica, not with the leaderboard.

## Method in one paragraph

1. **Join** the view log to the broadcast schedule (same channel + interval overlap; unlabelled airtime `tv_show_id = 0` is not an item) and
   compute the screen time of every (user, airing) ([`src/build_interactions.py`](src/build_interactions.py), `src/common.py`).
2. **Scorers**: the four methods of the course - MF (ALS), association rules, CBF, Personalised PageRank on a committed co-view graph
   ([`src/methods.py`](src/methods.py)) - plus truth-aligned behavioural scorers (counts of committed airings, conversion, the slot habit
   "channel x time of week") ([`src/hs.py`](src/hs.py)), popularity as the required baseline.
3. **Candidates**: the union of the top lists of 13 scorers; the target-window schedule is used only to know what airs when.
4. **Re-ranker** ([`src/rerank2.py`](src/rerank2.py), [`src/attempts2.py`](src/attempts2.py), [`src/final3.py`](src/final3.py)): an ensemble of three
   LightGBM models (LambdaRank on the replica's labels, Poisson regression of committed airings per week, graded LambdaRank on tie-aware
   soft labels) over 104 features (the ones that grow with the length of the history were removed after they broke the transfer between
   folds); trained on five temporal folds (history 6-12 weeks, targets 8-12 weeks), applied to the test window with the whole 20-week
   history. Windows are clipped at the cut-off, so no viewing after the cut-off enters a feature (covered by the tests).

Individual methods on the same folds / pool / protocol (replica, mean of five folds, [`results/methods_table_5folds.txt`](results/methods_table_5folds.txt)):

| Method | MAP@5 |
|---|---|
| popularity (baseline) | 0.267 |
| MF (ALS), watched-seconds input / committed-airings input | 0.243 / 0.319 |
| association rules, watched basket / committed basket | 0.259 / 0.272 |
| CBF, watched profile / committed profile | 0.017 / 0.162 |
| **PPR** (committed co-view graph) | **0.369** |
| (extra) slot habit | 0.377 |
| rank-fusion hybrid ALS + AR + CBF + PPR | 0.341 |
| **final re-ranker (ensemble)** | **0.414** |

Ablation ([`results/ablation_v6.txt`](results/ablation_v6.txt)): a re-ranker over the four course methods alone reaches 0.399; removing them from the full
model changes MAP@5 by +0.001 (within noise) because the behavioural features carry the same information.

## Repository layout

* `src/` - code. Core pipeline: `common.py`, `build_interactions.py`, `build_all_overlaps.py`, `habits.py`, `hs.py`, `methods.py`,
  `rerank.py`, `rerank2.py`, `build_folds2.py`, `attempts2.py`, `final3.py`. Experiments and checks: `exp_*.py`, `ablation*.py`, `calibrate.py`,
  `verify*.py`, `manual_checks.py`, `examples.py`; earlier iterations (v1-v3 models): `run_*.py`, `final.py`, `final2.py`, `rerank_*.py`.
* `tests/test_pipeline.py` - tests on tiny hand-checkable examples (including the two MAP@5 examples of the teacher's slides).
* `submissions/` - every submitted CSV and the log with scores. `results/` - outputs cited in the report.
* `reproduce.ps1` - rebuilds the final file from the raw data in a separate folder and compares its md5 with the submitted file.

## Reproduce

Python 3.14.5 (CPU only), packages in `requirements.txt`; the code was run on Windows with PowerShell.

```powershell
python -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\kaggle competitions download -c sweettv-tv-program-recommender -p data   # accept the rules on the competition page first
# unzip into data\raw\  (dataset11-30.csv, export_arh_11-20-final.csv, export_arh_21-30-final.csv, export_arh_31-42-final.csv, submission.csv)
.\.venv\Scripts\python tests\test_pipeline.py
powershell -ExecutionPolicy Bypass -File reproduce.ps1      # 15-25 minutes, prints the md5 of the rebuilt and of the submitted file
```

The competition data is not included (competition rules). A Kaggle API token is not included either.

## Limitations

* The replica is a proxy: the host's tie-breaking is unknown, and the test has 20 weeks of history instead of the 6-12 weeks of the folds
  (Kaggle scores of the final model are higher than the replica's).
* The four course methods add nothing measurable inside the re-ranker beyond the behavioural features (see the ablation); on their own they
  are far below the full model.
* Only 10.8% of the schedule rows carry a programme id; only those can be recommended.

## AI usage

The work was done with Claude Code (model Claude Sonnet 5.5) in one session; the unmodified session export is submitted separately, as the
exam requires. Numbers in the report are taken from the files in `results/` and from the scripts above.
