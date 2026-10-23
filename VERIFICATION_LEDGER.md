# Verification ledger

Who checked what. Two kinds of entries are kept apart on purpose: checks run by the AI agent (Claude Code) and checks the student
performs personally. A student check is marked *confirmed* only after the student reports the values he computed himself; nothing
is attributed to the student before that. The report's section "What I personally checked" uses only the confirmed student rows.

## A. Checks run by the agent (automatic, reproducible by the commands given)

| # | What | Result | Evidence / command |
|---|---|---|---|
| A1 | 30 unit tests on tiny hand-checkable examples (MAP@5 slide examples 0.9167 / 0.5333, overlap, window clipping, slots, screen time, ties, expected AP) | 30 OK, 0 FAIL (2026-10-22) | `python tests/test_pipeline.py` |
| A2 | Full rebuild from the raw files reproduces the submitted file | md5 identical (`0E5DE6EC...`), 9.5 min | `results/reproduce_check.txt`, `reproduce.ps1` |
| A3 | Independent recounts: raw row counts, join of 400 random views by brute force, no future views in training, one programme recounted from raw files, test-window pool recounted with the csv module | all equal | `results/verification_numbers.md`, `results/verification_numbers_v2.md`, `src/verify.py`, `src/verify2.py` |
| A4 | Local replica of the Kaggle score vs the three earliest Kaggle scores | replica 0.281 / 0.290 / 0.310 vs mean(public, private) 0.275 / 0.289 / 0.308 | `src/calibrate.py`, README |
| A5 | PPR convergence per fold and for the test window, every cohort user | see the files | `results/ppr_convergence_*.md`, `src/ppr_convergence.py` |
| A6 | Cohort decomposition (disjoint cells adding up to 1,260 users) for every fold | see the file | `results/cohorts.md`, `src/cohorts.py` |
| A7 | Paired bootstrap, shuffle canary, leak canary, seed sensitivity of the re-ranker | see the file | `results/verification_numbers_v2.md` |
| A8 | Kaggle Submissions page: description shows the Russian name correctly; scores read from the API and the page text | matches `submissions/SUBMISSIONS.md` | log in the session |

## B. Checks the student performs personally (numbers prepared by `src/manual_checks.py`, commands in `MANUAL_CHECKS.md`)

| # | What the student does | Expected | Status | Student's own result |
|---|---|---|---|---|
| S1 | Counts the rows of the four raw files (PowerShell) | 5,821,987 / 1,074,235 / 1,074,394 / 1,295,378 | pending | |
| S2 | Computes the md5 of `submission_v6_ensemble.csv` and compares it with the log | `0E5DE6ECAB6CF6433EE7E6516A4C3440` | **confirmed 2026-10-22** | `Get-FileHash -Algorithm MD5` on `submissions\submission_v6_ensemble.csv`: `0E5DE6ECAB6CF6433EE7E6516A4C3440`, equal to the md5 recorded in `submissions/SUBMISSIONS.md` at submission time |
| S3 | Opens the file: 1,260 rows, five distinct ids in a row, looks one id up in the test schedule | see `MANUAL_CHECKS.md` | **confirmed 2026-10-23** (row count `1260` reported by the student on 2026-10-23) | First data row: user 8377619604347126107 -> `2400480 2400508 240081 20088 12002896`; 5 ids, 5 distinct. Lookup of programme 2400480 in `export_arh_31-42-final.csv`: rows `27.07.2020 07:00:00`, `08:00:00`, `09:00:00` (title ТСН., length 600 s), i.e. in the test window |
| S4 | Screen time of one airing from raw view rows (hand arithmetic) | `MANUAL_CHECKS_data.md`, check 1 | **confirmed 2026-10-23; the student stated that he did the arithmetic himself** | Example A: overlaps 986 + 820 = 1806 s, 1806/2100 = 0.8600, committed; example B: 1141 + 191 = 1332 s, 1332/1800 = 0.7400, not committed. Equal to the code |
| S5 | Truth of one user: counts committed airings per programme, picks the top five, finds the tie | check 2 | **confirmed 2026-10-23; the student stated that he did the arithmetic himself** | User 4051396765385629500: counts 500281: 4; 240081, 10001261, 10003011: 3; 10001958, 2400508: 2; tie for the 5th place between the last two, one place left. Equal to the code |
| S6 | Expected committed airings of one programme from the slot habit (hand arithmetic) | check 3 | **confirmed 2026-10-23; the student stated that he did the arithmetic himself** | Programme 90067201: rates 0, 0, 1/4, 1/30, 0, 0, 1/30, 0; sum 0.3167 = `hab_hs_p`. Equal to the code |
| S7 | AP@5 by hand for two users (one without and one with a tie) | check 4 | **confirmed 2026-10-23; the student stated that he did the arithmetic himself** | User 181063287547536022: hits at ranks 1-4, AP@5 = 4/5 = 0.8000; user 3086774348345899441: AP@5 = 1.0000 or 0.8000 depending on the tie, expected 0.9000. Equal to the code |
| S8 | Runs `reproduce.ps1` himself and compares the md5 | identical | pending | |
| S9 | Compares the Kaggle Submissions page with `submissions/SUBMISSIONS.md` | equal | pending | |
| S10 | Opens every reference of the report and compares author / year / venue / DOI | to be done after the draft | pending | |

Not established personally (and not claimed): every user's metric, all intermediate tables, the correctness of the libraries' internals
(ALS, LightGBM), all-user convergence of PPR (that is an agent check, A5).
