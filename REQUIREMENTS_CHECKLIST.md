# Exam requirements checklist

Sources: `exam_quiz_AVITO.md` (md), the quiz slides (slides), the teacher's seminar of 2026-10-22 (seminar, transcript), the homework
guidelines (guidelines). Status as of 2026-10-23. "report" = the PDF that is still to be written.

| # | Requirement | Source | Status | Evidence / what is left |
|---|---|---|---|---|
| 1 | Individual work; deadline Sunday 25.10.2026 19:00 MSK; upload exactly two files (PDF report + unmodified AI-session export) to the LMS | md, slides, seminar | open | student uploads; the session export is copied at the very end |
| 2 | At least two of CBF / association rules / MF / PPR, each evaluated individually on the same split, pool and protocol | md, slides, seminar | done | all four: `results/methods_table_5folds.txt` |
| 3 | Popularity baseline as the reference | md, slides | done | same table |
| 4 | The "core idea" of an advanced / combined approach must relate to the studied methods; combining and literature-based approaches are allowed | seminar | done in design, text open | hybrid of the four methods + behavioural signals; ablation `results/ablation_v6.txt`; the report must say so and cite literature (LambdaMART, repeat consumption, ...) |
| 5 | Matching rule and threshold, handling of `tv_show_id = 0`, missing ids | md, slides | code done, report open | `src/build_interactions.py`, README |
| 6 | Own temporal validation inside weeks <= 30; models built from training data only; no future viewing | md, seminar | done | folds in `src/common.py`; window-clipping tests |
| 7 | Candidate pool, repeats, cold start stated explicitly and identical for all methods | md | code done, report open | README; `src/methods.py`, `src/rerank.py` |
| 8 | Exactly five unique programme ids per user | md | done | checks in `results/verification_numbers_v2.md` |
| 9 | Document hyper-parameters, seeds, runtime, environment | md | partly | README (environment, seeds in code); **runtime table still to be measured** |
| 10 | Hybrid: components, how weights were chosen, comparison with each component | md | done | `results/attempts_v6.txt`, `results/ablation_v6.txt` |
| 11 | Two users chosen by a rule and used for all methods: Top-5 with id, title, category, compared with the history | md, seminar | script ready, output open | `src/examples.py` -> `examples_two_users.md` (to be generated for the final model) |
| 12 | Business analysis: which strategy for the platform and why; differences between methods; limitations; no A/B or revenue claims | md, seminar | open | report |
| 13 | PDF contains the GitHub link (**and the repository is listed as a reference**), the Kaggle screenshot, method comparison, validation, business analysis | md, seminar | open | report; link https://github.com/samgopiter-byte/sweettv-tv-program-recommender-exam |
| 14 | Kaggle: late submission, description with the Russian full name and the Kaggle name; mark the best submission as Selected | slides, seminar | submissions done, Selected open | `submissions/SUBMISSIONS.md`; Selected + screenshot at the end |
| 15 | Screenshot of the Submissions page showing the best result; the teacher asks for the private score, md says public: show both columns; the name must be displayed correctly | slides, seminar, md | open | taken at the end |
| 16 | Repository: code, the final submission CSV (the one that gave the reported score), reproduction instructions | md, seminar | done | README, `reproduce.ps1`, `submissions/` |
| 17 | Report in the homework format (IEEE-style sections, <= 5 pages excluding references, AI usage disclosure with "what I personally checked" and "what I trusted without verification") | guidelines, md | open | report |
| 18 | "What I personally checked": a table of the report plus a calculation example taken from it so that a reader can reproduce it; not everything has to be checked by hand | seminar | in progress | `MANUAL_CHECKS.md`, `VERIFICATION_LEDGER.md` (S2 confirmed, S3 partly) |
| 19 | Gate 0: no invalid or invented references / DOIs, no fabricated numbers, session export matches the work | seminar, md, guidelines | open | every reference opened and compared; numbers taken from `results/`; the session is one file |
| 20 | Language of the implementation is free; no A/B test required | seminar | done | Python |
| 21 | Ranking inside the group by the returned score: 50 points in quintiles; the other 50 from the report | md, seminar | open | final submission designated at the end |
| 22 | Use of the test-period schedule only to know what airs when, never to look up who watched | md (the seminar says information about the target session must not be used for training) | done | pool, airing counts and slot habit use only the schedule; no viewing of weeks 31-42 exists in the log |
