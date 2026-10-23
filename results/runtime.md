# Runtime on the 12-week fold (one run, CPU only)

Machine: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel, Windows 11, Python 3.14.5; threads: BLAS default, LightGBM 4.

| step | seconds |
|---|---|
| load fold (interactions, schedule, history table) | 4.2 |
| index maps and user x item matrices | 0.3 |
| committed-airing matrix of all users | 0.9 |
| popularity (baseline) | 0.0 |
| MF: ALS, 32 factors, 20 iterations | 1.0 |
| AR: association rules, pairwise support/confidence | 1.7 |
| CBF: tf-idf attribute profiles | 3.3 |
| PPR: co-view graph + 15 power iterations (d = 0.85) | 8.7 |
| slot habit (committed), pooled profile | 1.6 |
| committed conversion | 1.4 |
| all features of one fold (13 scorers, candidates, 104 features) | 83.0 |
| join of the view log to the schedule (`build_interactions.py`, 5.8 M views, separate run) | 61 |
| final training + prediction of the submitted ensemble (`final3.py A5`, features cached, separate run; md5 of the output `0e5de6ec...`, identical to the submitted file) | 88 |
