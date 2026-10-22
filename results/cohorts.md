# Cohorts of the 1,260 template users (disjoint decomposition)

| fold | history window | target window | with history and truth (evaluated) | with history, no truth | no history, with truth | no history, no truth | total |
|---|---|---|---|---|---|---|---|
| val | 2020-03-09 .. 2020-06-01 | 2020-06-01 .. 2020-07-27 | 1126 | 119 | 7 | 8 | 1260 |
| val2 | 2020-03-09 .. 2020-05-04 | 2020-05-04 .. 2020-06-29 | 1017 | 222 | 6 | 15 | 1260 |
| val12 | 2020-03-09 .. 2020-05-04 | 2020-05-04 .. 2020-07-27 | 1186 | 53 | 13 | 8 | 1260 |
| val3 | 2020-03-09 .. 2020-05-18 | 2020-05-18 .. 2020-07-13 | 1040 | 202 | 5 | 13 | 1260 |
| val5 | 2020-03-09 .. 2020-04-20 | 2020-04-20 .. 2020-06-15 | 1106 | 127 | 8 | 19 | 1260 |
| test | 2020-03-09 .. 2020-07-27 | 2020-07-27 .. 2020-10-19 | 1256 | 0 | 4 | 0 | 1260 |

For `test` the 'truth' column is true by construction (every user of the solution file has exactly five programmes), so the cell 'with history, no truth' is empty there; users without a committed history get the popularity fallback.