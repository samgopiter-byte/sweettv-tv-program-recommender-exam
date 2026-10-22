# Manual checks (numbers for the student; every blank is filled in by the student)

## Check 1. Screen time of two airings (recompute by hand)

Rule of the host (Kaggle Data page): screen time = time the user watched the airing / length of the airing; only airings with screen time >= 0.8 count as 'committed'. Two examples, one clearly above and one clearly below the threshold.

### Example A

**Airing** (from the raw schedule file): channel 5 "ICTV", programme 240081 "Факты. Вечер.", starts **23.03.2020 18:45:00**, length **2100 s** (so it ends 23.03.2020 19:20:00).

**User** 11591782478356817074. Views of this user on this channel that overlap the airing (raw rows of `dataset11-30.csv`):

| # | start_time | stop_time | overlap with the airing, seconds (fill in) |
|---|---|---|---|
| 1 | 2020-03-23 18:43:14 | 2020-03-23 19:01:26 | ____ |
| 2 | 2020-03-23 19:06:20 | 2020-03-23 19:32:51 | ____ |

Your calculation: overlap = min(stop, airing end) - max(start, airing start) for each view; sum = ____ s; screen time = sum / 2100 = ____; committed (>= 0.8)? ____

> ---- the code's answer (look only after you have computed your own) ----
seconds 1806, length 2100, screen time 0.8600, committed: True

### Example B

**Airing** (from the raw schedule file): channel 5 "ICTV", programme 240081 "Факты. Утро.", starts **06.05.2020 08:45:00**, length **1800 s** (so it ends 06.05.2020 09:15:00).

**User** 3051565623483086097. Views of this user on this channel that overlap the airing (raw rows of `dataset11-30.csv`):

| # | start_time | stop_time | overlap with the airing, seconds (fill in) |
|---|---|---|---|
| 1 | 2020-05-06 08:46:28 | 2020-05-06 09:05:29 | ____ |
| 2 | 2020-05-06 09:10:56 | 2020-05-06 09:14:07 | ____ |

Your calculation: overlap = min(stop, airing end) - max(start, airing start) for each view; sum = ____ s; screen time = sum / 1800 = ____; committed (>= 0.8)? ____

> ---- the code's answer (look only after you have computed your own) ----
seconds 1332, length 1800, screen time 0.7400, committed: False

## Check 2. The truth of one user: top five programmes by the number of committed airings (user 4051396765385629500, window 04.05.2020 .. 27.07.2020)

Below are ALL airings of this user in the window with screen time >= 0.8 (programme, start of the airing, screen time). Count the airings per programme, order the programmes by count, and take the top five. Ties at the 5th place are possible.

| programme id | title | airing start | screen time |
|---|---|---|---|
| 240081 | Факты. День. | 15.07.2020 15:45:00 | 0.81 |
| 240081 | Факты. День. | 18.07.2020 12:45:00 | 1.00 |
| 240081 | Факты. День. | 19.07.2020 12:45:00 | 1.00 |
| 500281 | Ток-шоу 'Говорит Украина'. | 04.05.2020 20:10:00 | 0.84 |
| 500281 | Ток-шоу 'Говорит Украина'. | 05.05.2020 20:10:00 | 0.82 |
| 500281 | Ток-шоу 'Говорит Украина'. | 09.07.2020 20:10:00 | 0.81 |
| 500281 | Ток-шоу 'Говорит Украина'. | 22.07.2020 20:10:00 | 0.89 |
| 700389 | Кто сверху?, 6 сезон, 7 эп. | 23.07.2020 19:00:00 | 0.82 |
| 2400467 | 'Викна-Новости'. | 08.07.2020 14:30:00 | 1.00 |
| 2400480 | 'ТСН. Неделя'. | 22.07.2020 09:00:00 | 1.00 |
| 2400508 | Сегодня. Итоги с Олегом Панютой. | 13.07.2020 15:00:00 | 1.00 |
| 2400508 | Сегодня. Итоги с Олегом Панютой. | 24.07.2020 23:00:00 | 1.00 |
| 6200392 | Специя. | 07.05.2020 14:00:00 | 0.84 |
| 10001261 | Т/с 'Мыслить как преступник', 10 сезон,  | 26.07.2020 01:20:00 | 1.00 |
| 10001261 | Т/с 'Мыслить как преступник', 10 сезон,  | 26.07.2020 02:10:00 | 1.00 |
| 10001261 | Т/с 'Мыслить как преступник', 10 сезон,  | 26.07.2020 02:55:00 | 1.00 |
| 10001958 | Т/с 'Похищение богини', 1 с. | 07.07.2020 22:15:00 | 1.00 |
| 10001958 | Т/с 'Похищение богини', 1 с. | 07.07.2020 23:00:00 | 1.00 |
| 10003011 | Т/с 'Женский доктор 3', 1 и 2 с. | 08.07.2020 17:00:00 | 0.89 |
| 10003011 | Т/с 'Женский доктор 3', 1 и 2 с. | 16.07.2020 14:30:00 | 0.99 |
| 10003011 | Т/с 'Женский доктор 3', 1 и 2 с. | 21.07.2020 14:30:00 | 1.00 |
| 10003167 | Т/с 'Крепостная', 1 и 2 с. | 08.07.2020 14:20:00 | 1.00 |
| 90050203 | Х/ф 'Изгой', 1 ч. | 15.07.2020 15:25:00 | 1.00 |
| 90059866 | Х/ф 'Правила съема: Метод Хитча'. | 19.07.2020 12:20:00 | 1.00 |
| 90068637 | Х/ф 'Время грехов'. | 13.07.2020 17:25:00 | 0.89 |
| 90077292 | Х/ф 'Красавица и чудовище'. | 12.05.2020 14:45:00 | 0.97 |
| 90080566 | Х/ф 'Рецепт любви', 1 с. | 17.07.2020 21:00:00 | 1.00 |

Your counts per programme: ____ ; the five most frequent: ____ ; is there a tie at the 5th place? ____

> ---- the code's answer (look only after you have computed your own) ----
counts per programme: 500281: 4, 240081: 3, 10001261: 3, 10003011: 3, 10001958: 2, 2400508: 2, 2400467: 1, 700389: 1, 6200392: 1, 2400480: 1, 10003167: 1, 90050203: 1, 90059866: 1, 90068637: 1, 90077292: 1, 90080566: 1
strictly above the cut (always in the truth): [500281, 10003011, 240081, 10001261]; tied at the cut: [10001958, 2400508]; places left for the tied: 1

## Check 3. Expected committed airings of one programme from the slot habit (user 4051396765385629500, programme 90067201 "Х/ф 'Маша и море'.")

Model: for each airing of the programme in the target window take the user's rate in the airing's channel and time slot (slot = weekday class Mon-Fri / Sat / Sun and the half hour of the start, rounded to the nearest half hour): rate = hits / (airings + 1), where *hits* = airings in that channel and slot in the history window that the user watched for >= 80% and *airings* = all airings starting in that channel and slot in the history window. The expected number of committed airings of the programme is the sum of the rates over its airings.

| airing start | channel | slot | hits (history) | airings (history) | rate = hits / (airings + 1) (fill in) |
|---|---|---|---|---|---|
| 12.06.2020 21:20:00 (Fri) | 1276 | Mon-Fri 21:30 | 0 | 13 | ____ |
| 13.06.2020 15:45:00 (Sat) | 1276 | Sat 16:00 | 0 | 3 | ____ |
| 14.06.2020 17:55:00 (Sun) | 1276 | Sun 18:00 | 1 | 3 | ____ |
| 15.06.2020 09:20:00 (Mon) | 1276 | Mon-Fri 09:30 | 1 | 29 | ____ |
| 15.06.2020 23:00:00 (Mon) | 1276 | Mon-Fri 23:00 | 0 | 14 | ____ |
| 16.06.2020 10:50:00 (Tue) | 1276 | Mon-Fri 11:00 | 0 | 34 | ____ |
| 22.06.2020 09:20:00 (Mon) | 1276 | Mon-Fri 09:30 | 1 | 29 | ____ |
| 29.06.2020 12:35:00 (Mon) | 1276 | Mon-Fri 12:30 | 0 | 24 | ____ |

Your sum of the rates = ____ (expected number of committed airings of this programme for this user)

> ---- the code's answer (look only after you have computed your own) ----
sum of rates computed here: 0.3167; value of the model's scorer `hab_hs_p` for this user and programme: 0.3167

## Check 4. AP@5 by hand (final ensemble, fold val12)

AP@5 = (sum of the precisions at the ranks of the hits) / 5, because every user has exactly five programmes in the truth (the teacher's slides: normaliser min(K, |R|) = 5). Example from the slides: hits at ranks 1, 2, 4 -> (1/1 + 2/2 + 3/4) / 3 for a user with 3 relevant items.

### User 181063287547536022 (no randomness in the truth)

Recommended list (rank, programme):

1. 2400480 'ТСН. Неделя'.
2. 240081 Факты. День.
3. 2400508 Сегодня. Итоги с Олегом Панютой.
4. 2400467 'Викна-Новости'.
5. 20088 'Завтрак с 1+1'. Информационно-развл

Truth (5 programmes): 240081 Факты. День.; 2400467 'Викна-Новости'.; 2400480 'ТСН. Неделя'.; 2400508 Сегодня. Итоги с Олегом Панютой.; 12001682 М/с 'Губка Боб Квадратные Штаны'.

Your calculation: ranks of the hits = ____ ; precision at each = ____ ; AP@5 = ____

> ---- the code's answer (look only after you have computed your own) ----
hits at ranks [1, 2, 3, 4]; AP@5 = 0.8000

### User 3086774348345899441 (a tie decides one place of the truth)

Recommended list (rank, programme):

1. 12001682 М/с 'Губка Боб Квадратные Штаны'.
2. 2400508 Сегодня. Итоги с Олегом Панютой.
3. 240081 Факты. День.
4. 12002856 М/с 'Элвин и бурундуки'.
5. 12001734 М/с 'Волшебные покровители', 154 с. 

Sure (always in the truth): 12001682 М/с 'Губка Боб Квадратные Штаны'.; 240081 Факты. День.; 12002856 М/с 'Элвин и бурундуки'.; 2400508 Сегодня. Итоги с Олегом Панютой.
Tied for the last place (exactly one of them is in the truth, each with probability 1/2): 12001734 М/с 'Волшебные покровители', 154 с. ; 12002378 М/с 'Удивительный мир Гамбола'.

Your calculation: AP@5 if the truth is sure + first tied = ____ ; AP@5 if sure + second tied = ____ ; expected AP@5 = their mean = ____

> ---- the code's answer (look only after you have computed your own) ----
AP@5 for the two truths: 1.0000 and 0.8000; expected AP@5 = 0.9000
