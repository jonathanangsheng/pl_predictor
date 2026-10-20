# Plan: single-match predictions with fan sentiment (inside pl-predictor)

Status: revised after the /autoplan CEO review and approval gate (2026-10-18).
Decisions at the gate: build inside pl-predictor (D1), accept all taste calls (D2), revise (D3).
Review history: [predict-plan-review.md](predict-plan-review.md) (CEO review, 3 spec reviews, independent CEO voice).
Not yet reviewed: DX and Eng phases (skipped at the user's request). Run the Eng review before building.

## Implementation notes (built 2026-10-18)

Built as flat modules in `pl_predictor/` (matches, elo, features, model, predict, study, watch, model_commands) with `tests/test_predict.py`. Deviations from the plan below, with evidence:
- **Stage 1 is one logistic regression on 7 features** (Elo difference + 10-match form differences in goals,
  shots, shots on target). Walk-forward over 2023/24-2025/26: this model 0.9790 vs Elo 0.9814 (better every
  season); the planned HistGradientBoosting + calibration wrapper (1.014), the 69-feature set (0.993) and
  corners/fouls/table/rest-day features were all worse than Elo.
- **One fixture source** (2026-10-20): `run`, `fixtures`, `query` and `demo` read fixtures from the FPL API
  like `watch` and `predict`; `config/fixtures.csv` and `PLS_FIXTURES_PATH` were removed.
- **Stage 1 became a 50/50 blend with Dixon-Coles** (2026-10-20): 0.9746 over 2023/24-2025/26, better than
  the logistic regression in 5 of 6 seasons from 2020/21 (mean -0.0047, 95% CI -0.0090 to -0.0003).
- **Elo is computed from results** (`elo.py`). ClubElo returned 502 from this network throughout, so every result
  here used local Elo; the ClubElo hook and fallback were removed on 2026-10-20.
- `FootballDataCoUkHook.upcoming()` was deferred, then added on 2026-10-20: each prediction logs de-margined
  market odds (`mkt_home/draw/away`) when football-data.co.uk lists the fixture.
- Phase 0 collection is **not started**, at the user's request; `pl-predictor watch` replaces the scheduler step.
- Not yet reviewed: DX and Eng phases.

## Goal

Answer one question with a public, timestamped track record:
**does pre-match fan sentiment on X carry information beyond the betting market?**

To get there, pl-predictor gains a `predict` subpackage that outputs home/draw/away probabilities for each upcoming
Premier League fixture, scores them after the match, and runs an offline study of whether sentiment helps.

Success criteria:
1. `pl-predictor model predict` prints and logs H/D/A probabilities for every fixture in the next gameweek.
2. `pl-predictor model evaluate` reports, per season for the last 3 completed seasons (walk-forward), log loss and
   Brier first and accuracy second, for the model, an Elo-only baseline, de-margined pre-match odds (fair market)
   and de-margined closing odds (ceiling).
3. Model log loss beats the Elo-only baseline. The gap to the market is reported, not a target.
4. `pl-predictor model score` reports the live record: number of scored predictions and log loss vs Elo and market.
5. `pl-predictor model study` answers the headline question with a powered, pre-registered test, and records the
   answer even if it is "no".

Credit: the rolling-form feature design comes from Football_Prediction_Project by mhaythornthwaite (MIT); README
says so. That fork is not modified.

## Phase 0 — collect first (do this before any model code)

Sentiment cannot be backfilled (X recent search reaches back 7 days), so collection starts now.
- Set `PLS_PRE_MATCH_MINUTES=1440` and `PLS_MAX_TWEETS_PER_FIXTURE=400` in `.env`.
- Set `PLS_SENTIMENT_SINCE=<today>`; only fixtures from this date count for the study.
- Until `pl-predictor watch` exists, add a Windows Task Scheduler entry per kickoff slot that runs
  `pl-predictor run --fixture <id>` between kickoff − 60 min and kickoff (the first run for a fixture must be in
  that window; see "Collection timing").
- Verify: the DB holds tweets older than kickoff − 20 h for the first tracked gameweek.

## Layout

```
src/pl_predictor/predict/
  __init__.py
  names.py        club name mapping across sources (config/club_names.csv)
  data.py         match table from the hooks, caches, odds normalisation
  features.py     as-of rolling form, Elo, table context, rest days, sentiment columns
  model.py        stage 1, Elo baseline, walk-forward evaluation, calibration table, save/load
  predict.py      next-gameweek predictions, history.csv, scoring
  study.py        stage-2 study: power analysis, two-arm gate
  watch.py        scheduler loop: collect, score, predict per kickoff slot
config/club_names.csv
tests/predict/
```

CLI (in the existing `pl-predictor` command; `score` is already taken by tweet scoring, so model commands sit
under `model`):
`pl-predictor model evaluate | train | predict | score | study` and `pl-predictor watch`.

New dependencies: scikit-learn, joblib (in `[project.optional-dependencies] predict`). New settings in
`config.Settings`: `sentiment_since`, `stage2_enabled`, `predictions_dir`, `models_dir`.

## Data sources (existing hooks)

| Hook | Role |
|---|---|
| `FootballDataCoUkHook.results` | History from 2005/06: results, shots, shots on target, corners, fouls, cards, pre-match and closing odds. xG only exists from 2026/27: stored, never a model input. |
| `FootballDataCoUkHook.upcoming` (new) | Pre-match odds for upcoming fixtures from football-data.co.uk `fixtures.csv` (market arm). |
| `FPLHook.fixtures` | Upcoming fixtures and kickoff times (UTC). |
| `ClubEloHook.club_history` | Elo per club, joined as of the day before each match. |
| `FootballDataOrgHook.matches` | Optional: fills current-season results between football-data.co.uk updates. |

Caching: completed football-data.co.uk seasons and ClubElo histories are cached under `data/cache/`; the current
season is always refetched.

## Identity and keys

- `config/club_names.csv`: one row per club since 2005/06 with the canonical code (FPL short name, same as
  `teams.yaml`) and its name in football-data.co.uk, ClubElo and football-data.org. An unmapped name raises an
  error listing the names; nothing is silently dropped.
- Season = year if the Europe/London month is 8 or later, else year − 1 (a July 2020 kickoff is 2019/20).
  football-data.co.uk rows take the season of their file.
- Fixture key = (season, home code, away code). Each pairing happens once per season, so rescheduled matches keep
  their key. FPL fixtures, results, logged predictions and the sentiment `fixtures` table all join on it.
  Unmatched FPL fixtures or logged predictions raise; unmatched sentiment fixtures (demo data) only warn.

## Match table (`data.py`)

One row per match: date, kickoff (UTC, NaN where the season has no Time column), season, key, goals, result,
per-side stats, `avg_h/d/a` and `close_h/d/a` (normalised from Avg*/AvgC* from 2019/20 and BbAv*/PSC* before;
NaN when absent). Current-season matches missing from football-data.co.uk are filled from football-data.org
(goals and result only). On duplicate keys the football-data.co.uk row wins.

Failure rules:
- Current-season file 404 before the season's first FPL kickoff → empty current season, warning.
  Any other current-season fetch failure aborts with the URL and HTTP error.
- Missing expected columns → `SchemaError` naming the file and columns.
- football-data.org: missing token, 429, 5xx or any request error → warn (with the newest match date) and skip the fill.
- ClubElo request error → use the cache and warn with its age; no cache → exit non-zero naming the club.

## Features (`features.py`)

Every feature for a match on date D uses only matches strictly before D.
- Rolling 5 and 10 match means per side: goals for/against, shots, shots on target, corners, fouls; home-minus-away
  differences. Windows span seasons. Fill-only matches count for goals, table, rest days and the 5-match
  threshold; stat means use the last 5/10 matches where the stat exists.
- Features are NaN (rows kept) when a side has fewer than 5 league matches in the prior 400 days, so promoted clubs
  are trained on and predicted. HistGradientBoosting handles NaN; LogisticRegression uses
  `SimpleImputer(add_indicator=True)`.
- Elo home, away and difference as of D − 1. Table: points per game and goal difference this season. Rest days
  (NaN without a previous match).
- Sentiment (stage 2 only): T = kickoff − 60 min. Scored tweets with kickoff − 24 h ≤ created_at < T.
  Home uses side in {home, both}; away uses {away, both}; `neither` and unscored tweets are excluded.
  Model inputs are exactly `home_net`, `away_net`, `net_diff`. Tweet counts are shown and used for eligibility,
  never as inputs (the per-fixture cap makes volume meaningless). The window uses the FPL kickoff live and the
  match-table kickoff in training; a mismatch with the sentiment DB's kickoff means "no sentiment" plus a warning.

## Stage 1 model (`model.py`)

- Candidates: `HistGradientBoostingClassifier` and multinomial `LogisticRegression`, each in
  `CalibratedClassifierCV(cv=TimeSeriesSplit(5))` on time-ordered rows; pick by walk-forward log loss.
- Walk-forward: for each of the last 3 completed seasons, train on earlier seasons, test on that season.
- Baselines on the same splits: Elo-only logistic regression; de-margined pre-match odds (fair market) and
  closing odds (ceiling).
- Report: log loss and Brier, accuracy labelled secondary, 10-bin calibration table.
- `models/model.joblib`: stage 1, Elo baseline, optional stage 2, feature list, training date,
  `model_version = "<training date>-<git short sha>"`. README warns it is a pickle: load only self-produced files.

## Predictions and scoring (`predict.py`)

- Next gameweek = lowest gameweek with any unplayed fixture; only unplayed fixtures are predicted; double
  gameweeks predict both fixtures; null-kickoff fixtures are listed as unscheduled.
- Output table and `predictions/history.csv` rows: key, kickoff, predicted_at, model P(H/D/A), Elo-baseline
  P(H/D/A) (logged now so scoring never uses a model retrained after the result), stage, pre-cutoff tweet counts,
  model version. Every fixture gets probabilities summing to 1.
- `model score`: per fixture, the latest row with predicted_at < kickoff; rows after kickoff ignored; unplayed
  fixtures reported as "pending: n". Three-way comparison (model, Elo, pre-match odds) on fixtures that have all
  three, with n; closing-odds column where available; model-only metrics on all scored fixtures with n.
- Rescue paths: missing model → "run `pl-predictor model train` first", exit 2; sklearn version warning on load →
  retrain hint; `history.csv` locked (open in Excel) → retry 3 x 1 s then error naming the file; no upcoming
  fixtures → message, exit 0; empty history → message, exit 0; empty match table → evaluate/train refuse.

## Stage 2 study (`study.py`) — offline first

- Eligible fixture: kickoff ≥ `PLS_SENTIMENT_SINCE`, at least one side has ≥ 1 scored tweet before T, and the
  fixture's first collection happened at or after T − 5 min (see "Collection timing").
- Power analysis: simulate with the observed per-match log-loss variance at effect sizes 0.002, 0.005, 0.01 and
  print the fixtures needed for 80% power. This n replaces any fixed threshold.
- Two arms on the same eligible fixtures: (stage 1 + sentiment) vs stage 1, and (market + sentiment) vs market,
  where market = de-margined pre-match odds (history from football-data.co.uk, live from `upcoming()`).
- Inputs: log(P(H)/P(A)) and log(P(D)/P(A)) from an out-of-fold stage 1 (probabilities clipped to [1e-6, 1]),
  plus the three sentiment columns, through `SimpleImputer(add_indicator=True)` + `LogisticRegression`.
- Folds: `TimeSeriesSplit(5)` over eligible fixtures by kickoff; stage 1 refit on matches before each fold; stage 2
  trained on earlier eligible fixtures; the gate stays off if the first fold has fewer than 25 training fixtures.
- Gate: the paired bootstrap 95% CI (1,000 resamples, fixed seed) of the log-loss difference lies entirely below 0.
  Prints n, the CI and the decision for both arms.
- `model predict` uses stage 2 only when the study has passed and the user enables it (`--stage2` or
  `PLS_STAGE2=1`), and only for fixtures already past T with sentiment on at least one side; otherwise stage 1
  with the reason ("stage 1: no sentiment", "stage 1: before cutoff").

## Collection timing and `watch` (`watch.py`)

- pl-predictor's first run for a fixture splits its budget across 8 slices of the window, so the first run must be
  in [T, kickoff) to cover the 24 h pre-match window. Fixtures first collected earlier are excluded from the study
  with a warning.
- `pl-predictor watch`: refreshes FPL fixtures every 15 min; at kickoff − 55 min per fixture it upserts the fixture
  into the DB, collects and scores its tweets, then predicts it. Moved kickoffs are rescheduled; a slot passed
  without collection is logged as MISSED. One failing fixture does not stop the rest; SQLite opens with a 30 s
  timeout and a locked DB is retried next loop. A lock file (`data/watch.lock`) prevents a second instance; README
  says not to run `pl-predictor run --watch` at the same time. Every state change goes to `logs/watch.log`
  (rotating). README shows how to start it from Task Scheduler or cron.

## Cost and stop rule

README gives the X API cost to reach the power-analysis n: fixtures × 400 tweets × ~$0.005 per read (third-party
figure, unconfirmed; about 1 of the 8 first-run slices falls after kickoff and returns nothing), roughly $80/month
at 10 fixtures a week. Stop collecting at the earliest of reaching n, a user-set spend cap, or the end of the
2026/27 season; then run the study and record the result either way.

## Tests

All time-dependent code takes an injected `now`; hooks are mocked; no network in tests.
- As-of invariance: adding later matches never changes earlier features.
- Name mapping: every club since 2005/06 maps per source; an unknown name raises.
- Season rule (2020-07-26 → 2019; 2026-08-15 → 2026); rescheduled fixture keeps its key.
- Odds normalisation for old and new column sets; missing Time column builds features.
- Sentiment window edges (tweet at T excluded, at T − 1 s included, at kickoff − 24 h − 1 s excluded); side rule;
  unscored tweets excluded; kickoff mismatch → no sentiment.
- Promoted club with zero prior matches gets probabilities summing to 1; fill-only match handling.
- Walk-forward never trains on its test season; study folds strictly increase in time and never train on their
  own fixtures; changing a fixture's result does not change its out-of-fold stage-1 input.
- Gate: synthetic informative sentiment passes, noise does not; power n grows as the effect shrinks.
- Scoring: latest pre-kickoff row used, post-kickoff rows ignored, pending counted, three-way subset with n.
- Every failure rule above has a test (404 before/after season start, schema drift, fd.org errors, ClubElo cache,
  missing model, locked history, empty inputs).
- `watch` on a fake clock: order of collect, score, predict; reschedule; MISSED; a failing fixture does not block
  the next; second instance exits.
- One end-to-end test per CLI command on recorded fixture data, and a golden-file test for `model predict`.

## Hygiene and docs

- `.gitignore`: `models/`, `predictions/` (except `.gitkeep`), `logs/`, `data/cache/`.
- README: the headline question, Phase 0 setup, commands, cost and stop rule, evaluation table (log loss first),
  credit to Football_Prediction_Project.
- TODOS.md: pre-match odds as a stage-1 feature, predictions in the dashboard, local Elo fallback, FPL team xG,
  per-fixture explanations, Understat xG history, Dixon-Coles/Poisson model.

## Implementation order

1. Phase 0 collection settings and scheduler entry (no code).
2. `names.py` + `club_names.csv`, `data.py` with caches and failure rules.
3. `features.py`.
4. `model.py` + `model evaluate` / `model train`.
5. `predict.py` + `model predict` / `model score`.
6. `watch.py`.
7. `study.py` + `FootballDataCoUkHook.upcoming()`.
8. README, TODOS.md, .gitignore.

## Out of scope

Web app, betting or staking, player-level models, lineups and injuries, historical sentiment backfill.
