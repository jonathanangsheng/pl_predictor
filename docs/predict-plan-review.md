# Review history: match-predictions plan (originally written for the Football_Prediction_Project fork)

> Archived 2026-10-18. At the approval gate the user chose to build inside pl-predictor instead (D1),
> accepted all taste calls (D2) and asked for a revision (D3). The current plan is predict-plan.md.
> C-numbers below refer to this archive; predict-plan.md folds them into its sections.

# Plan: single-match Premier League predictions with fan sentiment

## Implementation plan

### Goal

Turn this fork of Football_Prediction_Project (FPP) into a tool that outputs home/draw/away
probabilities for each upcoming Premier League fixture, using four free data hooks plus pre-match
fan sentiment from the sibling `pl-predictor` project. Repo scope changes from "api-football +
classical ML experiments" to "maintained single-match predictor".

Success criteria:
1. `fpp predict` prints and saves H/D/A probabilities for every fixture in the next gameweek.
2. Walk-forward evaluation over the last 3 completed seasons reports log loss, Brier score and
   accuracy for the model, an Elo-only baseline and bookmaker implied probabilities.
3. Model log loss beats the Elo-only baseline. Bookmaker odds are the ceiling to compare against,
   not a target we must beat.
4. Sentiment is added as a feature only once enough fixtures with sentiment exist, and only kept if
   it improves log loss on held-out fixtures.

### What we keep from FPP

- The core idea: rolling averages of each team's recent match stats (5 and 10 games), expressed as
  team-minus-opponent differences, fed to a classifier that outputs probabilities.
- The evaluation focus on probability calibration (FPP README figure 4), not just accuracy.
- MIT licence and upstream credit.

### What we remove

- api-football ingestion (`01_api_data_request.py`, `02_cleaning_stats_data.py`) and its paid key.
- Committed data: `prem_game_stats_json_files/` (~2,000 JSON files), `prem_clean_fixtures_and_dataframes/`,
  pickled models (`*.pk1`), `figures/`, `predictions/pl_predictions.csv`.
- Experiment folders `ml_model_build_*`, `web_server/`, `execute_project.bat`,
  `venv_football_pred_project.yml`, numbered scripts `03_*`, `04_*`, and `ml_functions/`
  (logic is ported, not imported).
- Pinned 2020 requirements (pandas 1.0.5, scikit-learn 0.23.1).

### Data sources (all via `pl_predictor.hooks`)

| Hook | Role | Used for |
|---|---|---|
| `FootballDataCoUkHook` | Training history and current season | Results, shots, shots on target, corners, fouls, cards per match from 2005/06; xG columns exist only from 2026/27 (stored, not modelled); pre-match and closing odds (benchmark only) |
| `FPLHook` | Upcoming fixtures | `fixtures(future=True)` gives the next gameweek's fixtures, kickoff times and FPL team ids |
| `ClubEloHook` | Team strength | `club_history(club)` gives Elo per club over time; joined as-of the day before each match |
| `FootballDataOrgHook` | Freshness | Current-season finished results between football-data.co.uk updates (it refreshes about twice a week), so rolling features include last weekend's matches |

Past football-data.co.uk seasons never change, so they are cached on disk (`data/cache/`, gitignored).
Current-season files and all other hooks are fetched on every run.

### Team identity

`config/teams.csv`: one row per club with a canonical code (FPL short name, e.g. `MCI`) and its name
in each source (football-data.co.uk `Man City`, ClubElo `Man City`, football-data.org TLA `MCI`,
pl-predictor code `MCI`). Covers every club in the training window (2005/06 onwards). Loading any
source with an unmapped name raises an error listing the names, rather than silently dropping
matches.

### Package layout

```
fpp/
  __init__.py
  teams.py        load config/teams.csv, map names per source
  data.py         build one match table from the hooks (with cache)
  features.py     as-of rolling features, Elo, table context, sentiment
  model.py        train, calibrate, walk-forward evaluation, save/load
  predict.py      features for upcoming fixtures, probabilities
  cli.py          `fpp evaluate`, `fpp train`, `fpp predict`, `fpp score`
config/teams.csv
tests/
pyproject.toml    Python >=3.10; pandas, numpy, scikit-learn (pl-predictor installed separately, see below)
```

`pl-predictor` is a dependency (hooks + sentiment database). Install it with `pip install -e ../pl-predictor`.
pl-predictor has no git remote yet; publishing it is a user-owned prerequisite before pyproject can point at a URL.

### Match table (`data.py`)

One row per match: date, kickoff, season, home code, away code, full-time goals, result (H/D/A),
and per-side stats (shots, shots on target, corners, fouls, yellows, reds, xG if present), plus
average pre-match and closing odds. Built from football-data.co.uk, then current-season matches
missing from it are filled from football-data.org (goals and result only; stats left NaN).

### Features (`features.py`)

Every feature for a match on date D uses only matches strictly before D.

- Rolling 5 and 10 match means for each side of: goals for/against, shots, shots on target,
  corners, fouls. Windows span seasons, as in FPP. No xG (history has none before 2026/27).
- Home-minus-away differences of the above (FPP's "difference" features, oriented home/away
  instead of FPP's team/opponent double rows, so each match appears once).
- Elo: home Elo, away Elo, difference, as of D-1.
- Table context: points per game and goal difference this season, from matches strictly before D.
- Rest days since each side's previous league match.
- Sentiment (stage 2 only): net sentiment per side (mean of P(pos) - P(neg)) and their difference, from
  pl-predictor tweets with kickoff - 24h <= `created_at` < T, where T = kickoff - 60 min is the prediction
  cutoff. The same window is used for training rows and live predictions.

Rolling features are NaN (not dropped) when a side has fewer than 5 league matches in the prior 400 days,
so promoted clubs are still trained on and predicted.

### Model (`model.py`)

- Stage 1 (base): scikit-learn `HistGradientBoostingClassifier` and a multinomial
  `LogisticRegression`, each wrapped in `CalibratedClassifierCV(cv=TimeSeriesSplit(5))` fitted on
  time-ordered rows. Pick by walk-forward log loss.
- Walk-forward evaluation: for each of the last 3 completed seasons S, train on seasons before S,
  test on S. Never shuffle across time.
- Baselines on the same splits: Elo-only logistic regression; bookmaker implied probabilities from
  closing average odds (overround removed by normalising).
- Report per season and overall: log loss, Brier, accuracy, calibration table (10 bins).
- Stage 2 (sentiment): once at least 150 finished fixtures have pre-match sentiment, fit a logistic
  regression on [stage-1 log-odds, sentiment features], evaluate with time-ordered cross-validation
  on those fixtures, and enable it in `predict` only if the paired bootstrap 95% CI (1,000 resamples) of the
  log-loss difference vs stage 1 lies entirely below 0. Until then
  `predict` shows sentiment next to the base probabilities, unmodelled.
- Saved model: `models/model.joblib` (gitignored) with the feature list and training date.

### Prediction (`predict.py`, `fpp predict`)

1. FPL fixtures for the next unplayed gameweek (or `--gameweek N`).
2. Build features as of each kickoff from the match table, Elo and (if available) sentiment.
3. Output a table and `predictions/gw{N}.csv`: kickoff, home, away, P(H), P(D), P(A), Elo diff,
   pre-match sentiment per side and tweet count, model version.

### Tests

- As-of correctness: features for a match never change when later matches are added to the table.
- Team mapping: every club in each source's history maps; an unknown name raises.
- Walk-forward splits never put a test-season match in training.
- Sentiment features only count tweets with kickoff - 24h <= `created_at` < kickoff - 60 min.
- Probabilities sum to 1 per fixture; calibration table bins sum to the test size.
- Hooks are mocked in tests (no network), using small fixture files.

### Documentation

Rewrite README: what it does, data sources and their limits, quick start (`fpp evaluate`, `fpp train`,
`fpp predict`, `fpp score`), evaluation results table, how sentiment enters, credit to the original FPP.

### Out of scope

- Web app or dashboard (pl-predictor already has one).
- Betting strategy or staking.
- Player-level models, injuries, lineups.
- Backfilling historical sentiment (X recent search only reaches 7 days back).


<!-- autoplan-accepted:ceo -->
- C-1 xG: the stage-1 feature list never contains an xG column. Test: features built from a match table that includes 2026/27 xG columns contain no xG feature.
- C-2 Sentiment cutoff: T = kickoff - 60 min; sentiment features use tweets with kickoff - 24h <= created_at < T for training and prediction. README tells users to run pl-predictor with `PLS_PRE_MATCH_MINUTES=1440` for tracked fixtures. `fpp predict` applies stage 2 only when the run time is at or after T; earlier runs show base probabilities and label sentiment "partial (n tweets so far)". Tests: tweet at T excluded, at T - 1s included, at kickoff - 24h - 1s excluded; predict before T never uses stage 2.
- C-3 Stage-2 gate: enable only if the paired bootstrap 95% CI (1,000 resamples, fixed seed) of stage-2 minus stage-1 log loss on time-ordered held-out folds is entirely below 0; otherwise stay disabled and print the CI. Test: synthetic informative sentiment enables, pure-noise sentiment does not.
- C-4 Sparse history: rolling features are NaN when a side has < 5 league matches in the prior 400 days; such rows stay in training; HistGradientBoosting uses native NaN, LogisticRegression uses SimpleImputer(add_indicator=True). `fpp predict` outputs probabilities for every fixture. Test: a club with zero prior matches gets probabilities that sum to 1.
- C-5 Dependency: supported install is `pip install -e ../pl-predictor`; README lists publishing pl-predictor to a git remote as a user-owned prerequisite for a URL dependency.
- C-6 football-data.org optional: with no FOOTBALL_DATA_ORG_TOKEN, skip the freshness fill and warn with the newest match date in the table; never fail. Test: no token → warning, table unchanged.
- C-7 ClubElo resilience: cache each successful club_history pull under data/cache/clubelo/; on requests.RequestException use the cache and warn with its age; with no cache, exit non-zero naming the club and the error. Tests: HTTPError + cache → cached data used; HTTPError + no cache → clear error.
- C-8 Cost note: README states X API pay-per-use pricing (~$0.005/read, unconfirmed), a recommended PLS_MAX_TWEETS_PER_FIXTURE, and the resulting monthly estimate for 10 fixtures/week.
- C-9 Track record (accepted cherry-pick #1): `fpp predict` appends each prediction to predictions/history.csv (fixture key, kickoff, P(H/D/A), model version, stage used, predicted_at); `fpp score` joins finished results and reports log loss, Brier and accuracy for logged predictions vs closing-odds implied probabilities and the Elo baseline on the same fixtures. Test: scoring a hand-made history with known outcomes gives the expected log loss.
- C-10 Odds normalisation: data.py maps per-season odds columns (Avg*/AvgC* from 2019/20; BbAv*/PSC* earlier) to avg_h/d/a and close_h/d/a, NaN where absent. Test: an old-format and a new-format sample both normalise.
- C-11 Prediction history: history.csv rows are keyed by (fixture key, predicted_at); `fpp score` uses only the latest row per fixture with predicted_at < kickoff and ignores rows predicted at or after kickoff. Test: two pre-kickoff rows and one post-kickoff row for a fixture → only the latest pre-kickoff row is scored.
- C-12 Fixture key: (season, home code, away code), where season (C-34) is the start year of the season containing the kickoff. Each pairing is played once per PL season, so a rescheduled match keeps its key. FPL fixtures, football-data.co.uk results, football-data.org matches, logged predictions and pl-predictor fixtures all join on this key. An FPL fixture, or a logged prediction whose key is not a known FPL fixture of that season, is listed in an error (C-32); an unmatched pl-predictor fixture (e.g. demo or non-PL data) is ignored with a warning. Tests: a rescheduled fixture still matches its logged prediction; an unmatched FPL fixture raises with its key; a demo fixture in the sentiment DB only warns.
- C-13 Tweet sides and sentiment columns: home uses scored tweets with side in {home, both}; away uses {away, both} (same rule as pl-predictor `_for_side`); `neither` and unscored tweets are excluded. Model inputs are exactly home_net, away_net and net_diff (home_net - away_net). Tweet counts per side are shown in output and used for eligibility (C-33) only, never as model inputs, because pl-predictor's per-fixture cap makes volume mostly a function of the cap. Test: one tweet of each side plus one unscored tweet produce the expected means and counts.
- C-14 Stage-2 eligibility: only fixtures with kickoff on or after `FPP_SENTIMENT_SINCE` (a date in .env, set when pl-predictor was switched to the 24h window) count toward the 150-fixture threshold and stage-2 training; unset means no fixture is eligible and `fpp train` says so. Test: fixtures before the date are excluded.
- C-15 Edge cases: FPL fixtures with null kickoff_time are skipped and listed as unscheduled; "next gameweek" is the lowest gameweek with any unplayed fixture and only its unplayed fixtures are predicted; a team with two fixtures in a gameweek gets both; a failed current-season football-data.co.uk fetch aborts with the URL and HTTP error (no silent stale data); any football-data.org error (missing token, 429, 5xx, requests.RequestException) warns and skips the fill (extends C-6); rest days are NaN when a side has no previous match. Tests: one per listed case, with mocked hooks.
- C-16 TODOS.md: create TODOS.md listing deferred proposals 2-8 (what, why, effort, priority, context).
- C-17 CLI surface: `fpp score` appears in the package layout, `fpp --help` and the README quick start.
- C-18 Stage selection is per fixture: each output row carries `stage` (1 or 2); stage 2 is used only for fixtures whose cutoff T has passed at run time and only when the stage-2 model is enabled; README explains running `fpp predict` within the hour before kickoff to get stage 2. Test: a run where one fixture is past T and one is not yields stage 2 and stage 1 respectively (with an enabled stage-2 model).
- C-19 Saved models: models/model.joblib holds the stage-1 model, the Elo-only baseline, the optional stage-2 model, the feature list, training date and `model_version` = "<training date>-<git short sha>". `predictions/` and `models/` are gitignored except a `.gitkeep`. `fpp predict` writes the Elo-baseline P(H/D/A) into history.csv next to the model's, so `fpp score` never uses a baseline retrained after the result was known.
- C-20 Stage-2 inputs are out-of-fold: the stage-1 log-odds fed to stage 2 for a fixture come from a stage-1 model trained only on matches before that fixture's fold. Test: changing a fixture's own result does not change its stage-1 input.
- C-21 Sentiment database: `--sentiment-db PATH` (default from `FPP_SENTIMENT_DB` in .env); if the file is missing or has no matching fixtures, warn and run stage 1 only. Test: missing path → warning and stage-1 output.
- C-22 Install order: pyproject does not declare pl-predictor; README installs `pip install -e ../pl-predictor` before `pip install -e .`; importing fpp without pl-predictor exits with that install hint. Test: a simulated ImportError message contains the command.
- C-23 Collection sizing: the C-8 cost note is labelled an unconfirmed third-party estimate, sizes PLS_MAX_TWEETS_PER_FIXTURE for the 24h window plus match (recommended 400 across pl-predictor's 8 first-run slices), and `fpp predict` prints each fixture's pre-cutoff tweet count so thin coverage is visible.
- C-24 Live stage-2 workflow: README documents, per fixture: (1) the fixture is in pl-predictor `config/fixtures.csv` before kickoff - 24h, (2) `pl-predictor run` (collect and score) runs after T and before kickoff, (3) `fpp predict` runs after that. A fixture with no scored pre-cutoff tweet for either side uses stage 1 and its row says "stage 1: no sentiment". Test: a past-T fixture with zero scored tweets gets stage 1 with that reason.
- C-25 Stage-2 missing values: stage 2 is SimpleImputer(add_indicator=True) + LogisticRegression, so a side with no scored tweets (NaN net) is handled; C-24's fallback applies only when both sides have none. Test: one-sided sentiment produces probabilities summing to 1.
- C-26 Stage-2 inputs: log(P(H)/P(A)) and log(P(D)/P(A)) from the out-of-fold stage-1 model (C-20), plus the C-13 sentiment columns. Probabilities are clipped to [1e-6, 1] before the logs.
- C-27 Season start: before the first FPL kickoff of a season, a 404 for the current-season football-data.co.uk file means "no current-season matches yet" (warn and continue); after that kickoff, any failure aborts as in C-15. Test: 404 with all FPL fixtures in the future → empty current season, predictions still produced.
- C-28 Fill-only matches: matches filled from football-data.org (no stats) count toward goals rolling means, table context, rest days and the 5-match threshold; stat rolling means (shots, shots on target, corners, fouls) use the last 5/10 matches where that stat is present. Test: a window containing one fill-only match averages stats over the remaining matches and goals over all.
- C-29 Score comparison set: `fpp score` reports model, Elo baseline and closing-odds metrics on the fixtures that have all three (logged model probabilities, logged Elo probabilities, closing odds), with n, and separately reports model-only metrics on all scored fixtures with their n. Test: a fixture without closing odds is excluded from the three-way table but counted in model-only.
- C-30 Collection timing: README requires the first `pl-predictor run --fixture <id>` for a tracked fixture to happen in [T, kickoff), so pl-predictor's 8-slice first run covers the 24h pre-match window (C-23 sizing assumes this). A fixture whose earliest pl-predictor `collected_at` is before T - 5 min is excluded from stage-2 training with a warning, because later runs fill only from the newest tweet backwards. Test: a fixture first collected at T - 3h is excluded and named in the warning.
- C-31 Kickoff source: the sentiment window uses the FPL kickoff for live predictions and the match-table kickoff (football-data.co.uk Date + Time, Europe/London converted to UTC) for training. If pl-predictor's stored kickoff differs from it, warn and treat the fixture as having no sentiment. Test: a 2-hour kickoff mismatch yields no sentiment and a warning.
- C-32 Scoring unplayed fixtures: `fpp score` skips logged fixtures that have no result yet and reports them as "pending: n"; only keys that are not known FPL fixtures of that season raise. Test: a history with one played and one unplayed fixture scores one and reports pending: 1.
- C-33 Stage-2 eligibility (single definition): kickoff >= FPP_SENTIMENT_SINCE (C-14), at least one side has >= 1 scored tweet before T, and the C-30 timing holds. This drives the 150-fixture threshold and the stage-2 training set. Test: one fixture failing each condition is excluded; one meeting all is counted.
- C-34 Season rule: season = year if the Europe/London month is 8 or later, else year - 1 (so a July 2020 kickoff belongs to 2019/20). football-data.co.uk rows take the season of the file they came from. Test: 2020-07-26 maps to 2019; 2026-08-15 maps to 2026.
- C-35 Stage-2 folds: TimeSeriesSplit(5) over eligible fixtures ordered by kickoff; for each fold, stage 1 is refit on all matches dated before that fold's first kickoff and predicts the fold (this is C-20's out-of-fold input); stage 2 trains on the earlier eligible fixtures; the C-3 bootstrap resamples the pooled held-out predictions. If the first fold has fewer than 25 eligible training fixtures, the gate stays off. Test: fold boundaries are strictly increasing in kickoff and no fold trains on its own fixtures.
- C-36 Settings: fpp adds python-dotenv, loads its own .env at CLI start (FPP_SENTIMENT_SINCE, FPP_SENTIMENT_DB, FOOTBALL_DATA_ORG_TOKEN) and passes the token explicitly as FootballDataOrgHook(token=...); a missing token or the hook's ValueError follows C-6 (warn, skip fill). Test: with only fpp's .env setting the token, the hook receives it; with none, predict warns and continues.
- C-37 Collect first: the first implementation task, before any fpp code, sets pl-predictor `PLS_PRE_MATCH_MINUTES=1440`, records `FPP_SENTIMENT_SINCE` in fpp's .env and starts automated collection with `fpp watch` (C-38) or, until that exists, a documented Task Scheduler entry running pl-predictor per kickoff slot. Verification: pl-predictor's DB holds tweets older than kickoff - 20h for the first tracked gameweek.
- C-38 `fpp watch` (reopens deferred proposal 3; accepted): a long-running command that, for each upcoming FPL fixture, at kickoff - 55 min builds a pl-predictor Fixture from the FPL data and pl-predictor's teams.yaml, upserts it into the sentiment DB, runs pl-predictor's collect and score for that fixture through its Python API, then runs `fpp predict` for that fixture and appends to history.csv. One fixture failing is logged and does not stop the others. README shows how to start it from Windows Task Scheduler or cron. This replaces C-24's manual steps 1-2 (fixtures.csv edit and manual run); C-30's timing rule still holds. Tests: with a fake clock and mocked pipeline, two kickoff slots trigger collect, score and predict in order at kickoff - 55 min; a failing fixture does not block the next.
- C-39 Market arm: the stage-2 study reports two comparisons on the same eligible fixtures: (stage-1 + sentiment) vs stage 1, and (market + sentiment) vs market, where market = de-margined pre-match average odds (football-data.co.uk Avg*/BbAv* history; for live fixtures a new `FootballDataCoUkHook.upcoming()` reading football-data.co.uk fixtures.csv, added to pl-predictor with a mocked test). README frames the headline question as "does pre-match fan sentiment carry information beyond the betting market?". Test: the study output contains both comparisons with n and CI.
- C-40 Stage 2 offline first: stage 2 is an offline study (`fpp study`) that (1) runs a simulation power analysis using the observed per-match log-loss variance and effect sizes 0.002, 0.005 and 0.01, printing the fixtures needed for 80% power, and (2) applies the C-3 gate to both C-39 comparisons. `fpp predict` uses stage 2 only when `fpp study` has passed and the user passes `--stage2` (or sets FPP_STAGE2=1). The power-analysis n replaces the fixed 150 threshold wherever it appears. Until enabled, every output row has stage 1. Tests: power analysis on synthetic data returns larger n for smaller effects; predict without the flag never uses stage 2.
- C-41 Cost and stop rule: README gives the X API cost to reach the power-analysis n (fixtures x per-fixture cap x unconfirmed ~$0.005/read, noting about 1 of pl-predictor's 8 first-run slices falls after kickoff and returns nothing), and a stop rule: stop collecting at the earliest of reaching that n, a user-set spend cap, or the end of the 2026/27 season, then run `fpp study` and record the result either way.
- C-42 Reporting: success criteria add the count of scored live predictions and the model-vs-market log-loss gap; README and `fpp evaluate`/`fpp score` lead with log loss and Brier, with accuracy labelled secondary.
- C-43 Fair market comparator: `fpp score` and `fpp evaluate` compare against de-margined pre-match average odds (available before T) as the fair market and report closing odds separately as the ceiling. C-29's three-way set uses pre-match odds; closing-odds metrics are an extra column on the fixtures that have them.
- C-44 Missing kickoff times: seasons without a football-data.co.uk Time column get kickoff time NaN; all as-of features use the date only; sentiment windows (C-31) apply only to rows with a kickoff time. Test: a 2005/06 sample without Time builds features without error.
- C-45 TODOS.md additions (with C-16): Understat xG history since 2014/15 (rejected for now: unofficial scraping, terms and stability risk) and a Dixon-Coles/Poisson goal model as a stage-1 candidate.

- C-46 pl-predictor adapter: fpp/sentiment_source.py is the only fpp module that imports pl_predictor storage, pipeline, fixtures or teams; all other fpp modules use it or pl_predictor.hooks. Test: an import-graph test fails if another fpp module imports those pl_predictor modules.
- C-47 Single writer: `fpp watch` takes a lock file (data/watch.lock) and a second instance exits with a message; README says not to run `pl-predictor run --watch` alongside it; the adapter opens SQLite with a 30 s timeout and watch logs sqlite3.OperationalError and retries the fixture on the next loop. Tests: second instance exits; a simulated locked DB is retried.
- C-48 Watch refresh: `fpp watch` refreshes FPL fixtures every 15 minutes, reschedules fixtures whose kickoff moved, lists null-kickoff fixtures as unscheduled, and marks a fixture MISSED (logged; ineligible per C-30) if its slot passed without collection. Test: fake clock with a kickoff moved by 2 hours reschedules; a skipped slot is MISSED.
- C-49 Explicit rescue paths: a football-data.co.uk file missing expected columns raises SchemaError naming the file and columns; a missing models/model.joblib makes predict/score/study exit 2 with "run fpp train first"; sklearn InconsistentVersionWarning on load is shown with a retrain hint; history.csv PermissionError (file open elsewhere) retries 3 times 1 s apart, then errors naming the file; no upcoming fixtures prints "no upcoming fixtures" and exits 0; empty history makes score print "no logged predictions" and exit 0; empty match table makes evaluate/train refuse with a message; duplicate keys from fd.co.uk and fd.org keep the fd.co.uk row. Tests: one per case.
- C-50 Repo hygiene: .gitignore covers .env, data/, models/, predictions/ (except .gitkeep), logs/; README warns that model.joblib is a pickle and only self-produced files should be loaded.
- C-51 Layout: the package also has fpp/study.py, fpp/watch.py and fpp/sentiment_source.py; `fpp study` and `fpp watch` appear in the CLI and README alongside evaluate, train, predict and score.
- C-52 Test determinism: every time-dependent function takes an injected `now`; each CLI command has an end-to-end test on recorded fixture data with mocked hooks; `fpp predict` has a golden-file test on a recorded FPL + football-data.co.uk snapshot.
- C-53 Observability: each command logs one INFO summary (newest match date per source, rows used, warnings, model version); `fpp watch` logs every fixture state change to logs/watch.log (rotating). Test: watch on a fake clock writes the expected transitions.
<!-- /autoplan-accepted:ceo -->
## Review record

### Phase 1 (CEO) — Step 0

Mode: SELECTIVE EXPANSION (set by /autoplan override; no question asked). Review depth: implementation-ready.
Outside voice: Codex `not_authed` (installed, no credentials) → CEO outside coverage unavailable; native Claude passes run.

**Pre-review audit.** Fork at `0c47284` ("Project Cleaning 2"), identical to upstream (0 ahead / 0 behind). Branch
`feature/match-predictions`, base `master`. No CLAUDE.md, TODOS.md, design doc, stashes, TODO/FIXME markers, prior
learnings or binding decisions (`gstack-decision-search --scope branch --recent 50` returned nothing).
Taste references: good: `ml_functions/feature_engineering_functions.py:20` (`running_mean`, small and correct),
pl-predictor `src/pl_predictor/hooks.py` (thin clients, tested). Poor: `predictions/predictions.py:1-80` (script with
hardcoded file names and `sys.path` hacks), `01_api_data_request.py:42` (key read from a file at import time).

**Landscape (WebSearch fallback; Aside not installed).** [Layer 1] Elo/Bradley-Terry plus recent-form features
are the standard; closing odds are the benchmark. [Layer 2] A public EPL backtest reports log loss 0.973 for an
Elo-and-goals model vs 0.949 for de-margined closing odds (sportsedge.dedyn.io, self-reported); course notes find
adding Elo to odds barely helps (karlwhelan.com). [Layer 3] Sentiment is the only input here the market may not
fully price, which is why it is worth measuring, but its effect will be small and needs a strict test.

#### 0A. Premise challenge

Real problem: per-fixture H/D/A probabilities the user can trust, with a measured answer to "does fan sentiment
help?". The plan solves it directly. Do-nothing cost: FPP stays broken (2020 pins, paid API) and sentiment stays
unconnected to outcomes.

| # | Premise | Claim | Receipt | Verdict |
|---|---|---|---|---|
| P1 | FPP rolling features use only prior games | target is game N+1 of each window | `ml_functions/feature_engineering_functions.py:146-163` (`running_mean(...)[:x]`, targets from `games_slide-xx:`) | verified |
| P2 | football-data.co.uk has shots/corners/fouls back to 2005/06 | HS/HST/HC/HF present | live run 2026-10-18: 2005..2026 all report `shots` (`pl_predictor/hooks.py:40` `results()`) | verified |
| P3 | xG available for training ("where present") | HxG/AxG in history | same live run: HxG/AxG only in 2026/27 (60 matches) | **false** → xG cannot be a training feature |
| P4 | Closing odds exist for the 3 evaluation seasons | AvgC* columns | live run: `closing` for 2019..2026; `BbAv*`/`PSC*` before 2019 | verified (column names change by season) |
| P5 | pl-predictor installs "from its git repository" | git remote exists | `ls -a pl-predictor` has no `.git` | **false** |
| P6 | FPL gives next-gameweek fixtures | `fixtures(future=True)` | live run: 310 unplayed fixtures; `pl_predictor/hooks.py:80` | verified |
| P7 | ClubElo reachable | `club_history()` works | live run: HTTP 502 (http) / connect timeout (https) from this network | **unverified** |
| P8 | football-data.org fills fresh results | token available | 403 without token; user has no token yet | **unverified** |
| P9 | Pre-match sentiment exists at prediction time | tweets `phase=='pre'`, `created_at < kickoff` | pl-predictor `fixtures.py` window = kickoff − `PLS_PRE_MATCH_MINUTES` (default 120) | **flawed**: a prediction made days ahead has no pre-match tweets yet; training uses tweets up to kickoff, live use cannot |
| P10 | 150 fixtures can show a sentiment gain | log-loss comparison | [estimated] per-match log-loss SD ≈ 0.4 → SE of a 150-match mean difference ≈ 0.02–0.03, larger than the plausible gain | **weak**: needs a CI-based gate |
| P11 | Sentiment collection is affordable | X API Basic tier | X moved to pay-per-use Feb 2026, ~$0.005/read (third-party, unconfirmed) → 10 fixtures × 300 tweets × 4 weeks ≈ 12k reads ≈ $60/month [estimated] | cost must be stated |

unverified=2 (P7, P8).

#### 0B. Existing code leverage

| Sub-problem | Existing code | Plan |
|---|---|---|
| Fetch results/stats/odds | pl-predictor `FootballDataCoUkHook.results` | reuse |
| Upcoming fixtures | `FPLHook.fixtures(future=True)` | reuse |
| Elo | `ClubEloHook.club_history` | reuse (+ disk cache, see C-7) |
| Fresh results | `FootballDataOrgHook.matches` | reuse, optional |
| Sentiment rows | pl-predictor `Store.tweets_frame()` (tweets joined with scores) | reuse; join on kickoff date + home/away code |
| Club codes | pl-predictor `config/teams.yaml` codes = FPL short names | reuse as canonical code |
| Rolling form | FPP `running_mean` / `average_stats_df` | port to pandas `groupby().shift(1).rolling()`; FPP's dict-of-dataframes structure is replaced, not refactored, because it is keyed by api-football fixture IDs that no longer exist |
| Calibration view | FPP `ml_functions/ml_model_eval.py` `pred_proba_plot` | port idea as a 10-bin table |

#### 0C. Dream state

```
  CURRENT STATE                      THIS PLAN                              12-MONTH IDEAL
  FPP: 2020 pins, paid API,   --->   free data via 4 hooks, walk-forward  ---> weekly automated collect+predict,
  committed data, no tests;          eval vs Elo + odds, sentiment gated       a full season of logged predictions
  pl-predictor: sentiment            by CI, logged predictions + scoring       scored vs closing odds, a measured
  collected but unused                                                         answer on sentiment, shared view
```
The plan moves directly toward the ideal; scheduling and a shared view are deferred (TODOS.md).

#### 0D. Approach

A) restructure the FPP fork and depend on pl-predictor (user's stated direction). B) add the predictor inside
pl-predictor (one repo, one venv, no cross-repo install). C) monorepo merge. Chosen: **A** (user direction; P6).
B is simpler (no P5 problem) and is surfaced as a taste decision at the final gate. No question asked.

#### 0F/0G. Selective expansion

HOLD checks: ~10 new/changed files plus bulk deletions; 6 new modules, no services. Minimum change for the goal is
the planned package; nothing deferrable without losing a success criterion.
10x check: the 10x version is a season-long, honestly scored prediction record, not a better classifier. That is
cheap here (predictions already go to CSV).
Delight scan: (1) logged predictions + `fpp score` track record, (2) bookmaker odds shown next to each prediction,
(3) per-fixture feature explanations, (4) markdown output, (5) last-5 form string.

| # | Proposal | Effort | Decision | Reasoning |
|---|---|---|---|---|
| 1 | Prediction log + `fpp score` track record | S | ACCEPTED | in blast radius, <1d CC, makes success criterion 3 checkable live (P1, P2) |
| 2 | Pre-match odds as a model feature | M | DEFERRED | changes the model's nature (market follower); revisit after baseline results |
| 3 | Scheduled weekly collect + predict | S | DEFERRED | ops outside repo code |
| 4 | Predictions in the pl-predictor dashboard | M | DEFERRED | other repo |
| 5 | Local Elo computed from results (ClubElo fallback) | S | DEFERRED | taste: see gate; C-7 cache covers transient outages |
| 6 | FPL team xG as current-season feature | M | DEFERRED | no history to train on (P3) |
| 7 | Bookmaker odds next to predictions | S | DEFERRED | needs a new `fixtures.csv` hook method in pl-predictor |
| 8 | Per-fixture explanations | M | DEFERRED | nice-to-have |
| 9 | Markdown output | S | SKIPPED | CSV + table suffices |
| 10 | Last-5 form string | S | SKIPPED | cosmetic |

#### 0I. Temporal interrogation

- Hour 1: install (`pip install -e ../pl-predictor`), build `config/teams.csv` for ~45 clubs since 2005/06;
  ClubElo names must be read from a successful pull (blocked on P7 from this network).
- Hour 2-3: as-of features with `groupby(team).shift(1)`; season boundaries; promoted clubs with no recent PL
  history (Coventry: no PL match in the data at all) → C-4.
- Hour 4-5: football-data.co.uk `Time` is UK local time, FPL `kickoff_time` is UTC: join on date + codes, use FPL
  UTC for the sentiment cutoff; odds column names differ by season → C-10; no fd.org token → C-6.
- Hour 6+: tests with mocked hooks; README with cost note.
Effort: human ~5 days / CC + gstack ~3 hours.

#### Reality rows

```
REALITY: premise-table finding 11 premises, 3 false/flawed (P3,P5,P9), unverified=2 (P7,P8) PLAN.md:231-243
REALITY: already-done pass fork has 0 commits beyond upstream; no file in the plan exists yet .git (git log 0c47284)
REALITY: surface-check pass hook methods named in the plan exist: results() hooks.py:40, fixtures() hooks.py:80, club_history() hooks.py:110, matches() hooks.py:133; `fpp` CLI is new
REALITY: binding-decisions pass gstack-decision-search --scope branch returned no decisions; plan has no Binding decisions section PLAN.md:1
REALITY: numbers finding 150-fixture gate [estimated] PLAN.md:242; X cost [estimated] PLAN.md:243; column coverage [measured: live hook run] ../pl-predictor/src/pl_predictor/hooks.py:40
REALITY: deferred-asks pass no linked issue PLAN.md:1
REALITY: pr-count pass no .gstack/ship-policy.json → default one PR per wave PLAN.md:1
```

#### Step 0.5 — CEO dual voices

Claude CEO subagent: completed, INPUT ceo 43399617edfc… (read native-prompt.md lines 1-196). Codex: unavailable
(`not_authed`; run `codex login`). Tag: [single-model].

Findings (severity → disposition): 1 benchmark vs market (critical) → C-39; 2 underpowered gate (critical) → C-40;
3 collection is the binding constraint (critical) → C-37; 4 manual 60-min windows (high) → C-38; 5 no named consumer
(high) → C-42 + README framing, goal wording left to the user at the gate; 6 fork keeps little (medium-high) →
gate item #32; 7 cost/platform risk (medium) → C-41; 8 alternatives (medium) → C-39 (odds), C-45 (xG, Dixon-Coles);
9 competitive risk (medium) → README framing (C-39/C-42); 10a fair comparator → C-43; 10b missing Time → C-44;
10c accuracy → C-42; 10d spec size → C-40 moves stage-2 machinery offline.

```
CEO DUAL VOICES — CONSENSUS TABLE:
  Dimension                            Claude  Codex  Consensus
  1. Premises valid?                    no (P3,P5,P9-P11 + market premise)  N/A  N/A
  2. Right problem to solve?            partly (reframe to sentiment vs market)  N/A  N/A
  3. Scope calibration correct?         no (collection + scheduler missing; stage 2 too early)  N/A  N/A
  4. Alternatives sufficiently explored? no (odds-anchored, Understat xG, Dixon-Coles)  N/A  N/A
  5. Competitive/market risks covered?  no (stage 1 is commodity; rigour is the moat)  N/A  N/A
  6. 6-month trajectory sound?          no without C-37/C-40 (stage 2 likely never enables)  N/A  N/A
Outside voice unavailable: all Consensus cells N/A, never CONFIRMED. Single-voice criticals: findings 1, 2, 3.
```

#### CEO review sections 1-11

**Section 1: Architecture.**

```
                       pl-predictor (sibling repo, editable install)
   ┌───────────────────────────────────────────────────────────────────────┐
   │ hooks.py: FootballDataCoUk  FPL  ClubElo  FootballDataOrg  (+upcoming)  │
   │ storage.Store (SQLite: fixtures, tweets, sentiment)  pipeline  teams    │
   └──────────┬─────────────────────────┬───────────────────────┬──────────┘
              │ HTTP clients             │ read tweets           │ collect+score (watch only)
   ┌──────────▼──────────┐   ┌───────────▼────────────┐          │
   │ fpp/data.py         │   │ fpp/sentiment_source.py│◄─────────┘   (C-46: the only module
   │ match table + cache │   │ adapter over pl-sent.  │               importing pl_predictor
   └──────────┬──────────┘   └───────────┬────────────┘               storage/pipeline)
              │ matches                  │ per-fixture sentiment
   ┌──────────▼──────────────────────────▼────────┐
   │ fpp/features.py  as-of rolling, Elo, table,  │
   │                  rest days, sentiment cols   │
   └──────────┬───────────────────────────────────┘
   ┌──────────▼──────────┐  ┌──────────────┐  ┌─────────────────┐
   │ model.py (stage 1,  │  │ study.py     │  │ predict.py      │──► predictions/history.csv
   │ Elo baseline, eval) │  │ (stage 2 +   │  │ score (C-9/C-29)│◄── (append-only)
   └──────────┬──────────┘  │ power, C-40) │  └────────▲────────┘
              ▼             └──────────────┘           │
        models/model.joblib                   fpp/watch.py (C-38) ── scheduler loop
```

Data flows (happy / nil / empty / error):

```
 fd.co.uk season CSV ──► normalise odds/teams ──► match table ──► features ──► model
   nil: 404 before first kickoff → empty current season (C-27); 404 after → abort (C-15)
   empty: 0 rows → features empty → evaluate/train refuse with message (C-49)
   error: HTTPError/Timeout → abort with URL (C-15); unmapped team → UnmappedTeamError listing names
 FPL fixtures(future) ──► next gameweek ──► features as of kickoff ──► P(H/D/A) ──► history.csv
   nil: kickoff_time null → listed as unscheduled (C-15)
   empty: no future fixtures → "season over", exit 0 (C-49)
   error: HTTPError → abort; history.csv locked (Excel) → retry 3x, then error naming the file (C-49)
 pl-predictor DB ──► tweets in [kickoff-24h, T) ──► home/away net ──► stage-2 study
   nil: DB missing → warn, stage 1 only (C-21); empty: no tweets → "stage 1: no sentiment" (C-24)
   error: sqlite OperationalError (locked) → watch logs and retries next loop (C-47)
```

Fixture lifecycle in `fpp watch` (C-38, C-48):
```
 SCHEDULED ──(now >= kickoff-55m)──► COLLECTING ──► SCORED ──► PREDICTED ──(result known)──► SCORED_BY_FPP
     │  kickoff moved (FPL refresh every 15 min) → reschedule; null kickoff → UNSCHEDULED
     └─ now >= kickoff and not collected → MISSED (logged; ineligible for stage 2 per C-30)
 Invalid: PREDICTED before COLLECTING when stage 2 is enabled (watch always collects first).
```

Findings: (1) fpp would import several pl-predictor internals (Store, pipeline, Fixture, teams), coupling to a
non-public API → **C-46** single adapter module. (2) `fpp watch` and a separate `pl-predictor run --watch` can both
write the SQLite DB → **C-47**. (3) FPL kickoffs change after scheduling → **C-48** refresh and reschedule. Coupling to
pl-predictor is the main architectural debt; gate item #32 would remove it. Scaling: ~21 seasons x 380 = ~8k rows;
nothing breaks at 10x. Rollback: revert the branch; models and predictions are local files. Security surface: no
servers or endpoints.

**Section 2: Error & Rescue Map.**

```
  CODEPATH                         | WHAT CAN GO WRONG                     | EXCEPTION
  ---------------------------------|---------------------------------------|----------------------------
  data.load_season (fd.co.uk)      | 404 / 5xx / timeout                   | requests.HTTPError/Timeout
                                   | schema drift (missing column)         | KeyError → SchemaError
                                   | old-season encoding                   | handled in hook (cp1252)
  data.fill_from_fdorg             | no token / 429 / 5xx                  | ValueError / HTTPError
  data.elo (ClubElo)               | 502 / timeout                         | requests.RequestException
                                   | club name not in ClubElo              | empty frame → UnmappedTeamError
  teams.map_source                 | unknown name                          | UnmappedTeamError
  predict.next_gameweek (FPL)      | HTTP error / JSON change              | HTTPError / KeyError
  model.load                       | file missing / sklearn version change | FileNotFoundError / InconsistentVersionWarning
  history.append                   | file open in Excel (Windows lock)     | PermissionError
  sentiment_source.read            | DB missing / locked                   | FileNotFoundError / sqlite3.OperationalError
  watch.run_fixture                | any of the above, X API errors        | per-fixture isolation

  EXCEPTION                 | RESCUED? | ACTION                                         | USER SEES
  --------------------------|----------|------------------------------------------------|----------------------------
  HTTPError (fd.co.uk cur)  | Y        | abort with URL (C-15), 404 pre-season ok (C-27) | clear error / warning
  SchemaError               | N ← GAP  | → C-49: raise naming file + missing columns     | clear error
  fd.org errors             | Y        | warn, skip fill (C-6/C-36)                     | warning with newest date
  ClubElo RequestException  | Y        | cache fallback (C-7)                           | warning with cache age
  UnmappedTeamError         | Y        | raise listing names                            | actionable error
  FileNotFoundError model   | N ← GAP  | → C-49: "run fpp train first", exit 2          | clear message
  InconsistentVersionWarning| N ← GAP  | → C-49: warn and suggest retrain               | warning
  PermissionError history   | N ← GAP  | → C-49: retry 3x 1s, then error naming file    | clear error
  sqlite OperationalError   | Y        | C-47 timeout 30s; watch logs, retries next loop| log line
```

**Section 3: Security & threat model.** No network listeners or user-supplied inputs beyond CLI args and .env.
Threats: (a) secrets (FOOTBALL_DATA_ORG_TOKEN; pl-predictor's X token) committed by accident: likelihood Med, impact
Med → **C-50** gitignore .env, data/, models/, predictions/, logs/. (b) `model.joblib` is a pickle: loading an
untrusted file runs code; likelihood Low, impact High → C-50 README warning; fpp only loads models it wrote. (c)
ClubElo is plain HTTP: tampered ratings in transit; Low/Low; integrity only, accepted. (d) Tweet text and author IDs
stay in pl-predictor's DB; history.csv holds no tweet content. No injection vectors (no SQL built from input: Store
uses parameters).

**Section 4: Data flow & interaction edge cases.**

```
 INPUT ──► VALIDATION ──► TRANSFORM ──► PERSIST ──► OUTPUT
 fd.co.uk CSV → columns/teams checked → normalise (C-10) → data/cache/*.csv (past seasons) → match table
   wrong type: non-numeric stat → coerced to NaN (pd.to_numeric errors='coerce'), counted in log
   duplicate: same key from fd.co.uk and fd.org → keep fd.co.uk row (C-49)
   stale: cache only for completed seasons; current season always refetched
 FPL fixtures → kickoff present? → key (C-12) → history.csv → table
   partial gameweek / double gameweek / postponed → C-15
```

| INTERACTION | EDGE CASE | HANDLED? | HOW |
|---|---|---|---|
| `fpp predict` | run twice before kickoff | Y | C-11 latest pre-kickoff row scored |
| `fpp predict` | season over | GAP → C-49 | "no upcoming fixtures", exit 0 |
| `fpp watch` | machine sleeps through a slot | Y | MISSED state, logged (C-48) |
| `fpp watch` | two instances running | GAP → C-47 | lock file in data/, second instance exits |
| `fpp score` | history.csv empty | GAP → C-49 | "no logged predictions", exit 0 |
| `fpp study` | n below power target | Y | prints n needed (C-40) |

**Section 5: Code quality.** Layout fits a small package; risks are a large features.py and 45+ conditions
spread across modules. Add study.py, watch.py, sentiment_source.py to the layout (**C-51**). No duplicated
behaviour with pl-predictor beyond the side rule (C-13 reuses `_for_side` semantics; the adapter should call one
shared helper rather than re-implement it). Over-engineering risk is contained by C-40 (stage 2 offline).

**Section 6: Tests.**

```
 NEW CODEPATH                    TYPE         HAPPY                     FAILURE                     EDGE
 data.load_season               unit         sample CSV → table        404 cur season (C-15/C-27)  old cp1252 file, no Time (C-44)
 data.fill_from_fdorg           unit         fills missing match       429 → warn (C-15)           duplicate keys (C-49)
 data.elo                       unit         as-of join                RequestException→cache (C-7) unmapped club
 features.build                 unit         as-of rolling             —                           promoted club NaN (C-4), fill-only (C-28)
 features.sentiment             unit         window + sides (C-2,C-13) DB missing (C-21)           kickoff mismatch (C-31)
 model.walk_forward             unit         3 seasons report          empty table (C-49)          no shuffle (split test)
 study (power + gate)           unit         synthetic signal enables  noise stays off (C-3)       n < 25 first fold (C-35)
 predict                        unit/CLI     next GW rows sum to 1     no model (C-49)             partial/double GW (C-15)
 score                          unit/CLI     known log loss (C-9)      empty history (C-49)        pending/unplayed (C-32)
 watch                          unit (clock) order of steps (C-38)     one fixture fails           reschedule (C-48)
 CLI smoke (all commands)       integration  fixture data, mocked hooks (C-52)
```
Gaps: time-dependent code must take an injected clock to avoid flaky tests → **C-52**; an end-to-end CLI test per
command on fixture data → C-52. The 2am-Friday test: `fpp predict` on a recorded FPL + fd.co.uk snapshot produces
the same CSV as a golden file. Hostile test: a club renamed mid-season in fd.co.uk → UnmappedTeamError, not silent drop.

**Section 7: Performance.** First run fetches ~21 season CSVs and ~45 ClubElo histories (cached afterwards);
estimated 1-2 min on a normal link [estimated]. Training on ~8k rows and 1,000 bootstrap resamples is seconds.
The power simulation is bounded (3 effect sizes x 1,000 draws). No issues found beyond caching already planned.

**Section 8: Observability.** Each command logs one INFO summary: data freshness per source (newest match date),
rows used, warnings count, model version. `fpp watch` logs every state transition to logs/watch.log (rotating,
gitignored) so a missed slot can be reconstructed → **C-53**. history.csv is the audit trail for predictions.

**Section 9: Deployment & rollout.** Local tool on a branch, one PR. Deleting FPP's data and experiment folders is
reversible through git history. Order: C-37 collection change first (affects pl-predictor config only), then fpp.
Post-merge checks: `fpp evaluate` reproduces the README table from cached data; `fpp predict` for the next gameweek
prints 10 rows summing to 1; `fpp watch --once --dry-run` lists the next slots. No feature flags needed beyond
FPP_STAGE2 (C-40).

**Section 10: Long-term trajectory.** Reversibility 4/5 (the fork permanently diverges from upstream, which last
changed Mar 2026). Debt: cross-repo coupling (gate #32), spec size (45+ conditions), X API dependence (C-41 stop
rule). Phase 2: market-anchored model if `fpp study` passes; public track record view (deferred #4). Platform
potential: the match table and features are reusable for other leagues via the same hooks.

**Section 11: Design & UX.** SKIPPED (no UI scope).

#### Required outputs (CEO)

**NOT in scope.** Deferred (TODOS.md): pre-match odds as stage-1 feature (#2; partly superseded by C-39's market
arm), predictions in pl-predictor dashboard (#4), local Elo fallback (#5), FPL team xG (#6), odds next to predictions
(#7, superseded by C-39's `upcoming()` hook), per-fixture explanations (#8), Understat xG, Dixon-Coles model (C-45).
Rejected: markdown output (#9), form string (#10), web app, betting/staking, player-level models, historical sentiment backfill.

**What already exists.** See 0B table: four pl-predictor hooks (reused), Store.tweets_frame (reused via adapter),
pipeline collect/score (reused by watch), teams.yaml codes (canonical), FPP running_mean idea (ported).

**Dream state delta.** After this plan: a working predictor with honest walk-forward and live scoring, automated
collection, and a powered study answering "sentiment beyond the market". Still missing vs the 12-month ideal: a
shared/public view and a full season of logged predictions (time, not code).

**Failure Modes Registry.**

```
  CODEPATH              | FAILURE MODE                | RESCUED? | TEST? | USER SEES        | LOGGED?
  ----------------------|-----------------------------|----------|-------|------------------|--------
  data.load_season      | current-season fetch fails  | Y        | Y     | error with URL   | Y
  data.load_season      | schema drift                | Y (C-49) | Y     | error, columns   | Y
  data.elo              | ClubElo down                | Y (C-7)  | Y     | warning, age     | Y
  data.fill_from_fdorg  | token missing / 429         | Y        | Y     | warning          | Y
  features.sentiment    | DB missing                  | Y (C-21) | Y     | warning          | Y
  predict               | model file missing          | Y (C-49) | Y     | "run fpp train"  | Y
  history.append        | file locked                 | Y (C-49) | Y     | error naming file| Y
  watch                 | slot missed (sleep)         | Y (C-48) | Y     | MISSED in log    | Y
  watch                 | second instance             | Y (C-47) | Y     | exits with note  | Y
  study                 | underpowered                | Y (C-40) | Y     | n needed         | Y
```
CRITICAL GAPS: 0 after C-46..C-53 (4 rescue gaps found in Section 2, all assigned to C-49).

**Stale diagram audit.** Files this plan touches have no ASCII diagrams (FPP README uses PNG figures, all removed
with figures/). pl-predictor README's ASCII flow is unchanged by this plan.

#### Implementation Tasks (CEO)

- [ ] **T1 (P1, human: ~1h / CC: ~10min)** — pl-predictor config — switch to 24h pre-match window and start collection
  - Surfaced by: Step 0.5 finding 3 (C-37)
  - Files: pl-predictor `.env`, fpp `.env` (FPP_SENTIMENT_SINCE)
  - Verify: pl-predictor DB has tweets older than kickoff - 20h for the first tracked gameweek
- [ ] **T2 (P1, human: ~1d / CC: ~30min)** — fpp/data.py + teams.csv — match table, odds normalisation, caches, fill
  - Surfaced by: 0A P2-P4, C-10, C-15, C-27, C-28, C-44, C-49
  - Files: fpp/data.py, fpp/teams.py, config/teams.csv, tests/test_data.py
  - Verify: pytest tests/test_data.py
- [ ] **T3 (P1, human: ~1d / CC: ~30min)** — fpp/features.py — as-of features, Elo join, sentiment columns
  - Surfaced by: C-1, C-2, C-4, C-13, C-28, C-31, C-34
  - Files: fpp/features.py, fpp/sentiment_source.py, tests/test_features.py
  - Verify: pytest tests/test_features.py (as-of invariance test)
- [ ] **T4 (P1, human: ~1d / CC: ~30min)** — fpp/model.py — stage 1, Elo baseline, walk-forward, calibration table
  - Surfaced by: success criteria 2-3, C-19, C-42, C-43
  - Files: fpp/model.py, tests/test_model.py
  - Verify: `fpp evaluate` prints per-season log loss/Brier for model, Elo, pre-match and closing odds
- [ ] **T5 (P1, human: ~1d / CC: ~30min)** — predict/score — history.csv, scoring rules
  - Surfaced by: C-9, C-11, C-12, C-29, C-32, C-49
  - Files: fpp/predict.py, fpp/cli.py, tests/test_predict.py, tests/test_score.py
  - Verify: golden-file test of `fpp predict` on a recorded snapshot; scoring test with known log loss
- [ ] **T6 (P1, human: ~1d / CC: ~30min)** — fpp/watch.py — scheduler loop with pl-predictor collection
  - Surfaced by: Step 0.5 finding 4 (C-38, C-47, C-48, C-53)
  - Files: fpp/watch.py, tests/test_watch.py
  - Verify: fake-clock test; `fpp watch --once --dry-run` lists next slots
- [ ] **T7 (P2, human: ~1d / CC: ~30min)** — fpp/study.py — power analysis, two-arm gate, upcoming() hook
  - Surfaced by: Step 0.5 findings 1-2 (C-3, C-35, C-39, C-40)
  - Files: fpp/study.py, pl-predictor hooks.py (`upcoming()`), tests/test_study.py
  - Verify: synthetic signal passes, noise fails; power n decreases with effect size
- [ ] **T8 (P2, human: ~3h / CC: ~15min)** — repo cleanup, README, TODOS.md, .gitignore
  - Surfaced by: "What we remove", C-5, C-8, C-16, C-22, C-41, C-45, C-50
  - Files: README.md, TODOS.md, .gitignore, pyproject.toml; deletions listed in the plan
  - Verify: `pip install -e ../pl-predictor && pip install -e .` in a fresh venv; README commands run

#### Completion summary (CEO)

```
  +====================================================================+
  |            MEGA PLAN REVIEW — COMPLETION SUMMARY                   |
  +====================================================================+
  | Mode selected        | SELECTIVE EXPANSION (autoplan override)     |
  | System Audit         | fork = upstream; no tests/TODOs; 2020 pins  |
  | Step 0               | 11 premises (3 false/flawed, 2 unverified); |
  |                      | 1 of 10 proposals accepted; 3 spec reviews  |
  | Section 1  (Arch)    | 3 issues found (C-46, C-47, C-48)           |
  | Section 2  (Errors)  | 10 error paths mapped, 4 GAPS (→ C-49)      |
  | Section 3  (Security)| 4 issues found, 0 High severity (→ C-50)    |
  | Section 4  (Data/UX) | 6 edge cases mapped, 3 unhandled (→ C-47/49)|
  | Section 5  (Quality) | 1 issue found (C-51)                        |
  | Section 6  (Tests)   | Diagram produced, 2 gaps (C-52)             |
  | Section 7  (Perf)    | 0 issues found                              |
  | Section 8  (Observ)  | 1 gap found (C-53)                          |
  | Section 9  (Deploy)  | 1 risk flagged (collection first, C-37)     |
  | Section 10 (Future)  | Reversibility: 4/5, debt items: 3           |
  | Section 11 (Design)  | SKIPPED (no UI scope)                       |
  +--------------------------------------------------------------------+
  | NOT in scope         | written (14 items)                          |
  | What already exists  | written                                     |
  | Dream state delta    | written                                     |
  | Error/rescue registry| 10 rows, 0 CRITICAL GAPS after C-49         |
  | Failure modes        | 10 total, 0 CRITICAL GAPS                   |
  | TODOS.md updates     | 9 items proposed (deferred #2,#4-#8 + C-45) |
  | Scope proposals      | 10 proposed, 2 accepted (#1, #3 reopened)   |
  | CEO plan             | written (~/.gstack/.../ceo-plans/)          |
  | Outside voice        | codex: unavailable (not_authed)             |
  | Lake Score           | N/A (no scored questions asked)             |
  | Diagrams produced    | 4 (architecture, data flow, state, errors)  |
  | Stale diagrams found | 0                                           |
  | Unresolved decisions | 2 (gate #32 approach, goal wording)         |
  | Reality rows         | 7/7 rows, 2 unverified                      |
  +====================================================================+
```

<!-- autoplan-baseline-edits:ceo {"sourceSha256":"22934dbb63bf3a8b424a48908b86b2648b589b8273eef882186ce4a9d08936c6","replacements":[{"oldText":"Results, shots, shots on target, corners, fouls, cards per match from 2005/06; xG columns where present; pre-match and closing odds (benchmark only)","newText":"Results, shots, shots on target, corners, fouls, cards per match from 2005/06; xG columns exist only from 2026/27 (stored, not modelled); pre-match and closing odds (benchmark only)"},{"oldText":"`pl-predictor` is a dependency (hooks + sentiment database). It is installed from its git repository;\nfor local work `pip install -e ../pl-predictor`.","newText":"`pl-predictor` is a dependency (hooks + sentiment database). Install it with `pip install -e ../pl-predictor`.\npl-predictor has no git remote yet; publishing it is a user-owned prerequisite before pyproject can point at a URL."},{"oldText":"- Rolling 5 and 10 match means for each side of: goals for/against, shots, shots on target,\n  corners, fouls, xG for/against (when available). Windows span seasons, as in FPP.","newText":"- Rolling 5 and 10 match means for each side of: goals for/against, shots, shots on target,\n  corners, fouls. Windows span seasons, as in FPP. No xG (history has none before 2026/27)."},{"oldText":"- Sentiment (stage 2 only): pre-match net sentiment per side (mean of P(pos) - P(neg)) and tweet\n  volume, from pl-predictor tweets with `phase == 'pre'` and `created_at < kickoff`.","newText":"- Sentiment (stage 2 only): net sentiment per side (mean of P(pos) - P(neg)) and their difference, from\n  pl-predictor tweets with kickoff - 24h <= `created_at` < T, where T = kickoff - 60 min is the prediction\n  cutoff. The same window is used for training rows and live predictions."},{"oldText":"Matches where either side has fewer than 5 prior matches in the data are dropped from training.","newText":"Rolling features are NaN (not dropped) when a side has fewer than 5 league matches in the prior 400 days,\nso promoted clubs are still trained on and predicted."},{"oldText":"on those fixtures, and enable it in `predict` only if log loss improves on stage 1. Until then","newText":"on those fixtures, and enable it in `predict` only if the paired bootstrap 95% CI (1,000 resamples) of the\n  log-loss difference vs stage 1 lies entirely below 0. Until then"},{"oldText":"- Sentiment features only count tweets with `created_at < kickoff`.","newText":"- Sentiment features only count tweets with kickoff - 24h <= `created_at` < kickoff - 60 min."},{"oldText":"  cli.py          `fpp evaluate`, `fpp train`, `fpp predict`","newText":"  cli.py          `fpp evaluate`, `fpp train`, `fpp predict`, `fpp score`"},{"oldText":"pyproject.toml    Python >=3.10; pandas, numpy, scikit-learn, pl-predictor","newText":"pyproject.toml    Python >=3.10; pandas, numpy, scikit-learn (pl-predictor installed separately, see below)"},{"oldText":"- Table context: points per game and goal difference so far this season, as of D.","newText":"- Table context: points per game and goal difference this season, from matches strictly before D."},{"oldText":"  `LogisticRegression`, each wrapped in `CalibratedClassifierCV`. Pick by walk-forward log loss.","newText":"  `LogisticRegression`, each wrapped in `CalibratedClassifierCV(cv=TimeSeriesSplit(5))` fitted on\n  time-ordered rows. Pick by walk-forward log loss."},{"oldText":"quick start (`fpp evaluate`, `fpp train`,\n`fpp predict`), evaluation results table","newText":"quick start (`fpp evaluate`, `fpp train`,\n`fpp predict`, `fpp score`), evaluation results table"}]} -->

<!-- autoplan-accepted:ceo -->
- C-1 xG: the stage-1 feature list never contains an xG column. Test: features built from a match table that includes 2026/27 xG columns contain no xG feature.
- C-2 Sentiment cutoff: T = kickoff - 60 min; sentiment features use tweets with kickoff - 24h <= created_at < T for training and prediction. README tells users to run pl-predictor with `PLS_PRE_MATCH_MINUTES=1440` for tracked fixtures. `fpp predict` applies stage 2 only when the run time is at or after T; earlier runs show base probabilities and label sentiment "partial (n tweets so far)". Tests: tweet at T excluded, at T - 1s included, at kickoff - 24h - 1s excluded; predict before T never uses stage 2.
- C-3 Stage-2 gate: enable only if the paired bootstrap 95% CI (1,000 resamples, fixed seed) of stage-2 minus stage-1 log loss on time-ordered held-out folds is entirely below 0; otherwise stay disabled and print the CI. Test: synthetic informative sentiment enables, pure-noise sentiment does not.
- C-4 Sparse history: rolling features are NaN when a side has < 5 league matches in the prior 400 days; such rows stay in training; HistGradientBoosting uses native NaN, LogisticRegression uses SimpleImputer(add_indicator=True). `fpp predict` outputs probabilities for every fixture. Test: a club with zero prior matches gets probabilities that sum to 1.
- C-5 Dependency: supported install is `pip install -e ../pl-predictor`; README lists publishing pl-predictor to a git remote as a user-owned prerequisite for a URL dependency.
- C-6 football-data.org optional: with no FOOTBALL_DATA_ORG_TOKEN, skip the freshness fill and warn with the newest match date in the table; never fail. Test: no token → warning, table unchanged.
- C-7 ClubElo resilience: cache each successful club_history pull under data/cache/clubelo/; on requests.RequestException use the cache and warn with its age; with no cache, exit non-zero naming the club and the error. Tests: HTTPError + cache → cached data used; HTTPError + no cache → clear error.
- C-8 Cost note: README states X API pay-per-use pricing (~$0.005/read, unconfirmed), a recommended PLS_MAX_TWEETS_PER_FIXTURE, and the resulting monthly estimate for 10 fixtures/week.
- C-9 Track record (accepted cherry-pick #1): `fpp predict` appends each prediction to predictions/history.csv (fixture key, kickoff, P(H/D/A), model version, stage used, predicted_at); `fpp score` joins finished results and reports log loss, Brier and accuracy for logged predictions vs closing-odds implied probabilities and the Elo baseline on the same fixtures. Test: scoring a hand-made history with known outcomes gives the expected log loss.
- C-10 Odds normalisation: data.py maps per-season odds columns (Avg*/AvgC* from 2019/20; BbAv*/PSC* earlier) to avg_h/d/a and close_h/d/a, NaN where absent. Test: an old-format and a new-format sample both normalise.
- C-11 Prediction history: history.csv rows are keyed by (fixture key, predicted_at); `fpp score` uses only the latest row per fixture with predicted_at < kickoff and ignores rows predicted at or after kickoff. Test: two pre-kickoff rows and one post-kickoff row for a fixture → only the latest pre-kickoff row is scored.
- C-12 Fixture key: (season, home code, away code), where season (C-34) is the start year of the season containing the kickoff. Each pairing is played once per PL season, so a rescheduled match keeps its key. FPL fixtures, football-data.co.uk results, football-data.org matches, logged predictions and pl-predictor fixtures all join on this key. An FPL fixture, or a logged prediction whose key is not a known FPL fixture of that season, is listed in an error (C-32); an unmatched pl-predictor fixture (e.g. demo or non-PL data) is ignored with a warning. Tests: a rescheduled fixture still matches its logged prediction; an unmatched FPL fixture raises with its key; a demo fixture in the sentiment DB only warns.
- C-13 Tweet sides and sentiment columns: home uses scored tweets with side in {home, both}; away uses {away, both} (same rule as pl-predictor `_for_side`); `neither` and unscored tweets are excluded. Model inputs are exactly home_net, away_net and net_diff (home_net - away_net). Tweet counts per side are shown in output and used for eligibility (C-33) only, never as model inputs, because pl-predictor's per-fixture cap makes volume mostly a function of the cap. Test: one tweet of each side plus one unscored tweet produce the expected means and counts.
- C-14 Stage-2 eligibility: only fixtures with kickoff on or after `FPP_SENTIMENT_SINCE` (a date in .env, set when pl-predictor was switched to the 24h window) count toward the 150-fixture threshold and stage-2 training; unset means no fixture is eligible and `fpp train` says so. Test: fixtures before the date are excluded.
- C-15 Edge cases: FPL fixtures with null kickoff_time are skipped and listed as unscheduled; "next gameweek" is the lowest gameweek with any unplayed fixture and only its unplayed fixtures are predicted; a team with two fixtures in a gameweek gets both; a failed current-season football-data.co.uk fetch aborts with the URL and HTTP error (no silent stale data); any football-data.org error (missing token, 429, 5xx, requests.RequestException) warns and skips the fill (extends C-6); rest days are NaN when a side has no previous match. Tests: one per listed case, with mocked hooks.
- C-16 TODOS.md: create TODOS.md listing deferred proposals 2-8 (what, why, effort, priority, context).
- C-17 CLI surface: `fpp score` appears in the package layout, `fpp --help` and the README quick start.
- C-18 Stage selection is per fixture: each output row carries `stage` (1 or 2); stage 2 is used only for fixtures whose cutoff T has passed at run time and only when the stage-2 model is enabled; README explains running `fpp predict` within the hour before kickoff to get stage 2. Test: a run where one fixture is past T and one is not yields stage 2 and stage 1 respectively (with an enabled stage-2 model).
- C-19 Saved models: models/model.joblib holds the stage-1 model, the Elo-only baseline, the optional stage-2 model, the feature list, training date and `model_version` = "<training date>-<git short sha>". `predictions/` and `models/` are gitignored except a `.gitkeep`. `fpp predict` writes the Elo-baseline P(H/D/A) into history.csv next to the model's, so `fpp score` never uses a baseline retrained after the result was known.
- C-20 Stage-2 inputs are out-of-fold: the stage-1 log-odds fed to stage 2 for a fixture come from a stage-1 model trained only on matches before that fixture's fold. Test: changing a fixture's own result does not change its stage-1 input.
- C-21 Sentiment database: `--sentiment-db PATH` (default from `FPP_SENTIMENT_DB` in .env); if the file is missing or has no matching fixtures, warn and run stage 1 only. Test: missing path → warning and stage-1 output.
- C-22 Install order: pyproject does not declare pl-predictor; README installs `pip install -e ../pl-predictor` before `pip install -e .`; importing fpp without pl-predictor exits with that install hint. Test: a simulated ImportError message contains the command.
- C-23 Collection sizing: the C-8 cost note is labelled an unconfirmed third-party estimate, sizes PLS_MAX_TWEETS_PER_FIXTURE for the 24h window plus match (recommended 400 across pl-predictor's 8 first-run slices), and `fpp predict` prints each fixture's pre-cutoff tweet count so thin coverage is visible.
- C-24 Live stage-2 workflow: README documents, per fixture: (1) the fixture is in pl-predictor `config/fixtures.csv` before kickoff - 24h, (2) `pl-predictor run` (collect and score) runs after T and before kickoff, (3) `fpp predict` runs after that. A fixture with no scored pre-cutoff tweet for either side uses stage 1 and its row says "stage 1: no sentiment". Test: a past-T fixture with zero scored tweets gets stage 1 with that reason.
- C-25 Stage-2 missing values: stage 2 is SimpleImputer(add_indicator=True) + LogisticRegression, so a side with no scored tweets (NaN net) is handled; C-24's fallback applies only when both sides have none. Test: one-sided sentiment produces probabilities summing to 1.
- C-26 Stage-2 inputs: log(P(H)/P(A)) and log(P(D)/P(A)) from the out-of-fold stage-1 model (C-20), plus the C-13 sentiment columns. Probabilities are clipped to [1e-6, 1] before the logs.
- C-27 Season start: before the first FPL kickoff of a season, a 404 for the current-season football-data.co.uk file means "no current-season matches yet" (warn and continue); after that kickoff, any failure aborts as in C-15. Test: 404 with all FPL fixtures in the future → empty current season, predictions still produced.
- C-28 Fill-only matches: matches filled from football-data.org (no stats) count toward goals rolling means, table context, rest days and the 5-match threshold; stat rolling means (shots, shots on target, corners, fouls) use the last 5/10 matches where that stat is present. Test: a window containing one fill-only match averages stats over the remaining matches and goals over all.
- C-29 Score comparison set: `fpp score` reports model, Elo baseline and closing-odds metrics on the fixtures that have all three (logged model probabilities, logged Elo probabilities, closing odds), with n, and separately reports model-only metrics on all scored fixtures with their n. Test: a fixture without closing odds is excluded from the three-way table but counted in model-only.
- C-30 Collection timing: README requires the first `pl-predictor run --fixture <id>` for a tracked fixture to happen in [T, kickoff), so pl-predictor's 8-slice first run covers the 24h pre-match window (C-23 sizing assumes this). A fixture whose earliest pl-predictor `collected_at` is before T - 5 min is excluded from stage-2 training with a warning, because later runs fill only from the newest tweet backwards. Test: a fixture first collected at T - 3h is excluded and named in the warning.
- C-31 Kickoff source: the sentiment window uses the FPL kickoff for live predictions and the match-table kickoff (football-data.co.uk Date + Time, Europe/London converted to UTC) for training. If pl-predictor's stored kickoff differs from it, warn and treat the fixture as having no sentiment. Test: a 2-hour kickoff mismatch yields no sentiment and a warning.
- C-32 Scoring unplayed fixtures: `fpp score` skips logged fixtures that have no result yet and reports them as "pending: n"; only keys that are not known FPL fixtures of that season raise. Test: a history with one played and one unplayed fixture scores one and reports pending: 1.
- C-33 Stage-2 eligibility (single definition): kickoff >= FPP_SENTIMENT_SINCE (C-14), at least one side has >= 1 scored tweet before T, and the C-30 timing holds. This drives the 150-fixture threshold and the stage-2 training set. Test: one fixture failing each condition is excluded; one meeting all is counted.
- C-34 Season rule: season = year if the Europe/London month is 8 or later, else year - 1 (so a July 2020 kickoff belongs to 2019/20). football-data.co.uk rows take the season of the file they came from. Test: 2020-07-26 maps to 2019; 2026-08-15 maps to 2026.
- C-35 Stage-2 folds: TimeSeriesSplit(5) over eligible fixtures ordered by kickoff; for each fold, stage 1 is refit on all matches dated before that fold's first kickoff and predicts the fold (this is C-20's out-of-fold input); stage 2 trains on the earlier eligible fixtures; the C-3 bootstrap resamples the pooled held-out predictions. If the first fold has fewer than 25 eligible training fixtures, the gate stays off. Test: fold boundaries are strictly increasing in kickoff and no fold trains on its own fixtures.
- C-36 Settings: fpp adds python-dotenv, loads its own .env at CLI start (FPP_SENTIMENT_SINCE, FPP_SENTIMENT_DB, FOOTBALL_DATA_ORG_TOKEN) and passes the token explicitly as FootballDataOrgHook(token=...); a missing token or the hook's ValueError follows C-6 (warn, skip fill). Test: with only fpp's .env setting the token, the hook receives it; with none, predict warns and continues.
- C-37 Collect first: the first implementation task, before any fpp code, sets pl-predictor `PLS_PRE_MATCH_MINUTES=1440`, records `FPP_SENTIMENT_SINCE` in fpp's .env and starts automated collection with `fpp watch` (C-38) or, until that exists, a documented Task Scheduler entry running pl-predictor per kickoff slot. Verification: pl-predictor's DB holds tweets older than kickoff - 20h for the first tracked gameweek.
- C-38 `fpp watch` (reopens deferred proposal 3; accepted): a long-running command that, for each upcoming FPL fixture, at kickoff - 55 min builds a pl-predictor Fixture from the FPL data and pl-predictor's teams.yaml, upserts it into the sentiment DB, runs pl-predictor's collect and score for that fixture through its Python API, then runs `fpp predict` for that fixture and appends to history.csv. One fixture failing is logged and does not stop the others. README shows how to start it from Windows Task Scheduler or cron. This replaces C-24's manual steps 1-2 (fixtures.csv edit and manual run); C-30's timing rule still holds. Tests: with a fake clock and mocked pipeline, two kickoff slots trigger collect, score and predict in order at kickoff - 55 min; a failing fixture does not block the next.
- C-39 Market arm: the stage-2 study reports two comparisons on the same eligible fixtures: (stage-1 + sentiment) vs stage 1, and (market + sentiment) vs market, where market = de-margined pre-match average odds (football-data.co.uk Avg*/BbAv* history; for live fixtures a new `FootballDataCoUkHook.upcoming()` reading football-data.co.uk fixtures.csv, added to pl-predictor with a mocked test). README frames the headline question as "does pre-match fan sentiment carry information beyond the betting market?". Test: the study output contains both comparisons with n and CI.
- C-40 Stage 2 offline first: stage 2 is an offline study (`fpp study`) that (1) runs a simulation power analysis using the observed per-match log-loss variance and effect sizes 0.002, 0.005 and 0.01, printing the fixtures needed for 80% power, and (2) applies the C-3 gate to both C-39 comparisons. `fpp predict` uses stage 2 only when `fpp study` has passed and the user passes `--stage2` (or sets FPP_STAGE2=1). The power-analysis n replaces the fixed 150 threshold wherever it appears. Until enabled, every output row has stage 1. Tests: power analysis on synthetic data returns larger n for smaller effects; predict without the flag never uses stage 2.
- C-41 Cost and stop rule: README gives the X API cost to reach the power-analysis n (fixtures x per-fixture cap x unconfirmed ~$0.005/read, noting about 1 of pl-predictor's 8 first-run slices falls after kickoff and returns nothing), and a stop rule: stop collecting at the earliest of reaching that n, a user-set spend cap, or the end of the 2026/27 season, then run `fpp study` and record the result either way.
- C-42 Reporting: success criteria add the count of scored live predictions and the model-vs-market log-loss gap; README and `fpp evaluate`/`fpp score` lead with log loss and Brier, with accuracy labelled secondary.
- C-43 Fair market comparator: `fpp score` and `fpp evaluate` compare against de-margined pre-match average odds (available before T) as the fair market and report closing odds separately as the ceiling. C-29's three-way set uses pre-match odds; closing-odds metrics are an extra column on the fixtures that have them.
- C-44 Missing kickoff times: seasons without a football-data.co.uk Time column get kickoff time NaN; all as-of features use the date only; sentiment windows (C-31) apply only to rows with a kickoff time. Test: a 2005/06 sample without Time builds features without error.
- C-45 TODOS.md additions (with C-16): Understat xG history since 2014/15 (rejected for now: unofficial scraping, terms and stability risk) and a Dixon-Coles/Poisson goal model as a stage-1 candidate.

- C-46 pl-predictor adapter: fpp/sentiment_source.py is the only fpp module that imports pl_predictor storage, pipeline, fixtures or teams; all other fpp modules use it or pl_predictor.hooks. Test: an import-graph test fails if another fpp module imports those pl_predictor modules.
- C-47 Single writer: `fpp watch` takes a lock file (data/watch.lock) and a second instance exits with a message; README says not to run `pl-predictor run --watch` alongside it; the adapter opens SQLite with a 30 s timeout and watch logs sqlite3.OperationalError and retries the fixture on the next loop. Tests: second instance exits; a simulated locked DB is retried.
- C-48 Watch refresh: `fpp watch` refreshes FPL fixtures every 15 minutes, reschedules fixtures whose kickoff moved, lists null-kickoff fixtures as unscheduled, and marks a fixture MISSED (logged; ineligible per C-30) if its slot passed without collection. Test: fake clock with a kickoff moved by 2 hours reschedules; a skipped slot is MISSED.
- C-49 Explicit rescue paths: a football-data.co.uk file missing expected columns raises SchemaError naming the file and columns; a missing models/model.joblib makes predict/score/study exit 2 with "run fpp train first"; sklearn InconsistentVersionWarning on load is shown with a retrain hint; history.csv PermissionError (file open elsewhere) retries 3 times 1 s apart, then errors naming the file; no upcoming fixtures prints "no upcoming fixtures" and exits 0; empty history makes score print "no logged predictions" and exit 0; empty match table makes evaluate/train refuse with a message; duplicate keys from fd.co.uk and fd.org keep the fd.co.uk row. Tests: one per case.
- C-50 Repo hygiene: .gitignore covers .env, data/, models/, predictions/ (except .gitkeep), logs/; README warns that model.joblib is a pickle and only self-produced files should be loaded.
- C-51 Layout: the package also has fpp/study.py, fpp/watch.py and fpp/sentiment_source.py; `fpp study` and `fpp watch` appear in the CLI and README alongside evaluate, train, predict and score.
- C-52 Test determinism: every time-dependent function takes an injected `now`; each CLI command has an end-to-end test on recorded fixture data with mocked hooks; `fpp predict` has a golden-file test on a recorded FPL + football-data.co.uk snapshot.
- C-53 Observability: each command logs one INFO summary (newest match date per source, rows used, warnings, model version); `fpp watch` logs every fixture state change to logs/watch.log (rotating). Test: watch on a fake clock writes the expected transitions.
<!-- /autoplan-accepted:ceo -->

TODOS.md (to create in Required Outputs): proposals 2-8 above.

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|----------------|-----------|-----------|----------|
| 1 | CEO | Mode SELECTIVE EXPANSION | Mechanical | autoplan override | fixed by /autoplan | other modes |
| 2 | CEO | Keep approach A (restructure FPP fork) | Taste | P6 + user direction | user said "build on FPP"; B (inside pl-predictor) is simpler | B, C |
| 3 | CEO | Drop xG from training features (C-1) | Mechanical | P1 | xG absent before 2026/27 (P3) | train on 60 matches |
| 4 | CEO | Sentiment cutoff window (C-2) | Mechanical | P1 | live predictions cannot see tweets up to kickoff (P9) | phase=='pre' up to kickoff |
| 5 | CEO | CI-based stage-2 gate (C-3) | Mechanical | P1 | 150-fixture point estimate is noise (P10) | point-estimate gate |
| 6 | CEO | NaN features instead of dropping rows (C-4) | Mechanical | P1 | promoted clubs must be predicted | drop rows |
| 7 | CEO | Local editable install (C-5) | Mechanical | P5 | no git remote (P5) | URL dependency now |
| 8 | CEO | fd.org optional (C-6) | Mechanical | P1 | no token yet (P8) | hard dependency |
| 9 | CEO | ClubElo disk cache (C-7) | Mechanical | P1 | outage observed (P7) | no fallback |
| 10 | CEO | README cost note (C-8) | Mechanical | P1 | pay-per-use pricing (P11) | none |
| 11 | CEO | Accept track record (C-9) | Mechanical | P2 | in blast radius, S | — |
| 12 | CEO | Odds normalisation (C-10) | Mechanical | P1 | column names change by season (P4) | — |
| 13 | CEO | Defer proposals 2-8 to TODOS.md | Mechanical | P3 | outside blast radius or need evidence first | add now |
| 14 | CEO | Skip markdown output, form string | Mechanical | P3 | low value | add |
| 15 | CEO | Local Elo fallback deferred | Taste | P5 | cache handles outages; second Elo source adds complexity | build fallback now |
| 16 | CEO | Spec review 1 (6/10): apply all findings as C-11..C-23 + 5 baseline edits | Mechanical | P1, P5 | each finding is an in-scope gap with one clear fix | leave gaps for eng phase |
| 17 | CEO | Stage-2 eligibility by explicit `FPP_SENTIMENT_SINCE` date (C-14) | Taste | P5 | explicit setting over an earliest-tweet heuristic | infer from earliest tweet |
| 18 | CEO | Spec review 2 (7/10): apply all findings as C-24..C-29, revise C-12/C-13/C-19 | Mechanical | P1, P5 | each finding is an in-scope gap with one clear fix | defer to eng |
| 19 | CEO | Fixture key (season, home, away) instead of kickoff date (C-12) | Mechanical | P5 | survives rescheduling with no special case | date key + reschedule handling |
| 20 | CEO | Tweet volume removed from model inputs (C-13) | Taste | P5 | volume is dominated by the collection cap | keep volume features |
| 21 | CEO | Spec review 3 (7/10, cap reached): apply all 7 findings as C-30..C-36, self-verify | Mechanical | P1, P5 | in-scope gaps with one clear fix; loop cap of 3 launches reached | defer to eng |
| 22 | CEO | Document approval (0H) → A, approve | Mechanical | autoplan | plan and summary reflect all decisions; 7 final fixes not re-verified | revise |
| 23 | CEO | Collect first (C-37) | Mechanical | P1 | sentiment cannot be backfilled | build model first |
| 24 | CEO | Accept `fpp watch` (C-38), reversing deferral #3 | Taste | P1, P2 | manual 60-min windows make n small and biased | keep deferred |
| 25 | CEO | Market arm + reframed headline question (C-39) | Taste | P1 | sentiment beating a weak stage 1 can just be recovered team news | stage-1 comparison only |
| 26 | CEO | Stage 2 as offline study with power analysis (C-40) | Taste | P5, P1 | 150-fixture CI gate is underpowered; keeps live path simple | live stage 2 at 150 |
| 27 | CEO | Cost to decision + stop rule (C-41) | Mechanical | P1 | open-ended spend with no decision point | cost note only |
| 28 | CEO | Lead with log loss/Brier; market gap criterion (C-42) | Mechanical | P1 | accuracy misleads for H/D/A | accuracy first |
| 29 | CEO | Pre-match odds as fair comparator (C-43) | Mechanical | P1 | closing odds post-date T | closing only |
| 30 | CEO | Missing Time handling (C-44) | Mechanical | P1 | old seasons lack Time | — |
| 31 | CEO | Understat xG + Dixon-Coles to TODOS (C-45) | Mechanical | P3 | outside blast radius | add now |
| 32 | CEO | Build inside pl-predictor instead of the fork | Gate item | P6 | both native passes favour it; user stated "build on FPP"; Codex unavailable so not a formal User Challenge | auto-switch |
| 33 | CEO | S1: single pl-predictor adapter (C-46) | Mechanical | P5 | contain coupling to a non-public API | import internals everywhere |
| 34 | CEO | S1/S4: single writer + lock (C-47) | Mechanical | P1 | SQLite write contention | none |
| 35 | CEO | S1: watch refresh/reschedule (C-48) | Mechanical | P1 | kickoffs move | static schedule |
| 36 | CEO | S2/S4: explicit rescue paths (C-49) | Mechanical | P1 | 4 unrescued errors + 3 unhandled edges | leave to implementer |
| 37 | CEO | S3: gitignore + pickle warning (C-50) | Mechanical | P1 | secret and pickle risks | — |
| 38 | CEO | S5: layout adds study/watch/adapter (C-51) | Mechanical | P5 | keep modules small | fold into existing modules |
| 39 | CEO | S6: injected clock + CLI e2e + golden file (C-52) | Mechanical | P1 | time-dependent tests flake | — |
| 40 | CEO | S8: logging + watch log (C-53) | Mechanical | P1 | missed slots must be reconstructable | — |

### Phase skips

- Phase 2 (Design): skipped, no UI scope detected (uiRequired=false).
- Phase 2.5 (DX): skipped at the user's request ("skip to approval gate after CEO review"); DX scope was detected (dxRequired=true), so this is missing coverage, not a clean result.
- Phase 3 (Eng): skipped at the user's request; the required Eng review has not run.
