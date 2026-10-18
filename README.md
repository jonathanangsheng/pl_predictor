# pl-predictor

Fan sentiment analysis for English Premier League fixtures.

For each fixture, the pipeline pulls tweets posted around kickoff from the X (Twitter) API with **Tweepy**, attributes each tweet to the home club, the away club or both, scores it with a Twitter-tuned **Hugging Face** RoBERTa model, and shows how the mood moves through the match in a **Plotly Dash** dashboard.

```
config/fixtures.csv ─┐
config/teams.yaml ───┤
                     ▼
        ┌──────────────────────┐   search_recent_tweets   ┌─────────┐
        │ collector (Tweepy)   │ ───────────────────────▶ │  X API  │
        └──────────┬───────────┘                          └─────────┘
                   │ tweets + side (home/away/both) + phase (pre/live/post)
                   ▼
        ┌──────────────────────┐
        │  SQLite (data/*.db)  │◀──── sentiment (Hugging Face pipeline)
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Plotly Dash app     │  timeline · volume · mix · club ranking · top tweets
        └──────────────────────┘
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

### Try it without API access

`demo` fills a separate database (`data/demo.db`) with synthetic, already-scored tweets for every fixture in `config/fixtures.csv`. No X token and no model download needed.

```bash
pl-predictor demo
pl-predictor dashboard --db data/demo.db
```

Open http://127.0.0.1:8050.

### Collect real data

1. Put your X API bearer token in `.env` as `X_BEARER_TOKEN`. The recent-search endpoint needs a paid tier (Basic or above). It only reaches back **7 days**.
2. Add the fixtures you care about to `config/fixtures.csv`. Kickoff times are UTC.
3. Run a collection and scoring pass:

```bash
pl-predictor fixtures                 # which fixtures are collectable right now
pl-predictor query --fixture mw08-avl-mci   # preview the search query
pl-predictor run                      # collect all collectable fixtures, then score
pl-predictor run --watch 300          # repeat every 5 minutes during a matchday
```

The first `run` downloads the model (~500 MB) into the Hugging Face cache.

## Commands

| Command | What it does |
|---|---|
| `fixtures` | List configured fixtures; `*` marks those inside the 7-day search window |
| `query [--fixture ID]` | Print the X search query built for each fixture |
| `run [--fixture ID ...] [--watch SECONDS]` | Collect tweets and score anything unscored |
| `score` | Score tweets that were collected but not scored (e.g. if a run failed at scoring) |
| `demo [--tweets N] [--db PATH]` | Seed synthetic data into `data/demo.db` for offline use |
| `dashboard [--db PATH] [--port 8050] [--debug]` | Start the Dash app (refreshes every 60s) |

## How it works

**Collection** ([collector.py](pl_predictor/collector.py), [fixtures.py](pl_predictor/fixtures.py))
- Each fixture has a window from `PLS_PRE_MATCH_MINUTES` before kickoff to `PLS_POST_MATCH_MINUTES` after an estimated full time (kickoff + 115 min).
- The query combines the matchday hashtag (`#ARSMCI`), club hashtags, official handles and nicknames from `teams.yaml`, then adds `-is:retweet lang:en`. Home and away terms are interleaved and trimmed so the query fits your tier's limit (`PLS_MAX_QUERY_LENGTH`).
- The first run for a fixture splits the window into 8 equal slices and spends an equal share of `PLS_MAX_TWEETS_PER_FIXTURE` on each. Recent search returns the newest tweets first, so without the split a run after full time would only collect post-match tweets.
- Later runs are incremental: they pass the newest stored tweet ID as `since_id`. `PLS_MAX_TWEETS_PER_FIXTURE` caps each run to protect your monthly read quota. If a search fails partway through, the tweets already read are still saved.
- Once a fixture's window has closed and been searched without errors, it is marked complete and skipped. This keeps `--watch` from spending requests on finished matches.

**Attribution.** A tweet is tagged `home`, `away`, `both` or `neither` by matching club terms on word boundaries. Tweets tagged `both` count towards both clubs. Tweets that only use the matchday hashtag stay `neither`: they count towards fixture volume but not towards either club's sentiment.

**Sentiment** ([sentiment.py](pl_predictor/sentiment.py))
- Default model: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest). It returns negative, neutral and positive probabilities.
- Mentions become `@user` and links become `http`, which is how the model was trained.
- **Net sentiment** = P(positive) − P(negative), from −1 to +1. It is averaged per time bucket in the dashboard.
- Any 3-class model with negative, neutral and positive (or `LABEL_0..2`) labels works through `PLS_MODEL_NAME`.

**Storage** ([storage.py](pl_predictor/storage.py)). Uses SQLite with three tables: `fixtures`, `tweets` (keyed by fixture and tweet ID) and `sentiment` (keyed by tweet ID, so a tweet is scored once).

**Dashboard** ([dashboard.py](pl_predictor/dashboard.py)). For the selected fixture it shows stat tiles, net sentiment over time per club with kickoff and full-time markers, stacked tweet volume, the sentiment mix per club, a club-mood ranking across every stored fixture and the most-engaged tweets.

## Football data hooks

[hooks.py](pl_predictor/hooks.py) has one small client per data source. Each returns pandas DataFrames:

| Hook | Source | Methods | Key |
|---|---|---|---|
| `FootballDataCoUkHook` | football-data.co.uk | `results(season)`: results, shots, corners, cards, xG and betting odds per match, back to 1993 | none |
| `FPLHook` | Fantasy Premier League API | `teams()`, `players()`, `gameweeks()`, `fixtures(gameweek, future)`, `player_history(id)`, `live(gameweek)` | none |
| `ClubEloHook` | clubelo.com | `ratings(on)`, `club_history(club)`: Elo strength ratings | none |
| `FootballDataOrgHook` | football-data.org | `matches(season)`, `standings(season)`, `teams(season)` | `FOOTBALL_DATA_ORG_TOKEN` |

```python
from pl_predictor.hooks import FootballDataCoUkHook, FPLHook

results = FootballDataCoUkHook().results(2026)  # 2026 = the 2026/27 season
next_gw = FPLHook().fixtures(future=True)
```

Seasons are passed as their starting year. The FPL API is unofficial and covers the current season only. It also leaves its team table fields (played, points, etc.) at 0, so use `fixtures()` or `FootballDataOrgHook.standings()` for the table.

## Match predictions

The predictor answers one question: **does pre-match fan sentiment on X carry information beyond the betting market?** It predicts home/draw/away probabilities for every upcoming Premier League fixture, logs them before kickoff, scores them afterwards, and runs an offline study of whether sentiment helps. The full design is in [docs/predict-plan.md](docs/predict-plan.md).

```bash
pl-predictor model evaluate    # walk-forward over the last 3 completed seasons
pl-predictor model train       # fit on all history, save models/model.joblib
pl-predictor model predict     # next gameweek; appends to predictions/history.csv
pl-predictor model score       # live record: latest pre-kickoff prediction per fixture
pl-predictor model study       # stage-2 sentiment study with a power estimate
pl-predictor watch             # per fixture at kickoff - 55 min: collect, score tweets, predict
```

**How it works.** Matches since 2005/06 come from football-data.co.uk. Upcoming fixtures come from the FPL API. Team strength is ClubElo, or a local Elo computed from results when ClubElo is unreachable; the model records which source it used. Features use only matches before each fixture: Elo difference plus 10-match form differences in goals, shots and shots on target. The model is a multinomial logistic regression. Gradient boosting, a calibration wrapper, corners, fouls, table position and rest days were tested and all made log loss worse.

**Walk-forward results** (log loss, lower is better; 380 matches per season):

| Predictor | 2023/24 | 2024/25 | 2025/26 | Mean |
|---|---|---|---|---|
| Closing odds (ceiling) | 0.9007 | 0.9667 | 1.0118 | 0.9597 |
| Pre-match odds (market) | 0.9093 | 0.9706 | 1.0153 | 0.9650 |
| **Model** | **0.9242** | **0.9877** | **1.0252** | **0.9790** |
| Elo only | 0.9271 | 0.9899 | 1.0270 | 0.9814 |

The model beats Elo every season and is well calibrated, but the market is better. That is expected: the point is to test whether sentiment adds anything the market misses.

**Sentiment (stage 2).** Tweets count if they were posted between kickoff − 24 h and T = kickoff − 60 min. Stage 2 is tested offline by `model study` on two arms: model + sentiment vs model, and market + sentiment vs market. It only turns on if a paired bootstrap 95% CI of the log-loss change is entirely below zero, and then only when you set `PLS_STAGE2=1`. The power estimate is sobering: about 490 fixtures to detect a 0.01 improvement, about 1,970 for 0.005. Decide a stop rule before collecting: a fixture count, a spend cap or the end of the season, whichever comes first.

**Collecting for the study.**
1. In `.env`, set `PLS_PRE_MATCH_MINUTES=1440`, `PLS_MAX_TWEETS_PER_FIXTURE=400` and `PLS_SENTIMENT_SINCE=<first day you collect>`.
2. Run `pl-predictor watch`, for example from Task Scheduler or cron at login. Don't also run `pl-predictor run --watch`.
3. Only fixtures whose first collection happened in the hour before kickoff count, because the first run spreads its budget across the 24 h window.

X API reads cost about $0.005 each (third-party figure, unconfirmed), so 400 tweets × 10 fixtures a week is roughly $80 a month.

`models/model.joblib` is a pickle. Only load files this tool wrote. The form-feature idea comes from [Football_Prediction_Project](https://github.com/mhaythornthwaite/Football_Prediction_Project) (MIT).

## Project layout

```
config/
  teams.yaml          clubs, hashtags, handles, keywords (2026/27 clubs - update each season)
  fixtures.csv        fixtures to track (GW7-8 of 2026/27 from the FPL API - add more as the season goes)
  club_names.csv      club names per data source since 2005/06
pl_predictor/         one flat package
  config.py           settings from env / .env
  teams.py            Team model + text matching
  fixtures.py         Fixture model, windows, phases, query builder
  collector.py        Tweepy recent-search client
  sentiment.py        Hugging Face scorer
  storage.py          SQLite store
  pipeline.py         collect -> store -> score orchestration
  demo.py             synthetic data seeder
  dashboard.py        Dash app (CSS inline)
  figures.py          dashboard figures
  hooks.py            football-data.co.uk, FPL, ClubElo, football-data.org clients
  matches.py          match table since 2005/06, odds normalisation, caches
  elo.py              ClubElo ratings with local-Elo fallback
  features.py         as-of form features
  model.py            stage-1 model, baselines, walk-forward evaluation
  predict.py          next-gameweek predictions, history.csv, scoring
  study.py            stage-2 sentiment study and power estimate
  watch.py            per-kickoff collect/score/predict loop
  model_commands.py   `pl-predictor model ...` commands
  cli.py              `pl-predictor` command
tests/                pytest suite (no network, no model download)
```

## Development

```bash
pytest
ruff check pl_predictor tests
```

The tests use a fake Tweepy client and a fake sentiment pipeline, so they run offline and don't need `torch`.

## Limitations and next steps

- **Sarcasm and banter.** "Great defending again 🙃" often scores positive. Consider fine-tuning on labelled football tweets.
- **Attribution is keyword-based.** Rival-fan tweets ("Spurs bottled it again" from an Arsenal fan) count towards the club named. Author-level club affinity, inferred from bio or history, would improve this.
- **Fixture list is manual.** A loader for an official fixtures feed or football-data.org could fill `fixtures.csv` automatically, and could add goal events to annotate the timeline.
- **Full time is estimated** (kickoff + 115 min). Swap in real match events when available.
- **Volume is limited by your X API tier.** Large fixtures will hit `PLS_MAX_TWEETS_PER_FIXTURE` quickly, so treat the results as a sample, not a census.
- Follow X's developer terms on storing and displaying tweet content.

## Project status (2026-10-18)

The code is essentially complete; the headline question (does fan sentiment add information?) is still unanswered because no tweets have been collected.

| # | Plan step | Status |
|---|---|---|
| 1 | Phase 0: collection settings and scheduler | Settings in `.env`; collection not started; `pl-predictor watch` replaces the scheduler |
| 2 | Club names and match table | Done (`matches.py`, `config/club_names.csv`) |
| 3 | Features | Done |
| 4 | Stage 1 model (`evaluate`, `train`) | Done: log loss 0.9790 vs Elo 0.9814 (lower is better) |
| 5 | `predict` and `score` | Done |
| 6 | `watch` | Done |
| 7 | Stage 2 study | Study code done; `FootballDataCoUkHook.upcoming()` deferred to TODOS.md |
| 8 | README, TODOS, .gitignore | Done |

Not done yet:

- **No sentiment data.** Needs `X_BEARER_TOKEN` in `.env`, `watch` running for most of a season, and roughly $80/month of X API reads (third-party figure, unconfirmed).
- **Stage 2 is off** until the study shows sentiment helps.
- **DX and engineering plan reviews** were not run.
- **Optional extras** are parked in [TODOS.md](TODOS.md).
- **ClubElo is unreachable** from this network (502), so the model uses a locally computed Elo.

Stage 1 beats Elo alone but trails the betting market (0.9650), so on its own it gives no edge over the odds.
