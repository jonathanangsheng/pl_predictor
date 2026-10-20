# TODOS

Deferred work: from the /autoplan CEO review (docs/predict-plan-review.md, 2026-10-18) and later sessions. Each item says what it is waiting on.

## Pre-match odds as a stage-1 feature
- **What:** add de-margined pre-match odds to the stage-1 feature set.
- **Why:** the market is the strongest single predictor; partly covered by the stage-2 market arm (C-39).
- **Effort:** human M / CC S. **Priority:** P3. **Depends on:** `pl-predictor model study` results.

## xG form features from football-data.co.uk
- **What:** 10-match rolling xG for/against differences, from the `home_xg`/`away_xg` columns the match table now carries (football-data.co.uk `HxG`/`AxG`, published from 2026/27).
- **Why:** rolling xG usually beats shot counts; no new source or scraping, and the source keeps each season's xG, so history builds itself.
- **Effort:** human S / CC S. **Priority:** P2. **Depends on:** enough seasons with xG to walk-forward test (at least 2026/27 and 2027/28 complete), then keep only if it beats the current model.

## Understat xG history
- **What:** xG features from Understat (PL since 2014/15) via the soccerdata package.
- **Why:** rolling xG usually beats shots; rejected for now because it relies on unofficial scraping with terms and stability risk.
- **Effort:** human M / CC S. **Priority:** P2. **Depends on:** your decision to accept unofficial scraping.

## Fixture congestion across all competitions
- **What:** days since each club's last competitive match and matches in the last 14 days, counting FA Cup, League Cup and European games, as stage-1 features.
- **Why:** midweek European and cup games mean less rest and more rotation. The earlier rest-days feature (rejected in `features.py`) counted Premier League matches only, so it missed exactly these games: it was not a fair test.
- **Data (checked 2026-10-20):**
  - football-data.org free tier: Champions League only (no Europa/Conference League, FA Cup or League Cup); free history depth not stated.
  - API-Football: all competitions; the free plan (100 requests/day) is limited in seasons (exact range unconfirmed; one report says 2022-2024); paid Pro reported at about $15-19/month.
  - FBref: schedules for all competitions, but only by scraping (it also lost its Opta advanced stats in early 2026).
- **Effort:** human M / CC S. **Priority:** P3. **Depends on:** a source with full cup and European schedules for many seasons (likely a paid API, or accepting FBref scraping); then keep only if walk-forward testing shows it helps.
