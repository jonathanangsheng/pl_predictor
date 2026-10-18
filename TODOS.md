# TODOS

Deferred from the /autoplan CEO review (docs/predict-plan-review.md, 2026-10-18).

## Pre-match odds as a stage-1 feature
- **What:** add de-margined pre-match odds to the stage-1 feature set.
- **Why:** the market is the strongest single predictor; partly covered by the stage-2 market arm (C-39).
- **Effort:** human M / CC S. **Priority:** P3. **Depends on:** `pl-predictor model study` results.

## Predictions in the pl-predictor dashboard
- **What:** show history.csv probabilities and scores in pl-predictor's Dash app.
- **Why:** one place to see sentiment and predictions per fixture.
- **Effort:** human M / CC S. **Priority:** P3.

## FPL team xG as a current-season feature
- **What:** sum player xG per fixture from FPL `event/{gw}/live`.
- **Why:** xG beats shot counts, but FPL only covers the current season, so there is no training history.
- **Effort:** human M / CC S. **Priority:** P3.

## Per-fixture explanations
- **What:** show the top feature contributions for each prediction.
- **Why:** helps trust and debugging.
- **Effort:** human M / CC S. **Priority:** P3.

## Understat xG history
- **What:** xG features from Understat (PL since 2014/15) via the soccerdata package.
- **Why:** rolling xG usually beats shots; rejected for now because it relies on unofficial scraping with terms and stability risk.
- **Effort:** human M / CC S. **Priority:** P2.

## Dixon-Coles / Poisson goal model
- **What:** add a Poisson goal model as a stage-1 candidate.
- **Why:** standard, well calibrated, and better at draw probabilities than the current classifiers.
- **Effort:** human M / CC S. **Priority:** P2.

## Live market odds for upcoming fixtures
- **What:** `FootballDataCoUkHook.upcoming()` reading football-data.co.uk `fixtures.csv`, logged next to each prediction.
- **Why:** lets the live record compare against the market before results are published; history already has pre-match odds.
- **Effort:** human S / CC S. **Priority:** P3.

## Verify ClubElo names
- **What:** check `config/club_names.csv` ClubElo names against a successful pull (ClubElo returned 502 from the dev machine, so local Elo was used).
- **Why:** a wrong name falls back to local Elo for every club.
- **Effort:** human S / CC S. **Priority:** P2.
