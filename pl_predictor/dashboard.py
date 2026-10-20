"""Plotly Dash app: a FIFA-style hub, fixture sentiment, predictions and their track record."""

from __future__ import annotations

import functools
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import dash_ag_grid as dag
import pandas as pd
import requests
from dash import Dash, Input, Output, State, dcc, html

from . import figures
from .config import get_settings
from .demo import DEMO_MODEL
from .hooks import FPLHook
from .matches import SOURCE_ERRORS, fpl_matches
from .model_commands import load_results
from .predict import MARKET_P, MODEL_P, latest_per_fixture, read_history, score
from .storage import Store
from .teams import load_teams

REFRESH_MS = 60_000
CACHE_S = 15 * 60  # FPL fixtures and results are re-downloaded at most this often
HEAD_FONT = "'Barlow Condensed', system-ui, sans-serif"
TAB = {
    "border": "none",
    "background": "transparent",
    "padding": "4px 0",
    "marginRight": "40px",
    "flex": "0 0 auto",
    "fontFamily": HEAD_FONT,
    "fontSize": "26px",
    "fontWeight": 500,
    "textTransform": "uppercase",
    "color": "#6b6e70",
}
TAB_ON = {**TAB, "color": "#0b0b0b", "fontWeight": 700}

CSS = """
:root {
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --border: rgba(11, 11, 11, 0.10);
  --font: system-ui, -apple-system, "Segoe UI", sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  /* sky, stand, pitch: a nod to the FIFA hub without its stadium photo */
  background: linear-gradient(180deg, #e7eaec 0%, #f2f3f4 38%, #cdd7cc 60%, #4f8a3a 80%,
    #2d5e22 100%) fixed;
  color: var(--ink);
  font-family: var(--font);
}

.page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px 16px 48px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.header h1 { margin: 0; font-size: 24px; font-weight: 650; }
.subtitle { margin: 4px 0 0; color: var(--ink-2); font-size: 14px; }

.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-end;
}
.control { display: flex; flex-direction: column; gap: 6px; }
.control.grow { flex: 1 1 360px; }
.control label { font-size: 12px; font-weight: 600; color: var(--ink-2); }
.radio label { margin-right: 12px; font-size: 14px; font-weight: 400; color: var(--ink); }
.radio input { margin-right: 4px; }

.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}
.tile {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
}
.tile-label { font-size: 13px; color: var(--ink-2); }
.tile-value { font-size: 30px; font-weight: 650; margin: 4px 0 2px; }
.tile-caption { font-size: 12px; color: var(--muted); }

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 8px 12px;
  min-width: 0;
}
.card h2 { font-size: 15px; font-weight: 600; margin: 10px 0 12px; }

.tab-body { display: flex; flex-direction: column; gap: 16px; padding-top: 16px; }
.note { margin: 0; color: var(--ink-2); font-size: 14px; }
.banner {
  background: #fff4d6;
  border: 1px solid #e8c766;
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 14px;
}

.hub { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
.hub-tile {
  background: rgba(236, 238, 239, 0.93);
  padding: 12px 14px;
  min-height: 150px;
  min-width: 0;
}
.hub-tile.wide { grid-column: span 2; }
.hub-tile.dark { background: #1b2328; color: #fff; }
.hub-tile h2 {
  font-family: "Barlow Condensed", var(--font);
  text-transform: uppercase;
  font-size: 30px;
  font-weight: 600;
  line-height: 1;
  color: #6b6e70;
  margin: 0 0 8px;
}
.hub-tile.dark h2 { color: #fff; font-size: 40px; }
@media (max-width: 900px) {
  .hub { grid-template-columns: 1fr; }
  .hub-tile.wide { grid-column: auto; }
}
.hub-sub { margin: 4px 0; color: #5d6063; font-size: 14px; }
.dark .hub-sub { color: #aab4b8; }
.disc {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: none;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  font-size: 10px;
  font-weight: 700;
}
.disc.big {
  width: 64px;
  height: 64px;
  font-size: 16px;
  box-shadow: 0 0 0 3px #fff, 0 1px 4px rgba(0, 0, 0, 0.3);
}
.club-card {
  display: flex;
  align-items: center;
  gap: 14px;
  width: min(100%, 420px);
  padding: 8px 18px 8px 8px;
  border-radius: 40px 8px 8px 40px;
  background: rgba(255, 255, 255, 0.8);
}
.club-pick { flex: 1; min-width: 0; }
.club-pick label { font-size: 12px; font-weight: 600; color: var(--ink-2); }
.match-line { display: flex; justify-content: space-between; align-items: center; font-size: 15px; }
.prob { display: flex; height: 10px; border-radius: 5px; overflow: hidden; margin: 10px 0 4px; }
.prob-labels {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: var(--ink-2);
}
.ll-row {
  display: flex;
  justify-content: space-between;
  padding: 6px 8px;
  font-size: 16px;
  border-bottom: 1px solid var(--border);
}
.ll-row.best { background: #b9ef9f; font-weight: 650; }
.mood-pair { display: flex; flex-wrap: wrap; gap: 24px; margin: 10px 0; }
.mood { display: flex; align-items: center; gap: 10px; font-size: 15px; }
.dots { display: flex; justify-content: flex-end; gap: 8px; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: #c3c6c8; }
.dot.on { background: #5d6063; }
.feed-row {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
  font-size: 14px;
}
.feed-text {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.league { max-height: 320px; overflow-y: auto; overflow-x: hidden; font-size: 14px; }
.lg-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lg-row {
  display: grid;
  grid-template-columns: 18px 28px minmax(0, 1fr) 24px 26px 36px;
  align-items: center;
  gap: 4px;
  padding: 3px 4px;
}
.lg-head { color: #5d6063; font-size: 13px; }
.lg-row.top { background: #b9ef9f; }
.lg-row.ucl { background: #cfe0ee; }
.lg-row.mine { outline: 2px solid #0b0b0b; outline-offset: -2px; font-weight: 650; }
"""

INDEX = (
    "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&display=swap">'
    f"<style>{CSS}</style></head><body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}"
    "{%renderer%}</footer></body></html>"
)
GRAPH_CONFIG = {"displayModeBar": False}
NO_HISTORY = "No logged predictions yet: run `pl-predictor model predict`."
NO_TWEETS = "No tweets yet: run `pl-predictor run` or `demo`."

# Concurrent callbacks wait for one download instead of each starting their own. One lock per
# cache, so a slow results download doesn't hold up the FPL-backed tiles.
_FPL_LOCK = threading.Lock()
_RESULTS_LOCK = threading.Lock()


# A failed fetch raises, so only successful downloads are cached.
@functools.lru_cache(maxsize=1)
def _fpl_cached(_bucket: int) -> pd.DataFrame:
    hook = FPLHook(timeout=10)
    return fpl_matches(hook.teams(), hook.fixtures())


@functools.lru_cache(maxsize=1)
def _results_cached(_bucket: int) -> pd.DataFrame:
    started = bool(_fpl()["started"].any())
    return load_results(get_settings(), datetime.now(timezone.utc), season_started=started)


def _fpl() -> pd.DataFrame:
    with _FPL_LOCK:
        return _fpl_cached(int(time.time() // CACHE_S))


def _results() -> pd.DataFrame:
    with _RESULTS_LOCK:
        return _results_cached(int(time.time() // CACHE_S))


def _frames(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fixtures, scored tweets) from the sentiment database."""
    store = Store(db_path)
    try:
        return store.fixtures_frame(), store.tweets_frame()
    finally:
        store.close()


def _scored(history: pd.DataFrame) -> tuple[dict, int, int] | str:
    """(metrics by predictor, fixtures scored, fixtures pending), or a message to show instead."""
    if history.empty:
        return NO_HISTORY
    try:
        results = _results()
    except SOURCE_ERRORS as exc:
        return f"Could not load results ({exc})."
    summary, done, pending = score(history, results)
    return summary, len(done), pending


def _colour(code: str, teams: dict) -> str:
    return teams[code].colour if code in teams else figures.MUTED


def _name(code: str, teams: dict) -> str:
    return teams[code].name if code in teams else code


def _sub(text: str) -> html.P:
    return html.P(text, className="hub-sub")


def _disc(code: str, teams: dict, size: str = "") -> html.Span:
    colour = _colour(code, teams)
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    ink = "#0b0b0b" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#ffffff"
    return html.Span(
        code or "–", className=f"disc {size}", style={"background": colour, "color": ink}
    )


def _hub(title: str, *children, extra: str = "") -> html.Div:
    return html.Div(className=f"hub-tile {extra}", children=[html.H2(title), *children])


def _next_match_tile(match: pd.Series | None, club: str, history: pd.DataFrame, teams: dict):
    if match is None:
        return _sub("No upcoming fixture.")
    home, away = match["home"], match["away"]
    opponent = away if home == club else home
    children = [
        html.Div(
            className="match-line",
            children=[html.Span("League match"), _disc(opponent, teams, "big")],
        ),
        _sub(
            f"{'Home' if home == club else 'Away'} v {_name(opponent, teams)} · "
            f"{match['kickoff']:%a %d %b %H:%M} UTC"
        ),
    ]
    upcoming = latest_per_fixture(history[history["kickoff"] > pd.Timestamp.now(tz="UTC")])
    logged = upcoming[(upcoming["home"] == home) & (upcoming["away"] == away)]
    if logged.empty:
        return [*children, _sub(NO_HISTORY)]
    p = logged.iloc[-1]
    parts = [
        (home, p["p_home"], _colour(home, teams)),
        ("Draw", p["p_draw"], figures.OTHER),
        (away, p["p_away"], _colour(away, teams)),
    ]
    return [
        *children,
        html.Div(
            className="prob",
            children=[
                html.Div(style={"width": f"{share:.0%}", "background": c}) for _, share, c in parts
            ],
        ),
        html.Div(
            className="prob-labels",
            children=[html.Span(f"{who} {share:.0%}") for who, share, _ in parts],
        ),
        *([_sub(p["why"])] if pd.notna(p["why"]) else []),
        _sub(
            "Market: no odds listed yet"
            if pd.isna(p[MARKET_P[0]])
            else "Market: "
            + " · ".join(
                f"{who} {p[col]:.0%}" for (who, _, _), col in zip(parts, MARKET_P, strict=True)
            )
        ),
    ]


def _record_tile(history: pd.DataFrame):
    scored = _scored(history)
    if isinstance(scored, str):
        return _sub(scored)
    summary, _, pending = scored
    n = summary["model"]["n"]
    if not n:
        return _sub(f"No finished predictions with odds yet; {pending} waiting for results.")
    rows = {
        "Model": summary["model"],
        "Elo": summary["elo"],
        "Market": summary["market (pre-match odds)"],
    }
    best = min(m["log_loss"] for m in rows.values())
    return [
        *(
            html.Div(
                className="ll-row best" if m["log_loss"] == best else "ll-row",
                children=[html.Span(who), html.Span(f"{m['log_loss']:.3f}")],
            )
            for who, m in rows.items()
        ),
        _sub(f"Log loss on {n} finished fixtures; lower is better."),
    ]


def _league_tile(table: list[dict], club: str, teams: dict, xg: pd.DataFrame):
    """Points table, with xGD (season xG difference) for clubs that have xG."""
    xgd = xg["xgd"].to_dict()
    head = html.Div(
        className="lg-row lg-head",
        children=[
            *(html.Span("") for _ in range(3)),  # position, disc, club name
            html.Span("PLD"),
            html.Span("PTS"),
            html.Span("xGD", title="Season xG difference (context; not a model input)"),
        ],
    )
    rows = [
        html.Div(
            className="lg-row"
            + (" top" if r["pos"] == 1 else " ucl" if r["pos"] <= 4 else "")
            + (" mine" if r["club"] == club else ""),
            children=[
                html.Span(r["pos"]),
                _disc(r["club"], teams),
                html.Span(_name(r["club"], teams), className="lg-name"),
                html.Span(r["played"]),
                html.Span(r["points"]),
                html.Span(f"{xgd[r['club']]:+.1f}" if r["club"] in xgd else ""),
            ],
        )
        for r in table
    ]
    return [head, *rows]


def _xg_line(match: pd.Series | None, xg: pd.DataFrame):
    """Each side's recent xG for-against per match, if both have xG this season."""
    if match is None or not {match["home"], match["away"]} <= set(xg.index):
        return None
    sides = [
        f"{club} {xg.at[club, 'xg_for']:.1f}-{xg.at[club, 'xg_against']:.1f}"
        for club in (match["home"], match["away"])
    ]
    return _sub(f"Recent xG for-against per match: {', '.join(sides)}")


def _feed_tile(tweets: list[dict], teams: dict):
    if not tweets:
        return _sub(NO_TWEETS)
    return [
        html.Div(
            className="feed-row",
            children=[
                _disc(t["club"], teams),
                html.Div([html.Div(t["text"], className="feed-text"), _sub(f"♥ {t['likes']}")]),
            ],
        )
        for t in tweets
    ]


def _mood(code: str, net: float | None, teams: dict) -> html.Div:
    if net is None:
        word = "no tweets"
    else:
        word = f"{net:+.2f} " + ("upbeat" if net > 0.1 else "gloomy" if net < -0.1 else "mixed")
    return html.Div(className="mood", children=[_disc(code, teams), html.Span(word)])


def _fixture_option(row: pd.Series) -> dict:
    return {
        "label": f"MW{row.matchweek} · {row.home_name} v {row.away_name} · "
        f"{row.kickoff:%a %d %b %H:%M} UTC",
        "value": row.fixture_id,
    }


def _tile(label: str, value: str, caption: str) -> html.Div:
    return html.Div(
        className="tile",
        children=[
            html.Div(label, className="tile-label"),
            html.Div(value, className="tile-value"),
            html.Div(caption, className="tile-caption"),
        ],
    )


def _card(*children) -> html.Div:
    return html.Div(className="card", children=list(children))


def _graph(graph_id: str) -> dcc.Graph:
    return dcc.Graph(id=graph_id, config=GRAPH_CONFIG)


def _col(field: str, header: str, flex: int = 1, **extra) -> dict:
    return {"field": field, "headerName": header, "flex": flex, **extra}


def _grid(grid_id: str, columns: list[dict]) -> dag.AgGrid:
    """Read-only table that grows to fit its rows."""
    return dag.AgGrid(
        id=grid_id,
        columnDefs=columns,
        rowData=[],
        defaultColDef={"sortable": True, "resizable": True, "minWidth": 72},
        dashGridOptions={"domLayout": "autoHeight", "animateRows": False},
        className="ag-theme-alpine",
        style={"width": "100%"},
    )


def _central_tab(teams: dict) -> list:
    clubs = sorted(teams.values(), key=lambda t: t.name)
    return [
        html.Div(
            className="club-card",
            children=[
                html.Div(id="club-disc"),
                html.Div(
                    className="club-pick",
                    children=[
                        html.Label("Your club", htmlFor="club"),
                        dcc.Dropdown(
                            id="club",
                            options=[{"label": t.name, "value": t.code} for t in clubs],
                            value=clubs[0].code if clubs else None,
                            clearable=False,
                            persistence=True,
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="hub",
            children=[
                _hub("Next match", html.Div(id="next-match"), html.Div(id="next-xg")),
                _hub("Model vs market", html.Div(id="vs-market")),
                html.Div(id="preview", className="hub-tile wide"),
                _hub(
                    "Mood map",
                    _sub("Mean net sentiment per club across stored fixtures"),
                    _graph("mood-map"),
                    extra="wide dark",
                ),
                _hub("Top tweets", html.Div(id="top-tweets")),
                _hub("Premier League", html.Div(id="league", className="league")),
            ],
        ),
        dcc.Store(id="previews"),
        dcc.Interval(id="carousel", interval=6000),
    ]


def _fixture_tab() -> list:
    return [
        html.Div(
            className="controls",
            children=[
                html.Div(
                    className="control grow",
                    children=[
                        html.Label("Fixture", htmlFor="fixture"),
                        dcc.Dropdown(id="fixture", clearable=False),
                    ],
                ),
                html.Div(
                    className="control",
                    children=[
                        html.Label("Bucket"),
                        dcc.RadioItems(
                            id="bucket",
                            options=[{"label": f"{m} min", "value": m} for m in (5, 10, 15)],
                            value=10,
                            inline=True,
                            className="radio",
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="kpis", className="tiles"),
        _card(html.H2(id="timeline-title"), _graph("timeline")),
        _card(html.H2("Tweet volume"), _graph("volume")),
        _card(html.H2("Sentiment mix by club"), _graph("mix")),
        _card(
            html.H2("Most-engaged tweets"),
            _grid(
                "tweets",
                [
                    _col("time", "Time"),
                    _col("club", "Club", 2, minWidth=120),
                    _col("label", "Label", minWidth=90),
                    _col("score", "Net", type="numericColumn"),
                    _col("likes", "Likes", type="numericColumn"),
                    _col(
                        "text",
                        "Tweet",
                        5,
                        minWidth=240,
                        wrapText=True,
                        autoHeight=True,
                        cellStyle={"lineHeight": "20px", "padding": "8px 16px"},
                    ),
                ],
            ),
        ),
    ]


def _predictions_tab() -> list:
    pct = [
        _col(col, f"{who} {side} %")
        for who, cols in (("Model", MODEL_P), ("Market", MARKET_P))
        for col, side in zip(cols, "HDA", strict=True)
    ]
    return [
        html.P(
            "Latest prediction per upcoming fixture from predictions/history.csv "
            "(run `pl-predictor model predict` or `watch` to add more).",
            className="note",
        ),
        _card(
            _grid(
                "predictions",
                [_col("kickoff", "Kickoff (UTC)", 2), _col("fixture", "Fixture", 2)]
                + pct
                + [_col("why", "Why", 6, minWidth=320, wrapText=True, autoHeight=True)],
            )
        ),
    ]


def _record_tab() -> list:
    return [
        html.P(id="record-note", className="note"),
        _card(
            _grid(
                "record",
                [
                    _col("predictor", "Predictor", 3),
                    _col("n", "Fixtures"),
                    _col("log_loss", "Log loss"),
                    _col("brier", "Brier"),
                    _col("accuracy", "Accuracy"),
                ],
            )
        ),
        html.P(
            "Lower log loss and Brier are better. Model, Elo and market are compared on fixtures "
            "with pre-match odds; closing odds are the ceiling to aim for.",
            className="note",
        ),
    ]


def create_app(db_path: str | Path, teams_path: str | Path = "config/teams.yaml") -> Dash:
    app = Dash(__name__, title="PL Predictor", index_string=INDEX)
    teams = load_teams(teams_path)

    app.layout = html.Div(
        className="page",
        children=[
            html.Header(
                className="header",
                children=[
                    html.H1("Premier League predictor"),
                    html.P(
                        "Home/draw/away predictions from results and Elo, and fan mood on X scored "
                        "with a Twitter-tuned RoBERTa model (net sentiment = P(positive) − "
                        "P(negative)).",
                        className="subtitle",
                    ),
                ],
            ),
            html.Div(id="demo-banner"),
            dcc.Tabs(
                id="tabs",
                value="central",
                # Dash stacks tabs vertically below 800px by default; keep FIFA's single row.
                mobile_breakpoint=0,
                style={"borderBottom": "none", "flexWrap": "wrap"},
                children=[
                    dcc.Tab(
                        label=label,
                        value=value,
                        style=TAB,
                        selected_style=TAB_ON,
                        children=html.Div(body, className="tab-body"),
                    )
                    for label, value, body in (
                        ("Central", "central", _central_tab(teams)),
                        ("Fixture", "fixture", _fixture_tab()),
                        ("Predictions", "predictions", _predictions_tab()),
                        ("Track record", "record", _record_tab()),
                    )
                ],
            ),
            dcc.Interval(id="refresh", interval=REFRESH_MS),
        ],
    )

    # Dash only runs a callback while its outputs are on screen, so each tab's callbacks
    # fire when that tab is opened and then on every refresh while it stays open.

    @app.callback(Output("demo-banner", "children"), Input("refresh", "n_intervals"))
    def update_banner(_):
        store = Store(db_path)
        try:
            demo = store.scored_with(DEMO_MODEL)
        finally:
            store.close()
        if not demo:
            return None
        return html.Div(
            "Demo data: these tweets are synthetic (from `pl-predictor demo`), not real fan posts.",
            className="banner",
            role="status",
        )

    @app.callback(
        Output("club-disc", "children"),
        Output("next-match", "children"),
        Input("club", "value"),
        Input("refresh", "n_intervals"),
    )
    def update_club(club, _):
        try:
            fpl = _fpl()
        except requests.RequestException as exc:
            return _disc(club, teams, "big"), _sub(f"Could not reach the FPL API ({exc}).")
        match = figures.next_match(fpl, club, pd.Timestamp.now(tz="UTC"))
        return _disc(club, teams, "big"), _next_match_tile(match, club, read_history(), teams)

    # Its own callback: xG comes from the results file, whose first download can take a while.
    @app.callback(
        Output("league", "children"),
        Output("next-xg", "children"),
        Input("club", "value"),
        Input("refresh", "n_intervals"),
    )
    def update_league(club, _):
        try:
            fpl = _fpl()
        except requests.RequestException as exc:
            return _sub(f"Could not reach the FPL API ({exc})."), None
        try:
            xg = figures.xg_table(_results())
        except SOURCE_ERRORS:  # the table still shows, without xG
            xg = pd.DataFrame(columns=["xg_for", "xg_against", "xgd"])
        match = figures.next_match(fpl, club, pd.Timestamp.now(tz="UTC"))
        return _league_tile(figures.league_table(fpl), club, teams, xg), _xg_line(match, xg)

    # Its own callback: the first results download can take a while; other tiles needn't wait.
    @app.callback(Output("vs-market", "children"), Input("refresh", "n_intervals"))
    def update_vs_market(_):
        return _record_tile(read_history())

    @app.callback(
        Output("mood-map", "figure"),
        Output("top-tweets", "children"),
        Output("previews", "data"),
        Input("refresh", "n_intervals"),
    )
    def update_hub(_):
        fixtures, all_tweets = _frames(db_path)
        return (
            figures.mood_map(all_tweets, fixtures, teams),
            _feed_tile(figures.top_tweets_overall(all_tweets, fixtures), teams),
            figures.fixture_previews(all_tweets, fixtures),
        )

    @app.callback(
        Output("preview", "children"), Input("carousel", "n_intervals"), Input("previews", "data")
    )
    def update_preview(n, previews):
        if not previews:
            return [html.H2("Preview"), _sub(NO_TWEETS)]
        i = (n or 0) % len(previews)
        p = previews[i]
        return [
            html.H2(f"Preview: {p['title']}"),
            _sub(f"{p['kickoff']} · {p['tweets']} tweets"),
            html.Div(
                className="mood-pair",
                children=[
                    _mood(p["home"], p["home_net"], teams),
                    _mood(p["away"], p["away_net"], teams),
                ],
            ),
            html.Div(
                className="dots",
                children=[
                    html.Span(className="dot on" if j == i else "dot") for j in range(len(previews))
                ],
            ),
        ]

    @app.callback(
        Output("fixture", "options"),
        Output("fixture", "value"),
        Input("refresh", "n_intervals"),
        State("fixture", "value"),
    )
    def update_fixture_options(_, current):
        store = Store(db_path)
        try:
            fixtures = store.fixtures_frame()
            with_data = store.scored_fixture_ids()
        finally:
            store.close()
        options = [_fixture_option(row) for row in fixtures.itertuples()]
        ids = [o["value"] for o in options]
        if current in ids:
            return options, current
        # Default to the most recent fixture that has data, else the most recent one.
        preferred = [i for i in ids if i in with_data] or ids
        return options, preferred[-1] if preferred else None

    @app.callback(
        Output("kpis", "children"),
        Output("timeline-title", "children"),
        Output("timeline", "figure"),
        Output("volume", "figure"),
        Output("mix", "figure"),
        Output("tweets", "rowData"),
        Input("fixture", "value"),
        Input("bucket", "value"),
        Input("refresh", "n_intervals"),
    )
    def update_fixture(fixture_id, bucket, _):
        fixtures, all_tweets = _frames(db_path)
        title = f"Net sentiment over time ({bucket}-min buckets)"
        match = fixtures[fixtures["fixture_id"] == fixture_id]
        if match.empty:
            empty = figures.empty_figure("No fixtures yet: run `pl-predictor run` or `demo`")
            return [], title, empty, empty, empty, []

        fixture = match.iloc[0]
        df = all_tweets[all_tweets["fixture_id"] == fixture_id]
        return (
            [_tile(*k) for k in figures.kpis(df, fixture)],
            title,
            figures.sentiment_timeline(df, fixture, bucket),
            figures.volume_timeline(df, fixture, bucket),
            figures.sentiment_mix(df, fixture),
            figures.top_tweets(df, fixture),
        )

    @app.callback(Output("predictions", "rowData"), Input("refresh", "n_intervals"))
    def update_predictions(_):
        return figures.upcoming_predictions(read_history(), pd.Timestamp.now(tz="UTC"))

    @app.callback(
        Output("record-note", "children"),
        Output("record", "rowData"),
        Input("refresh", "n_intervals"),
    )
    def update_record(_):
        scored = _scored(read_history())
        if isinstance(scored, str):
            return scored, []
        summary, done, pending = scored
        note = f"Scored fixtures: {done}. Not played yet (or no result published): {pending}."
        return note, figures.track_record(summary)

    return app
