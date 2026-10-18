"""Plotly Dash app: per-fixture sentiment timeline, volume, mix, club ranking and top tweets."""

from __future__ import annotations

from pathlib import Path

import dash_ag_grid as dag
import pandas as pd
from dash import Dash, Input, Output, State, dcc, html

from . import figures
from .storage import Store

REFRESH_MS = 60_000

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
  background: var(--page);
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

.grid-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
  gap: 16px;
}
"""

INDEX = (
    "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
    f"<style>{CSS}</style></head><body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}"
    "{%renderer%}</footer></body></html>"
)
GRAPH_CONFIG = {"displayModeBar": False}


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


def _card(*children, className: str = "card") -> html.Div:
    return html.Div(className=className, children=list(children))


def _graph(graph_id: str) -> dcc.Graph:
    return dcc.Graph(id=graph_id, config=GRAPH_CONFIG)


def create_app(db_path: str | Path) -> Dash:
    app = Dash(__name__, title="PL Fixture Sentiment", index_string=INDEX)

    app.layout = html.Div(
        className="page",
        children=[
            html.Header(
                className="header",
                children=[
                    html.H1("Premier League fixture sentiment"),
                    html.P(
                        "Fan mood on X around each match, scored with a Twitter-tuned "
                        "RoBERTa model. Net sentiment = P(positive) − P(negative).",
                        className="subtitle",
                    ),
                ],
            ),
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
            html.Div(
                className="grid-2",
                children=[
                    _card(html.H2("Sentiment mix by club"), _graph("mix")),
                    _card(html.H2("Club mood across all collected fixtures"), _graph("clubs")),
                ],
            ),
            _card(
                html.H2("Most-engaged tweets"),
                dag.AgGrid(
                    id="tweets",
                    columnDefs=[
                        {"field": "time", "headerName": "Time", "flex": 1, "minWidth": 72},
                        {"field": "club", "headerName": "Club", "flex": 2, "minWidth": 120},
                        {"field": "label", "headerName": "Label", "flex": 1, "minWidth": 90},
                        {
                            "field": "score",
                            "headerName": "Net",
                            "flex": 1,
                            "minWidth": 72,
                            "type": "numericColumn",
                        },
                        {
                            "field": "likes",
                            "headerName": "Likes",
                            "flex": 1,
                            "minWidth": 72,
                            "type": "numericColumn",
                        },
                        {
                            "field": "text",
                            "headerName": "Tweet",
                            "flex": 5,
                            "minWidth": 240,
                            "wrapText": True,
                            "autoHeight": True,
                            "cellStyle": {"lineHeight": "20px", "padding": "8px 16px"},
                        },
                    ],
                    rowData=[],
                    defaultColDef={"sortable": True, "resizable": True},
                    dashGridOptions={"domLayout": "autoHeight", "animateRows": False},
                    className="ag-theme-alpine",
                    style={"width": "100%"},
                ),
            ),
            dcc.Interval(id="refresh", interval=REFRESH_MS),
        ],
    )

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
            with_data = set(store.tweets_frame()["fixture_id"])
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
        Output("clubs", "figure"),
        Output("tweets", "rowData"),
        Input("fixture", "value"),
        Input("bucket", "value"),
        Input("refresh", "n_intervals"),
    )
    def update_fixture(fixture_id, bucket, _):
        store = Store(db_path)
        try:
            fixtures = store.fixtures_frame()
            all_tweets = store.tweets_frame()
        finally:
            store.close()

        title = f"Net sentiment over time ({bucket}-min buckets)"
        clubs = figures.club_ranking(all_tweets, fixtures)
        match = fixtures[fixtures["fixture_id"] == fixture_id]
        if match.empty:
            empty = figures.empty_figure("No fixtures yet: run `pl-predictor run` or `demo`")
            return [], title, empty, empty, empty, clubs, []

        fixture = match.iloc[0]
        df = all_tweets[all_tweets["fixture_id"] == fixture_id]
        return (
            [_tile(*k) for k in figures.kpis(df, fixture)],
            title,
            figures.sentiment_timeline(df, fixture, bucket),
            figures.volume_timeline(df, fixture, bucket),
            figures.sentiment_mix(df, fixture),
            clubs,
            figures.top_tweets(df, fixture),
        )

    return app
