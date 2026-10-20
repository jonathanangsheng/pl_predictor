"""Pure data -> figure functions for the dashboard (kept free of Dash so they're testable)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .fixtures import SIDE_TWEETS
from .predict import MARKET_P, MODEL_P, latest_per_fixture, summary_table

# Categorical slots: home / away. Sentiment uses its own violet<->red poles with a
# gray midpoint so it never collides with the team colours.
HOME = "#2a78d6"
AWAY = "#eb6834"
OTHER = "#c3c2b7"
LABEL_COLORS = {"positive": "#6250d6", "neutral": "#c3c2b7", "negative": "#e34948"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_2),
        # traceorder="normal" stops stacked bars from listing their legend backwards.
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", title=None, traceorder="normal"),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK)),
        bargap=0.25,
    )
    axis = dict(
        gridcolor=GRID,
        linecolor=BASELINE,
        zerolinecolor=BASELINE,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def empty_figure(message: str = "No scored tweets yet") -> go.Figure:
    fig = _base_layout(go.Figure())
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _for_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Tweets about one club; tweets mentioning both clubs count for each."""
    return df[df["side"].isin(SIDE_TWEETS[side])]


def _add_match_markers(fig: go.Figure, fixture: pd.Series) -> None:
    for ts, text in ((fixture["kickoff"], "Kickoff"), (fixture["full_time"], "Full time")):
        fig.add_vline(x=ts, line=dict(color=BASELINE, width=1, dash="dot"))
        fig.add_annotation(
            x=ts,
            y=1,
            yref="paper",
            text=text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=4,
            font=dict(color=MUTED, size=11),
        )


def sentiment_timeline(df: pd.DataFrame, fixture: pd.Series, bucket_min: int) -> go.Figure:
    if df.empty:
        return empty_figure()

    fig = go.Figure()
    for side, name, color in (
        ("home", fixture["home_name"], HOME),
        ("away", fixture["away_name"], AWAY),
    ):
        sub = _for_side(df, side).set_index("created_at")["score"]
        if sub.empty:
            continue
        agg = sub.resample(f"{bucket_min}min").agg(["mean", "count"])
        agg = agg[agg["count"] > 0]
        fig.add_trace(
            go.Scatter(
                x=agg.index,
                y=agg["mean"],
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                customdata=agg["count"],
                # resample labels buckets by their start; draw them at their midpoint.
                xperiod=bucket_min * 60_000,
                xperiodalignment="middle",
                hovertemplate=f"<b>{name}</b> %{{y:+.2f}} · %{{customdata}} tweets<extra></extra>",
            )
        )
    _add_match_markers(fig, fixture)
    fig.add_hline(y=0, line=dict(color=BASELINE, width=1))
    _base_layout(fig)
    fig.update_yaxes(range=[-1, 1], tickformat="+.1f", title="P(positive) − P(negative)")
    fig.update_xaxes(hoverformat="%H:%M")
    fig.update_layout(hovermode="x unified")
    return fig


def volume_timeline(df: pd.DataFrame, fixture: pd.Series, bucket_min: int) -> go.Figure:
    if df.empty:
        return empty_figure()

    groups = (
        ("home", fixture["home_name"], HOME),
        ("away", fixture["away_name"], AWAY),
        ("other", "Both / unattributed", OTHER),
    )
    sides = df["side"].where(df["side"].isin(["home", "away"]), "other")
    fig = go.Figure()
    for side, name, color in groups:
        sub = df[sides == side].set_index("created_at")["tweet_id"]
        if sub.empty:
            continue
        counts = sub.resample(f"{bucket_min}min").count()
        fig.add_trace(
            go.Bar(
                x=counts.index,
                y=counts.values,
                name=name,
                xperiod=bucket_min * 60_000,
                xperiodalignment="middle",
                marker=dict(color=color, line=dict(color=SURFACE, width=1)),
                hovertemplate=f"<b>{name}</b> %{{y}} tweets<extra></extra>",
            )
        )
    _add_match_markers(fig, fixture)
    _base_layout(fig, height=260)
    fig.update_layout(barmode="stack", hovermode="x unified")
    fig.update_xaxes(hoverformat="%H:%M")
    fig.update_yaxes(title="Tweets")
    return fig


def sentiment_mix(df: pd.DataFrame, fixture: pd.Series) -> go.Figure:
    if df.empty:
        return empty_figure()

    clubs = [fixture["home_name"], fixture["away_name"]]
    shares = {
        name: _for_side(df, side)["label"].value_counts(normalize=True)
        for side, name in zip(("home", "away"), clubs, strict=True)
    }
    fig = go.Figure()
    for label in ("negative", "neutral", "positive"):
        values = [shares[c].get(label, 0.0) for c in clubs]
        fig.add_trace(
            go.Bar(
                y=clubs,
                x=values,
                name=label.capitalize(),
                orientation="h",
                marker=dict(color=LABEL_COLORS[label], line=dict(color=SURFACE, width=2)),
                text=[f"{v:.0%}" if v >= 0.08 else "" for v in values],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color=INK if label == "neutral" else "white"),
                hovertemplate=f"%{{y}}<br>{label.capitalize()}: %{{x:.1%}}<extra></extra>",
            )
        )
    _base_layout(fig, height=260)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def club_table(all_tweets: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Mean net sentiment and tweet count per club across every stored fixture."""
    if all_tweets.empty:
        return pd.DataFrame(columns=["club", "score", "tweets"])
    merged = all_tweets.merge(fixtures, on="fixture_id")
    frames = []
    for side in ("home", "away"):
        sub = _for_side(merged, side)
        frames.append(pd.DataFrame({"club": sub[f"{side}_name"], "score": sub["score"]}))
    stacked = pd.concat(frames)
    table = stacked.groupby("club")["score"].agg(score="mean", tweets="count").reset_index()
    return table.sort_values("score")


def kpis(df: pd.DataFrame, fixture: pd.Series) -> list[tuple[str, str, str]]:
    """(label, value, caption) tuples for the stat tiles."""
    if df.empty:
        return [("Tweets scored", "0", "waiting for data")]

    def net(side: str) -> str:
        value = _net(df, side)
        return "–" if value is None else f"{value:+.2f}"

    live = df[df["phase"] == "live"]
    return [
        ("Tweets scored", f"{len(df):,}", f"{len(live):,} during the match"),
        (fixture["home_name"], net("home"), "mean net sentiment"),
        (fixture["away_name"], net("away"), "mean net sentiment"),
        ("Positive share", f"{(df['label'] == 'positive').mean():.0%}", "of all scored tweets"),
    ]


def _net(df: pd.DataFrame, side: str) -> float | None:
    """Mean net sentiment of one side's tweets, or None without any."""
    scores = _for_side(df, side)["score"]
    return None if scores.empty else round(float(scores.mean()), 2)


def _most_engaged(df: pd.DataFrame, n: int) -> pd.DataFrame:
    engagement = df["like_count"] + df["retweet_count"] * 2 + df["reply_count"]
    return df.assign(engagement=engagement).nlargest(n, "engagement")


def top_tweets(df: pd.DataFrame, fixture: pd.Series, n: int = 15) -> list[dict]:
    if df.empty:
        return []
    names = {"home": fixture["home_name"], "away": fixture["away_name"], "both": "Both"}
    top = _most_engaged(df, n)
    return [
        {
            "time": row.created_at.strftime("%H:%M"),
            "club": names.get(row.side, "–"),
            "label": row.label.capitalize(),
            "score": round(row.score, 2),
            "likes": int(row.like_count),
            "text": row.text,
        }
        for row in top.itertuples()
    ]


def upcoming_predictions(history: pd.DataFrame, now: pd.Timestamp) -> list[dict]:
    """Latest prediction per fixture that has not kicked off, as grid rows (percentages)."""
    latest = latest_per_fixture(history[history["kickoff"] > now]).sort_values("kickoff")
    rows = pd.DataFrame(
        {
            "kickoff": latest["kickoff"].dt.strftime("%a %d %b %H:%M"),
            "fixture": latest["home"] + " v " + latest["away"],
            **{col: (latest[col] * 100).round() for col in MODEL_P + MARKET_P},
            "why": latest["why"],
        }
    ).astype(object)
    return rows.where(rows.notna(), None).to_dict("records")  # no odds yet -> blank cell


def track_record(summary: dict) -> list[dict]:
    """predict.score() metrics as grid rows; NaN (no fixtures yet) becomes blank."""
    table = summary_table(summary).reset_index(names="predictor").astype(object)
    return table.where(table.notna(), None).to_dict("records")


# ---------------------------------------------------------------- CENTRAL tab


def league_table(matches: pd.DataFrame) -> list[dict]:
    """Points table from finished matches; every club in the fixture list appears, played or not."""
    done = matches[matches["finished"]]
    sides = pd.concat(
        [
            pd.DataFrame(
                {"club": done["home"], "gf": done["home_goals"], "ga": done["away_goals"]}
            ),
            pd.DataFrame(
                {"club": done["away"], "gf": done["away_goals"], "ga": done["home_goals"]}
            ),
        ]
    )
    sides["points"] = np.select([sides["gf"] > sides["ga"], sides["gf"] == sides["ga"]], [3, 1], 0)
    sides["gd"] = sides["gf"] - sides["ga"]
    table = sides.groupby("club").agg(
        played=("gf", "size"), gd=("gd", "sum"), gf=("gf", "sum"), points=("points", "sum")
    )
    clubs = sorted(set(matches["home"]) | set(matches["away"]))
    table = table.reindex(clubs, fill_value=0).reset_index(names="club")
    table = table.sort_values(["points", "gd", "gf", "club"], ascending=[False, False, False, True])
    table.insert(0, "pos", range(1, len(table) + 1))
    return table.astype({c: int for c in ("played", "gd", "points")}).to_dict("records")


def next_match(matches: pd.DataFrame, club: str, now: pd.Timestamp) -> pd.Series | None:
    """The club's next unstarted fixture with a kickoff time."""
    mine = matches[
        ((matches["home"] == club) | (matches["away"] == club))
        & ~matches["started"]
        & (matches["kickoff"] > now)
    ]
    return None if mine.empty else mine.sort_values("kickoff").iloc[0]


def fixture_previews(all_tweets: pd.DataFrame, fixtures: pd.DataFrame, n: int = 5) -> list[dict]:
    """Mood summary of the latest `n` fixtures that have tweets, newest first."""
    with_data = fixtures[fixtures["fixture_id"].isin(all_tweets["fixture_id"])].tail(n)
    previews = []
    for fx in with_data.iloc[::-1].itertuples():
        df = all_tweets[all_tweets["fixture_id"] == fx.fixture_id]
        previews.append(
            {
                "home": fx.home_code,
                "away": fx.away_code,
                "title": f"{fx.home_name} v {fx.away_name}",
                "kickoff": f"{fx.kickoff:%a %d %b %H:%M} UTC",
                "home_net": _net(df, "home"),
                "away_net": _net(df, "away"),
                "tweets": len(df),
            }
        )
    return previews


def top_tweets_overall(all_tweets: pd.DataFrame, fixtures: pd.DataFrame, n: int = 3) -> list[dict]:
    """Most-engaged tweets across every fixture, with the club they were attributed to."""
    top = _most_engaged(all_tweets, n).merge(fixtures, on="fixture_id")
    club = np.select(
        [top["side"] == "home", top["side"] == "away"], [top["home_code"], top["away_code"]], ""
    )
    return [
        {"club": c, "text": t, "likes": int(likes)}
        for c, t, likes in zip(club, top["text"], top["like_count"], strict=True)
    ]


def mood_map(all_tweets: pd.DataFrame, fixtures: pd.DataFrame, teams: dict) -> go.Figure:
    """Stadiums on a dark map; clubs with tweets glow by mean net sentiment."""
    clubs = pd.DataFrame(
        [(t.name, t.lat, t.lon) for t in teams.values() if t.lat is not None],
        columns=["club", "lat", "lon"],
    ).merge(club_table(all_tweets, fixtures), on="club", how="left")
    scored = clubs[clubs["score"].notna()]
    hover = scored["club"] + "<br>net " + scored["score"].map("{:+.2f}".format)
    hover += "<br>" + scored["tweets"].astype(int).astype(str) + " tweets"
    colour = dict(
        color=scored["score"],
        colorscale=[[0, LABEL_COLORS["negative"]], [0.5, OTHER], [1, "#9be22d"]],
        cmin=-0.25,
        cmax=0.25,
    )
    fig = go.Figure(
        [
            go.Scattergeo(
                lat=clubs["lat"],
                lon=clubs["lon"],
                text=clubs["club"],
                hoverinfo="text",
                mode="markers",
                marker=dict(size=6, color="#5d6b70"),
            ),
            go.Scattergeo(
                lat=scored["lat"],
                lon=scored["lon"],
                hoverinfo="skip",
                mode="markers",
                marker=dict(size=30, opacity=0.25, **colour),
            ),
            go.Scattergeo(
                lat=scored["lat"],
                lon=scored["lon"],
                text=hover,
                hoverinfo="text",
                mode="markers",
                marker=dict(size=11, line=dict(width=1, color="#ffffff"), **colour),
            ),
        ]
    )
    fig.update_geos(
        projection_type="mercator",
        lataxis_range=[49.8, 55.9],
        lonaxis_range=[-6.5, 2.2],
        resolution=50,
        showland=True,
        landcolor="#2b3a40",
        showocean=True,
        oceancolor="#141b1f",
        showcountries=True,
        countrycolor="#3d4f56",
        showcoastlines=True,
        coastlinecolor="#3d4f56",
        showlakes=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT),
    )
    return fig


def xg_table(matches: pd.DataFrame, last: int = 10) -> pd.DataFrame:
    """Per club this season: mean xG for/against over its last `last` matches, and xG difference.

    Only matches with xG count. Shown as context only; the model does not use xG yet.
    """
    season = matches[matches["season"] == matches["season"].max()]
    season = season.dropna(subset=["home_xg", "away_xg"])
    sides = pd.concat(
        [
            pd.DataFrame(
                {
                    "club": season[side],
                    "date": season["date"],
                    "xg_for": season[f"{side}_xg"],
                    "xg_against": season[f"{other}_xg"],
                }
            )
            for side, other in (("home", "away"), ("away", "home"))
        ]
    ).sort_values("date")
    recent = sides.groupby("club").tail(last).groupby("club")[["xg_for", "xg_against"]].mean()
    totals = sides.groupby("club")[["xg_for", "xg_against"]].sum()
    return recent.assign(xgd=totals["xg_for"] - totals["xg_against"])
