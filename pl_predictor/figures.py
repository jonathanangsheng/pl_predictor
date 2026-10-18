"""Pure data -> figure functions for the dashboard (kept free of Dash so they're testable)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .fixtures import SIDE_TWEETS

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


def club_ranking(all_tweets: pd.DataFrame, fixtures: pd.DataFrame) -> go.Figure:
    table = club_table(all_tweets, fixtures)
    if table.empty:
        return empty_figure()

    colors = [
        LABEL_COLORS["positive"] if s >= 0 else LABEL_COLORS["negative"] for s in table["score"]
    ]
    fig = go.Figure(
        go.Bar(
            y=table["club"],
            x=table["score"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=SURFACE, width=2)),
            customdata=table["tweets"],
            # round() + 0.0 turns -0.00 into +0.00
            text=[f"{round(s, 2) + 0.0:+.2f}" for s in table["score"]],
            textposition="outside",
            textfont=dict(color=INK_2),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Net sentiment %{x:+.2f}<br>%{customdata} tweets"
            "<extra></extra>",
        )
    )
    _base_layout(fig, height=max(260, 30 * len(table) + 80))
    limit = max(0.25, table["score"].abs().max() * 1.4)
    fig.update_xaxes(range=[-limit, limit], tickformat="+.1f", zeroline=True, zerolinewidth=1)
    fig.update_yaxes(showgrid=False)
    return fig


def kpis(df: pd.DataFrame, fixture: pd.Series) -> list[tuple[str, str, str]]:
    """(label, value, caption) tuples for the stat tiles."""
    if df.empty:
        return [("Tweets scored", "0", "waiting for data")]

    def net(side: str) -> str:
        sub = _for_side(df, side)
        return f"{sub['score'].mean():+.2f}" if not sub.empty else "–"

    live = df[df["phase"] == "live"]
    return [
        ("Tweets scored", f"{len(df):,}", f"{len(live):,} during the match"),
        (fixture["home_name"], net("home"), "mean net sentiment"),
        (fixture["away_name"], net("away"), "mean net sentiment"),
        ("Positive share", f"{(df['label'] == 'positive').mean():.0%}", "of all scored tweets"),
    ]


def top_tweets(df: pd.DataFrame, fixture: pd.Series, n: int = 15) -> list[dict]:
    if df.empty:
        return []
    names = {"home": fixture["home_name"], "away": fixture["away_name"], "both": "Both"}
    top = df.assign(engagement=df["like_count"] + df["retweet_count"] * 2 + df["reply_count"])
    top = top.nlargest(n, "engagement")
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
