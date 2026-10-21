"""Tiles and figures for the rolling backtest dashboard.

Every function is a pure function of a frame and returns numbers or a Plotly figure, so
none of it needs a running server. No number is computed here: the tiles call
`backtest.metrics`, which is the only module in the project that divides.

Colour follows the job the data does. Profit has polarity, so a calendar cell is green
above zero and red below, with a neutral grey at zero. The high-volatility dot is amber,
which is the one hue that separates from both poles under red-green colour blindness. It
also sits under the date rather than over the fill, so the two never have to be told
apart by hue alone.
"""
import calendar as calendarModule

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
from backtest import metrics

surface = "#fcfcfb"
ink = "#0b0b0b"
secondaryInk = "#52514e"
mutedInk = "#898781"
gridline = "#e1e0d9"
axisLine = "#c3c2b7"
profitPole = "#1f9d6a"
lossPole = "#e34948"
volDot = "#e8a33a"
neutral = "#f0efec"
series = {"realised": "#2a78d6", "forecast": "#eb6834",
          "agent": "#1baf7a", "oracle": "#4a3aa7"}

weekdayNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
highVolKey = f"spike intensity at or above ${config.high_vol_intensity:.2f}"


def _money(value):
    return "no data" if pd.isna(value) else f"${value:,.2f}"


def _percent(value):
    return "no data" if pd.isna(value) else f"{value * 100:.1f}%"


def summaryTiles(frame):
    """The headline numbers for whatever scope is selected, straight from metrics.

    One day holds one row, so a win rate and a spread across folds would say nothing.
    The last three tiles therefore describe the day itself in that case.
    """
    got = metrics.summarise(frame)
    empty = got["days"] == 0
    rows = [
        ("Captured", _percent(got["captured"]), got["captured"],
         "agent dollars over oracle dollars"),
        ("Agent profit", _money(got["agentProfit"]), got["agentProfit"],
         "after degradation"),
        ("Oracle profit", _money(got["oracleProfit"]), got["oracleProfit"],
         "perfect foresight bound"),
    ]
    if got["days"] == 1:
        facts = metrics.dayFacts(frame)
        rows += [
            ("Revenue", _money(facts["revenue"]), facts["revenue"],
             "before degradation"),
            ("Degradation", _money(facts["degradation"]), facts["degradation"],
             "cost of the day's cycling"),
            ("Spike intensity", _money(facts["spikeIntensity"]),
             facts["spikeIntensity"],
             "high volatility day" if facts["highVol"] else "a normal day"),
        ]
    else:
        rows += [
            ("Win rate", _percent(got["winRate"]), got["winRate"], "days above zero"),
            ("Fold median", _percent(got["foldMedian"]), got["foldMedian"],
             f"IQR {_percent(got['foldIqr'])}"),
            ("Days", "no data" if empty else f"{got['days']:,}", float(got["days"]),
             "in this scope"),
        ]
    return [{"label": label, "value": "no data" if empty else value,
             "raw": raw, "sub": sub} for label, value, raw, sub in rows]


def _emptyFigure(message):
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5,
                          xref="paper", yref="paper",
                          font=dict(color=secondaryInk, size=14))
    figure.update_layout(**_layout(height=260))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def _layout(**extra):
    base = dict(paper_bgcolor=surface, plot_bgcolor=surface,
                font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                          color=ink, size=13),
                margin=dict(l=56, r=24, t=28, b=40), hovermode="closest")
    base.update(extra)
    return base


def monthFigure(frame, month, selected=None):
    """One calendar month, weekdays across and weeks down.

    Fill is the day's profit, green above zero and red below, so a loss can never read
    as a win. An amber dot under the date marks a high-volatility day. A day the backtest
    never tested is left unfilled but still carries its number, so the grid reads as a
    month rather than as a chart.

    Day profit is heavily tailed: a few spike days are worth twenty times a normal one,
    and on a raw scale every other day collapses to the midpoint. The scale therefore
    clamps at the 95th percentile of absolute profit, and the hover carries the true
    figures.
    """
    month = pd.Timestamp(month)
    first = month.normalize().replace(day=1)
    weekdayOffset = int(first.dayofweek)
    length = calendarModule.monthrange(first.year, first.month)[1]
    rows = -(-(weekdayOffset + length) // 7)

    days = metrics.perDay(frame)
    if not days.empty:
        days = days.groupby("testDay", as_index=False).agg(
            agentProfit=("agentProfit", "sum"), oracleProfit=("oracleProfit", "sum"),
            highVol=("highVol", "first"), fold=("fold", "first"))

    profit = {}
    for row in days.to_dict("records"):
        profit[pd.Timestamp(row["testDay"]).day] = row

    grid = np.full((rows, 7), np.nan)
    extra = np.empty((rows, 7, 4), dtype=object)
    extra[:] = ""
    for number in range(1, length + 1):
        cell = weekdayOffset + number - 1
        row, column = divmod(cell, 7)
        found = profit.get(number)
        extra[row, column] = [str(first.replace(day=number).date()),
                              "" if found is None else int(found["fold"]),
                              np.nan if found is None else found["agentProfit"],
                              np.nan if found is None else found["oracleProfit"]]
        if found is not None:
            grid[row, column] = found["agentProfit"]

    # A month the agent made nothing on would give a NaN percentile, and a NaN colour
    # range draws no fill at all.
    limit = (float(np.nanpercentile(np.abs(grid), 95))
             if np.isfinite(grid).any() else 1.0) or 1.0

    figure = go.Figure()
    figure.add_trace(go.Heatmap(
        z=grid, x=list(range(7)), y=list(range(rows)), customdata=extra,
        zmid=0, zmin=-limit, zmax=limit,
        colorscale=[[0.0, lossPole], [0.5, neutral], [1.0, profitPole]],
        xgap=3, ygap=3, hoverongaps=False, showscale=False,
        hovertemplate=("%{customdata[0]} · fold %{customdata[1]}<br>"
                       "profit $%{customdata[2]:,.2f}"
                       "<br>oracle $%{customdata[3]:,.2f}<extra></extra>"),
    ))

    numbers = dict(x=[], y=[], text=[], colour=[], customdata=[])
    dots = dict(x=[], y=[], customdata=[])
    for number in range(1, length + 1):
        row, column = divmod(weekdayOffset + number - 1, 7)
        found = profit.get(number)
        filled = found is not None
        # White on a saturated fill, ink on a pale one. Judged against the clamped
        # range, which is what the fill actually uses.
        strong = filled and abs(found["agentProfit"]) > limit * 0.55
        numbers["x"].append(column)
        numbers["y"].append(row - 0.22)
        numbers["text"].append(str(number))
        numbers["colour"].append(surface if strong else
                                 (ink if filled else mutedInk))
        numbers["customdata"].append(extra[row, column])
        if filled and bool(found["highVol"]):
            dots["x"].append(column)
            dots["y"].append(row + 0.2)
            dots["customdata"].append(extra[row, column])

    figure.add_trace(go.Scatter(
        x=numbers["x"], y=numbers["y"], mode="text", text=numbers["text"],
        textfont=dict(color=numbers["colour"], size=13),
        customdata=numbers["customdata"], hoverinfo="skip", showlegend=False))
    if dots["x"]:
        figure.add_trace(go.Scatter(
            x=dots["x"], y=dots["y"], mode="markers", name="High volatility",
            marker=dict(symbol="circle", size=8, color=volDot),
            customdata=dots["customdata"], hoverinfo="skip", showlegend=True))

    shapes = []
    if selected is not None:
        chosen = pd.Timestamp(selected)
        if (chosen.year, chosen.month) == (first.year, first.month):
            row, column = divmod(weekdayOffset + chosen.day - 1, 7)
            shapes = [dict(type="rect", x0=column - 0.5, x1=column + 0.5,
                           y0=row - 0.5, y1=row + 0.5, line=dict(color=ink, width=2),
                           fillcolor="rgba(0,0,0,0)")]

    figure.update_layout(**_layout(
        height=120 + 62 * rows, margin=dict(l=24, r=24, t=34, b=46), shapes=shapes,
        legend=dict(orientation="h", yanchor="top", y=-0.04, x=0,
                    font=dict(color=secondaryInk)),
    ))
    figure.update_xaxes(side="top", showgrid=False, zeroline=False, showline=False,
                        tickmode="array", tickvals=list(range(7)), ticktext=weekdayNames,
                        range=[-0.5, 6.5], tickfont=dict(color=mutedInk))
    figure.update_yaxes(autorange="reversed", showgrid=False, zeroline=False,
                        showline=False, showticklabels=False)
    return figure


def dayFigure(trajectory, day, algo=None, trainMode=None, priceSource=None):
    """One test day: the price paths on top, the state of charge below.

    Two panels rather than two y-scales. Price in $/MWh and state of charge in 0-1 do
    not share an axis. The oracle series the runner writes is kept whichever agent train mode
    is selected, so the day always shows what perfect foresight would have done.
    """
    rows = trajectory
    oracle = rows[rows.trainMode == "oracle"] if not rows.empty else rows
    for column, wanted in (("algo", algo), ("trainMode", trainMode),
                           ("priceSource", priceSource)):
        if wanted is not None and not rows.empty:
            rows = rows[rows[column] == wanted]
    if rows.empty:
        return _emptyFigure(f"no trajectory written for {pd.Timestamp(day):%Y-%m-%d}")

    realised = rows[rows.priceSource == "realised"]
    forecast = rows[rows.priceSource == "forecast"]
    spine = realised if not realised.empty else forecast
    clock = pd.to_datetime(spine.settlementdate)

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=clock, y=spine.price, name="Realised price", mode="lines", yaxis="y",
        line=dict(color=series["realised"], width=2),
        hovertemplate="%{x|%H:%M} · $%{y:.2f}<extra>Realised price</extra>"))
    if not forecast.empty:
        figure.add_trace(go.Scatter(
            x=pd.to_datetime(forecast.settlementdate), y=forecast.observedPrice,
            name="Forecast seen", mode="lines", yaxis="y",
            line=dict(color=series["forecast"], width=2, dash="dot"),
            hovertemplate="%{x|%H:%M} · $%{y:.2f}<extra>Forecast seen</extra>"))

    for source, label, dash in (("realised", "Agent SoC", "solid"),
                                ("forecast", "Agent SoC on forecast", "dot")):
        side = rows[rows.priceSource == source]
        if side.empty:
            continue
        figure.add_trace(go.Scatter(
            x=pd.to_datetime(side.settlementdate), y=side.soc, name=label,
            mode="lines", yaxis="y2",
            line=dict(color=series["agent"], width=2, dash=dash),
            hovertemplate="%%{x|%%H:%%M} · %%{y:.1%%}<extra>%s</extra>" % label))

    if not oracle.empty:
        oracle = oracle[oracle.priceSource == oracle.priceSource.iloc[0]]
        figure.add_trace(go.Scatter(
            x=pd.to_datetime(oracle.settlementdate), y=oracle.soc,
            name="Oracle SoC", mode="lines", yaxis="y2",
            line=dict(color=series["oracle"], width=2),
            hovertemplate="%{x|%H:%M} · %{y:.1%}<extra>Oracle SoC</extra>"))

    figure.update_layout(**_layout(
        height=520, hovermode="x unified",
        # Anchored to the lower panel so the shared tick row sits under the whole
        # figure rather than between the two panels.
        xaxis=dict(domain=[0, 1], anchor="y2", showgrid=False, linecolor=axisLine,
                   tickfont=dict(color=mutedInk)),
        yaxis=dict(domain=[0.54, 1.0], anchor="free", position=0.0,
                   title=dict(text="Price $/MWh"),
                   gridcolor=gridline, zeroline=True, zerolinecolor=axisLine,
                   tickfont=dict(color=mutedInk)),
        yaxis2=dict(domain=[0.0, 0.42], title=dict(text="State of charge"),
                    tickformat=".0%", range=[-0.02, 1.02], gridcolor=gridline,
                    zeroline=False, tickfont=dict(color=mutedInk), anchor="x"),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                    font=dict(color=secondaryInk)),
    ))
    return figure
