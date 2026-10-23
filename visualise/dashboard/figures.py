"""Tiles and figures for the rolling backtest dashboard.

Every function is a pure function of a frame and returns numbers or a Plotly figure, so
none of it needs a running server. No number is computed here: the tiles call
`backtest.metrics`, which is the only module in the project that divides.

Colour follows the job the data does. Profit has polarity, so a calendar cell is green
above zero and red below, with a neutral grey at zero. The high-volatility dot is amber,
which is the one hue that separates from both poles under red-green colour blindness. It
also sits under the date rather than over the fill, so the two never have to be told
apart by hue alone.

A dispatch block keeps one colour on both screens: blue to buy, red to sell. The day
view and the forecast view call `_dispatchFigure`, so neither screen teaches a second
key.
"""
import calendar as calendarModule
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
from backtest import metrics
from data.forecasts import p5minIntervals

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
# Opaque enough to read as a block, open enough to keep the price path visible.
chargeBlock = "rgba(42,120,214,0.42)"
dischargeBlock = "rgba(227,73,72,0.42)"
weekdayNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
capPayoutKey = (f"sum(max(RRP - ${config.spike_threshold:,.0f}, 0))"
                f" / {config.episode_intervals}")
highVolKey = f"cap payout at or above ${config.high_vol_intensity:.2f}"


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
            ("Cap Payout", _money(facts["spikeIntensity"]),
             facts["spikeIntensity"], capPayoutKey),
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


def forecastTiles(frame, initialSoc=config.initial_soc):
    """The headline numbers over the forecast board: what the plan earns, and where the
    battery starts.

    `live.suggest` runs both totals forward, so the last row already covers the whole
    horizon and nothing is summed here. The figures are an expectation over an AEMO
    forecast and not a settled result, which is what "expected" says.

    A board drawn without a policy still carries a state of charge, because the screen
    assumes that rather than the agent choosing it. Its revenue is unknown, so it reads
    as no data in the way every other tile does.
    """
    planned = not frame.empty and "revenue" in frame
    revenue = float(frame.revenue.iloc[-1]) if planned else float("nan")
    profit = float(frame.profit.iloc[-1]) if planned else float("nan")
    endSoc = float(frame.soc.iloc[-1]) if planned else float("nan")
    return [
        {"label": "Expected revenue", "value": _money(revenue), "raw": revenue,
         "sub": f"{_money(profit)} after degradation" if planned
                else "no dispatch planned"},
        {"label": "Current state of charge", "value": _percent(initialSoc),
         "raw": initialSoc,
         "sub": f"ends the forecast at {_percent(endSoc)}" if planned
                else "assumed at the run time"},
    ]


def _wrap(text, width=84):
    """Plotly never wraps annotation text, so a long message runs off the panel."""
    return "<br>".join(textwrap.wrap(str(text), width))


def _emptyFigure(message):
    figure = go.Figure()
    figure.add_annotation(text=_wrap(message), showarrow=False, x=0.5, y=0.5,
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


def dayFigure(frame, blocks=(), day=None, region=None, model=None):
    """One test day: the realised price path, and the dispatch the agent ran over it.

    The same figure the current forecast screen draws, over realised prices rather than
    a forecast. `_dispatchFigure` explains how a block is placed.
    """
    if frame.empty:
        named = f" under {model}" if model else ""
        return _emptyFigure(
            f"no trajectory written for {pd.Timestamp(day):%Y-%m-%d}{named}. A fold "
            "scores every seed but writes the per-interval trajectory of one seed "
            "only, so another seed of the same model holds this day.")
    clock = pd.to_datetime(frame.settlementdate)
    # Every realised interval is its own published price, so none is thinned out.
    drawn = np.ones(len(clock), dtype=bool)
    return _dispatchFigure(
        clock, frame.price.to_numpy(dtype=float), blocks, drawn,
        name="Realised price", yTitle="Realised price $/MWh", hoverTime="%H:%M",
        title=" · ".join(part for part in (region, model) if part))


def publishedIntervals(clock, fine=p5minIntervals):
    """Which rows of a forecast path AEMO actually published as separate numbers.

    P5MIN gives the first `fine` intervals one price each. PREDISPATCH gives the rest
    one price per half hour, which `data.forecasts` then holds flat across the six
    5-minute slots of that half hour. Drawing all six invents detail the run does not
    have, so only the half-hour stamp is kept past the P5MIN part.
    """
    return (np.arange(len(clock)) < fine) | (clock.dt.minute % 30 == 0).to_numpy()


def _dispatchFigure(clock, price, blocks, drawn, *, name, yTitle, hoverTime,
                    title=None, note=None):
    """A price path with the dispatch blocks of one policy over it.

    The path is drawn as the step function a dispatch price is: each point holds from
    the end of the interval before it, so a PREDISPATCH half hour reads as one flat
    30-minute block and a 5-minute interval as one flat 5-minute block.

    The horizontal rule is the mean price, which is the line the blocks are read
    against: a discharge block stands above it and a charge block hangs below it. Each
    block spans the intervals of one unbroken run of the same action, and reaches from
    the rule to the furthest price inside that run. The fills sit under the price line,
    so the path stays readable through them.

    `drawn` selects which rows the line uses. The blocks and the rule always read every
    interval, so thinning the line never moves a block.
    """
    rule = float(np.mean(price))
    shapes = []
    for action, start, end in blocks:
        inside = price[((clock > start) & (clock <= end)).to_numpy()]
        if not len(inside):
            continue
        far = float(inside.max() if action == "discharge" else inside.min())
        shapes.append(dict(
            type="rect", xref="x", yref="y", x0=start, x1=end, layer="below",
            y0=min(rule, far), y1=max(rule, far), line=dict(width=0),
            fillcolor=dischargeBlock if action == "discharge" else chargeBlock))

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=clock[drawn], y=price[drawn], name=name, mode="lines",
        line=dict(color=ink, width=2, shape="vh"),
        hovertemplate="%%{x|%s} · $%%{y:.2f}<extra>%s</extra>" % (hoverTime, name)))
    # Two empty traces carry the key. A shape cannot hold a legend entry of its own.
    for label, colour in (("Discharge: sell", dischargeBlock),
                          ("Charge: buy", chargeBlock)):
        figure.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=label, hoverinfo="skip",
            marker=dict(symbol="square", size=13, color=colour, line=dict(width=0)),
            showlegend=True))

    figure.update_layout(**_layout(
        height=460, hovermode="x unified", shapes=shapes,
        title=dict(text=title or "", x=0, font=dict(color=secondaryInk, size=13)),
        margin=dict(l=56, r=24, t=58, b=40),
        xaxis=dict(showgrid=False, linecolor=axisLine, tickfont=dict(color=mutedInk)),
        yaxis=dict(title=dict(text=yTitle), gridcolor=gridline,
                   zeroline=False, tickfont=dict(color=mutedInk)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1,
                    xanchor="right", font=dict(color=secondaryInk)),
    ))
    figure.add_hline(y=rule, line=dict(color=axisLine, width=1))
    if note:
        figure.add_annotation(text=_wrap(note), showarrow=False, x=0, y=1.0,
                              xanchor="left", yanchor="bottom", xref="paper",
                              yref="paper", align="left",
                              font=dict(color=lossPole, size=12))
    return figure


def forecastFigure(frame, blocks=(), region=None, model=None, note=None):
    """The forward AEMO price path, with the dispatch a policy would run over it."""
    if frame.empty:
        return _emptyFigure(note or "no forecast published for this run time")
    clock = pd.to_datetime(frame.settlementdate)
    return _dispatchFigure(
        clock, frame.price.to_numpy(dtype=float), blocks, publishedIntervals(clock),
        name="AEMO forecast", yTitle="Forecast price $/MWh",
        hoverTime="%d/%m %H:%M", note=note,
        title=" · ".join(part for part in (region, model) if part))
