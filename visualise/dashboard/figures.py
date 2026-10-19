"""Tiles and figures for the rolling backtest dashboard.

Every function is a pure function of a frame and returns numbers or a Plotly figure, so
none of it needs a running server. No number is computed here: the tiles call
`backtest.metrics`, which is the only module in the project that divides.

Colour follows the job the data does. Profit has polarity, so the calendar uses the
diverging blue-to-red pair with a neutral grey at zero. The high-volatility marker
carries no hue at all: it is a dot in primary ink with a surface ring. A third hue has
nowhere to sit beside a diverging scale that already holds a cool and a warm pole, and
every candidate failed the colour-vision separation check against the red pole.
"""
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
profitPole = "#2a78d6"
lossPole = "#e34948"
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
    """The headline numbers for whatever scope is selected, straight from metrics."""
    got = metrics.summarise(frame)
    empty = got["days"] == 0
    rows = [
        ("Captured", _percent(got["captured"]), got["captured"],
         "agent dollars over oracle dollars"),
        ("Agent profit", _money(got["agentProfit"]), got["agentProfit"],
         "after degradation"),
        ("Oracle profit", _money(got["oracleProfit"]), got["oracleProfit"],
         "perfect foresight bound"),
        ("Win rate", _percent(got["winRate"]), got["winRate"], "days above zero"),
        ("Fold median", _percent(got["foldMedian"]), got["foldMedian"],
         f"IQR {_percent(got['foldIqr'])}"),
        ("Days", "no data" if empty else f"{got['days']:,}", float(got["days"]),
         "in this scope"),
    ]
    return [{"label": label, "value": "no data" if empty else value,
             "raw": raw, "sub": sub} for label, value, raw, sub in rows]


def volatilityTiles(frame):
    """The same summary split on the volatility label. Both sides are always present."""
    buckets = metrics.byBucket(frame)
    total = sum(side["oracleProfit"] for side in buckets.values())
    out = []
    for high, name in ((True, "High volatility"), (False, "Other days")):
        side = buckets[high]
        out.append({
            "bucket": name, "days": side["days"],
            "agentProfit": side["agentProfit"], "oracleProfit": side["oracleProfit"],
            "captured": side["captured"],
            "oracleShare": side["oracleProfit"] / total if total else float("nan"),
        })
    return out


def modelPanel(manifest):
    """The model and MDP description, read from the manifest so a new algorithm needs
    no change here."""
    def listed(key, default="--"):
        value = manifest.get(key)
        if value in (None, [], ()):
            return default
        return ", ".join(str(item) for item in value) if isinstance(
            value, (list, tuple)) else str(value)

    weeks = manifest.get("trainWeeks")
    return [
        ("Algorithm", listed("algos")),
        ("Arm", listed("arms")),
        ("Seeds", listed("seeds")),
        ("Training window", "--" if weeks is None else f"{weeks} weeks"),
        ("Test window", f"{manifest.get('testWeeks', '--')} week(s), "
                        f"step {manifest.get('stepWeeks', '--')}"),
        ("Timesteps", f"{manifest.get('coldTimesteps', '--')} cold, "
                      f"{manifest.get('warmTimesteps', '--')} warm top-up"),
        ("Price source", listed("priceSources")),
        ("Price cap pinned to", str(manifest.get("capDate", "--"))),
        ("Observation", f"{config.observation_size} slots: state of charge, "
                        f"{config.five_minute_interval_count} five-minute prices, "
                        f"{config.price_bucket_count} half-hour means, intervals left"),
        ("Actions", "3: charge at the rated power, idle, discharge at the rated power"),
        ("Reward", "interval profit after degradation, less the oracle baseline, "
                   f"scaled by {manifest.get('rewardScale', '--')}"),
        ("Oracle", f"perfect foresight MILP, calendar aging "
                   f"{'on' if manifest.get('calendar') else 'off'}, "
                   f"{'discrete' if manifest.get('discretePower') else 'continuous'} "
                   f"power, gap {manifest.get('gap', '--')}"),
    ]


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


calendarMetrics = {
    "captured": dict(label="Captured", tickformat=".0%", hover="%{z:.1%} captured"),
    "agentProfit": dict(label="Profit $", tickformat="$,.0f", hover="$%{z:,.2f}"),
}


def calendarFigure(frame, metric="captured"):
    """One cell per test day, weeks down and weekdays across.

    Fill is diverging and centred on zero, so a loss can never read as a win. A dot in
    primary ink marks a high-volatility day.

    Captured share is the default rather than dollars because day profit is heavily
    tailed: a handful of spike days are worth twenty times a normal one, and on a dollar
    scale every other day collapses to the midpoint. Captured carries the same sign, so
    the profitable and unprofitable reading is unchanged. Either way the scale clamps at
    the 95th percentile of absolute value, and the hover carries both true figures.
    """
    days = metrics.perDay(frame)
    if days.empty:
        return _emptyFigure("no days in this selection")
    days = days.groupby("testDay", as_index=False).agg(
        agentProfit=("agentProfit", "sum"), oracleProfit=("oracleProfit", "sum"),
        highVol=("highVol", "first"), fold=("fold", "first")).sort_values("testDay")
    days["captured"] = np.where(days.oracleProfit > 0,
                                days.agentProfit / days.oracleProfit, np.nan)
    style = calendarMetrics[metric]

    stamps = pd.to_datetime(days.testDay)
    firstMonday = (stamps.min()
                   - pd.Timedelta(days=int(stamps.min().dayofweek))).normalize()
    column = stamps.dt.dayofweek.to_numpy()
    row = ((stamps.dt.normalize() - firstMonday).dt.days // 7).to_numpy()
    weeks = int(row.max()) + 1

    grid = np.full((weeks, 7), np.nan)
    extra = np.empty((weeks, 7, 4), dtype=object)
    extra[:] = ""
    grid[row, column] = days[metric].to_numpy(dtype=float)
    extra[row, column, 0] = stamps.dt.date.astype(str).to_numpy()
    extra[row, column, 1] = days.fold.astype(int).to_numpy()
    extra[row, column, 2] = days.agentProfit.to_numpy(dtype=float)
    extra[row, column, 3] = days.oracleProfit.to_numpy(dtype=float)

    limit = float(np.nanpercentile(np.abs(grid), 95)) or 1.0
    weekLabels = [f"{(firstMonday + pd.Timedelta(weeks=n)):%-d %b}"
                  for n in range(weeks)]

    figure = go.Figure()
    figure.add_trace(go.Heatmap(
        z=grid, x=weekdayNames, y=weekLabels, customdata=extra,
        zmid=0, zmin=-limit, zmax=limit,
        colorscale=[[0.0, lossPole], [0.5, neutral], [1.0, profitPole]],
        # A 2px surface gap between fills, as the mark spec asks.
        xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(title=dict(text=style["label"], side="top"), thickness=10,
                      outlinewidth=0, tickcolor=axisLine, tickformat=style["tickformat"],
                      tickfont=dict(color=mutedInk)),
        hovertemplate=("%{customdata[0]} · fold %{customdata[1]}<br>"
                       + style["hover"]
                       + "<br>agent $%{customdata[2]:,.2f}"
                       "<br>oracle $%{customdata[3]:,.2f}<extra></extra>"),
    ))

    marked = days.highVol.to_numpy(dtype=bool)
    figure.add_trace(go.Scatter(
        x=[weekdayNames[index] for index in column[marked]],
        y=[weekLabels[index] for index in row[marked]],
        mode="markers", name="High volatility",
        marker=dict(symbol="circle", size=9, color=ink,
                    line=dict(color=surface, width=2)),
        hoverinfo="skip", showlegend=True,
    ))

    figure.update_layout(**_layout(
        height=140 + 44 * weeks, margin=dict(l=76, r=24, t=34, b=56),
        # The weekday row sits on top, so the legend goes under the grid rather than
        # colliding with it.
        legend=dict(orientation="h", yanchor="top", y=-0.04, x=0,
                    font=dict(color=secondaryInk)),
    ))
    figure.update_xaxes(side="top", showgrid=False, zeroline=False, showline=False,
                        tickfont=dict(color=mutedInk))
    figure.update_yaxes(autorange="reversed", showgrid=False, zeroline=False,
                        showline=False, tickfont=dict(color=mutedInk),
                        title=dict(text="Week starting",
                                   font=dict(color=mutedInk, size=12)))
    return figure


def dayFigure(trajectory, day, algo=None, arm=None):
    """One test day: the price paths on top, the state of charge below.

    Two panels rather than two y-scales. Price in $/MWh and state of charge in 0-1 do
    not share an axis. The oracle series the runner writes is kept whichever agent arm
    is selected, so the day always shows what perfect foresight would have done.
    """
    rows = trajectory
    oracle = rows[rows.arm == "oracle"] if not rows.empty else rows
    for column, wanted in (("algo", algo), ("arm", arm)):
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
        if side.empty or (source == "forecast" and realised.empty):
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
