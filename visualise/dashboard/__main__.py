"""The rolling backtest dashboard. One screen.

    uv run python -m visualise.dashboard --region NSW1

The summary at the top, the model and MDP beside it, and a calendar of every test day.
Click a day to narrow the summary to that day and draw its trajectory. Click the weekday
header area, or Whole job, to go back.
"""
import argparse

import pandas as pd
from dash import Dash, Input, Output, State, dcc, html

from visualise.dashboard import data, figures

tileStyle = {"flex": "1 1 150px", "padding": "14px 16px", "borderRadius": "10px",
             "border": f"1px solid {figures.gridline}",
             "background": figures.surface}
panelStyle = {"padding": "16px 20px", "borderRadius": "10px",
              "border": f"1px solid {figures.gridline}",
              "background": figures.surface}


def _tile(tile):
    return html.Div([
        html.Div(tile["label"], style={"color": figures.mutedInk, "fontSize": "12px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.04em"}),
        html.Div(tile["value"], style={"color": figures.ink, "fontSize": "26px",
                                       "fontWeight": 600, "marginTop": "4px"}),
        html.Div(tile["sub"], style={"color": figures.secondaryInk,
                                     "fontSize": "12px", "marginTop": "2px"}),
    ], style=tileStyle)


def _bucketTable(rows):
    """The volatility split as a table, which is also the relief for the colour scale."""
    header = ["", "Days", "Agent $", "Oracle $", "Captured", "Share of oracle $"]
    cells = [html.Tr([html.Th(name, style={"textAlign": "left" if not index else "right",
                                           "color": figures.mutedInk,
                                           "fontWeight": 500, "fontSize": "12px",
                                           "padding": "4px 10px"})
                      for index, name in enumerate(header)])]
    for row in rows:
        cells.append(html.Tr([
            html.Td(row["bucket"], style={"padding": "4px 10px"}),
            html.Td(f"{row['days']:,}", style=_number),
            html.Td(figures._money(row["agentProfit"]), style=_number),
            html.Td(figures._money(row["oracleProfit"]), style=_number),
            html.Td(figures._percent(row["captured"]), style=_number),
            html.Td(figures._percent(row["oracleShare"]), style=_number),
        ]))
    return html.Table(cells, style={"borderCollapse": "collapse", "width": "100%",
                                    "fontSize": "13px"})


_number = {"textAlign": "right", "padding": "4px 10px",
           "fontVariantNumeric": "tabular-nums"}


def _controls(choices):
    """One row of filters above the charts, built from what the data holds."""
    def dropdown(key, label):
        values = choices[key]
        return html.Div([
            html.Label(label, style={"color": figures.mutedInk, "fontSize": "12px"}),
            dcc.Dropdown(id=f"pick-{key}", options=[{"label": str(v), "value": v}
                                                    for v in values],
                         value=values[0], clearable=False,
                         style={"minWidth": "170px"}),
        ])
    return html.Div([dropdown("algo", "Algorithm"), dropdown("arm", "Arm"),
                     dropdown("priceSource", "Price source"),
                     html.Div([html.Label(" ", style={"fontSize": "12px",
                                                      "display": "block"}),
                               html.Button("Whole job", id="reset-scope",
                                           n_clicks=0,
                                           style={"padding": "7px 14px",
                                                  "borderRadius": "8px",
                                                  "cursor": "pointer",
                                                  "border": f"1px solid {figures.axisLine}",
                                                  "background": figures.surface})])],
                    style={"display": "flex", "gap": "14px", "alignItems": "flex-end",
                           "flexWrap": "wrap", "marginBottom": "18px"})


def layout(run):
    choices = data.selectors(run.days)
    return html.Div([
        dcc.Store(id="scope", data={"kind": "job", "day": None}),
        html.H1(f"{run.region} rolling backtest",
                style={"fontSize": "22px", "margin": "0 0 4px"}),
        html.Div(id="scope-label",
                 style={"color": figures.secondaryInk, "marginBottom": "18px"}),
        _controls(choices),
        html.Div(id="tiles", style={"display": "flex", "gap": "12px",
                                    "flexWrap": "wrap", "marginBottom": "18px"}),
        html.Div([
            html.Div([html.H2("By volatility", style=_h2),
                      html.Div(f"A day is high volatility when its "
                               f"{figures.highVolKey}.",
                               style={"color": figures.secondaryInk,
                                      "fontSize": "12px", "marginBottom": "10px"}),
                      html.Div(id="buckets")],
                     style={**panelStyle, "flex": "2 1 420px"}),
            html.Div([html.H2("Model and MDP", style=_h2),
                      html.Div(id="model")],
                     style={**panelStyle, "flex": "1 1 340px"}),
        ], style={"display": "flex", "gap": "14px", "marginBottom": "18px"}),
        html.Div([
            html.H2("Test days", style=_h2),
            html.Div(["Fill is profit, blue above zero and red below. A black dot marks "
                      "a high-volatility day. Click a day for its trajectory."],
                     style={"color": figures.secondaryInk, "fontSize": "12px",
                            "marginBottom": "6px"}),
            dcc.Graph(id="calendar", config={"displayModeBar": False}),
        ], style={**panelStyle, "marginBottom": "18px"}),
        html.Div([html.H2("Day view", style=_h2),
                  dcc.Graph(id="day", config={"displayModeBar": False})],
                 style=panelStyle),
    ], style={"maxWidth": "1180px", "margin": "0 auto", "padding": "28px 20px",
              "background": "#f9f9f7", "minHeight": "100vh",
              "fontFamily": 'system-ui, -apple-system, "Segoe UI", sans-serif',
              "color": figures.ink})


_h2 = {"fontSize": "14px", "margin": "0 0 8px", "color": figures.secondaryInk,
       "textTransform": "uppercase", "letterSpacing": "0.04em"}


wholeJob = {"kind": "job", "day": None}


def nextScope(triggered, click, current):
    """What the scope becomes after a click. Pure, so it is testable off the server.

    The high-volatility marker trace carries no customdata, so a click that lands on a
    dot must leave the scope where it was rather than clearing it.
    """
    if triggered == "reset-scope" or not click:
        return wholeJob
    point = click["points"][0]
    if not point.get("customdata"):
        return current
    return {"kind": "day", "day": str(point["customdata"][0])}


def register(app, run):
    @app.callback(Output("scope", "data"),
                  Input("calendar", "clickData"), Input("reset-scope", "n_clicks"),
                  State("scope", "data"), prevent_initial_call=True)
    def setScope(click, _resets, current):
        from dash import ctx
        return nextScope(ctx.triggered_id, click, current)

    @app.callback(Output("tiles", "children"), Output("buckets", "children"),
                  Output("model", "children"), Output("scope-label", "children"),
                  Output("calendar", "figure"), Output("day", "figure"),
                  Input("scope", "data"), Input("pick-algo", "value"),
                  Input("pick-arm", "value"), Input("pick-priceSource", "value"))
    def redraw(scope, algo, arm, source):
        chosen = run.days[(run.days.algo == algo) & (run.days.arm == arm)
                          & (run.days.priceSource == source)]
        day = None if scope["kind"] == "job" else pd.Timestamp(scope["day"]) \
            + pd.Timedelta(hours=4)
        scoped = data.scope(chosen, day=day)
        label = (f"{len(data.selectors(chosen)['seed'])} seed(s), whole job"
                 if day is None else f"{day:%A %-d %B %Y}, 04:00 to 04:00")
        dayFigure = (figures._emptyFigure("pick a day on the calendar") if day is None
                     else figures.dayFigure(data.trajectories(run, day), day,
                                            algo=algo, arm=arm))
        return (
            [_tile(tile) for tile in figures.summaryTiles(scoped)],
            _bucketTable(figures.volatilityTiles(scoped)),
            [html.Div([html.Span(name, style={"color": figures.mutedInk}),
                       html.Span(value, style={"marginLeft": "8px"})],
                      style={"fontSize": "13px", "padding": "3px 0"})
             for name, value in figures.modelPanel(run.manifest)],
            label,
            figures.calendarFigure(chosen),
            dayFigure,
        )


def build(region, resultsDir):
    run = data.loadRun(region, resultsDir)
    app = Dash(__name__, title=f"{region} rolling backtest")
    app.layout = layout(run)
    register(app, run)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="visualise.dashboard",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--region", default="NSW1")
    parser.add_argument("--results", default=str(data.defaultResults))
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)
    build(args.region, args.results).run(port=args.port)


if __name__ == "__main__":
    main()
