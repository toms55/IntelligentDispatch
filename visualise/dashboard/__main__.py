"""The rolling backtest dashboard. One screen.

    uv run python -m visualise.dashboard --region NSW1

Pick a model and a region at the top. Below them a date field, a day stepper and a
calendar button drive one board, which shows either the month or one day's dispatch,
never both. Click a day in the month to open it.
"""
import argparse

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html

from visualise.dashboard import data, figures

tileStyle = {"flex": "1 1 150px", "padding": "14px 16px", "borderRadius": "10px",
             "border": f"1px solid {figures.gridline}",
             "background": figures.surface}
panelStyle = {"padding": "16px 20px", "borderRadius": "10px",
              "border": f"1px solid {figures.gridline}",
              "background": figures.surface}
buttonStyle = {"padding": "7px 14px", "borderRadius": "8px", "cursor": "pointer",
               "border": f"1px solid {figures.axisLine}",
               "background": figures.surface, "color": figures.ink}
labelStyle = {"color": figures.mutedInk, "fontSize": "12px", "display": "block",
              "marginBottom": "4px"}
dateFormat = "%d/%m/%Y"


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


def parseDay(text):
    """A typed DD/MM/YYYY date as an ISO day, or None when it is not a date yet."""
    # An empty string parses to NaT rather than raising, so the result is checked too.
    try:
        parsed = pd.to_datetime(str(text).strip(), format=dateFormat)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(parsed) else str(parsed.date())


def showDay(day):
    return pd.Timestamp(day).strftime(dateFormat)


def nextState(triggered, state, click=None, text=None, region=None, models=()):
    """What the state becomes after a control fires. Pure, so it needs no server.

    A click is only a day selection while the board shows the month. The same graph
    draws the day view, and a click there must not move the day under the reader.
    """
    state = dict(state)
    if triggered == "pick-region" and region:
        state["region"] = region
        state["view"] = "calendar"
    elif triggered == "show-calendar":
        state["view"] = "calendar"
    elif triggered in ("step-prev", "step-next"):
        step = -1 if triggered == "step-prev" else 1
        state["day"] = str((pd.Timestamp(state["day"]) + pd.Timedelta(days=step)).date())
    elif triggered == "day-text":
        parsed = parseDay(text)
        if parsed is not None:
            state["day"] = parsed
    elif triggered == "board" and click and state["view"] == "calendar":
        point = click["points"][0]
        if point.get("customdata"):
            state["day"] = str(point["customdata"][0])
            state["view"] = "day"
    if state.get("model") not in models and models:
        state["model"] = models[0]
    return state


def _controls(regions, region):
    return html.Div([
        html.Div([
            html.Div([html.Label("Model", style=labelStyle),
                      dcc.Dropdown(id="pick-model", clearable=False,
                                   style={"minWidth": "260px"})]),
            html.Div([html.Label("Region", style=labelStyle),
                      dcc.Dropdown(id="pick-region", value=region, clearable=False,
                                   options=[{"label": name, "value": name}
                                            for name in regions],
                                   style={"minWidth": "120px"})]),
        ], style={"display": "flex", "gap": "14px", "justifyContent": "flex-end",
                  "marginBottom": "20px", "flexWrap": "wrap"}),
        html.Div([
            html.Div([html.Label("Day", style=labelStyle),
                      dcc.Input(id="day-text", type="text", debounce=True,
                                placeholder=dateFormat.replace("%d", "DD")
                                .replace("%m", "MM").replace("%Y", "YYYY"),
                                style={**buttonStyle, "width": "140px"})]),
            html.Div([html.Label(" ", style=labelStyle),
                      html.Div([html.Button("<", id="step-prev", n_clicks=0,
                                            style=buttonStyle),
                                html.Button(">", id="step-next", n_clicks=0,
                                            style=buttonStyle)],
                               style={"display": "flex", "gap": "6px"})]),
            html.Div([html.Label(" ", style=labelStyle),
                      html.Button("Calendar", id="show-calendar", n_clicks=0,
                                  style=buttonStyle)]),
            html.Div(id="board-label",
                     style={"color": figures.secondaryInk, "fontSize": "15px",
                            "alignSelf": "flex-end", "paddingBottom": "7px"}),
        ], style={"display": "flex", "gap": "14px", "alignItems": "flex-end",
                  "flexWrap": "wrap", "marginBottom": "18px"}),
    ])


def layout(regions, region, day):
    return html.Div([
        dcc.Store(id="state", data={"region": region, "day": day, "view": "calendar",
                                    "model": None}),
        _controls(regions, region),
        html.Div(id="tiles", style={"display": "flex", "gap": "12px",
                                    "flexWrap": "wrap", "marginBottom": "18px"}),
        html.Div([dcc.Graph(id="board", config={"displayModeBar": False})],
                 style=panelStyle),
    ], style={"maxWidth": "1180px", "margin": "0 auto", "padding": "28px 20px",
              "background": "#f9f9f7", "minHeight": "100vh",
              "fontFamily": 'system-ui, -apple-system, "Segoe UI", sans-serif',
              "color": figures.ink})


def register(app, loader):
    @app.callback(Output("pick-model", "options"), Output("pick-model", "value"),
                  Input("pick-region", "value"), State("pick-model", "value"))
    def fillModels(region, current):
        names = data.models(loader(region).days)
        return ([{"label": name, "value": name} for name in names],
                current if current in names else (names[0] if names else None))

    @app.callback(Output("state", "data"), Output("day-text", "value"),
                  Input("board", "clickData"), Input("step-prev", "n_clicks"),
                  Input("step-next", "n_clicks"), Input("show-calendar", "n_clicks"),
                  Input("day-text", "value"), Input("pick-region", "value"),
                  Input("pick-model", "value"), State("state", "data"))
    def setState(click, _prev, _next, _calendar, text, region, model, state):
        names = data.models(loader(region).days)
        state = dict(state, model=model)
        state = nextState(ctx.triggered_id, state, click=click, text=text,
                          region=region, models=names)
        return state, showDay(state["day"])

    @app.callback(Output("tiles", "children"), Output("board", "figure"),
                  Output("board-label", "children"), Input("state", "data"))
    def redraw(state):
        run = loader(state["region"])
        chosen = data.modelRows(run.days, state["model"])
        day = pd.Timestamp(state["day"]) + pd.Timedelta(hours=data.dayStartHour)
        # The tiles report the selected day in both views, so stepping the day or
        # clicking a calendar cell always moves them.
        scoped = data.scope(chosen, day=day)
        if state["view"] == "calendar":
            board = figures.monthFigure(data.monthScope(chosen, day), day,
                                        selected=state["day"])
            label = f"{day:%B %Y}"
        else:
            row = chosen.iloc[0] if not chosen.empty else {}
            board = figures.dayFigure(
                data.trajectories(run, day), day,
                algo=row.get("algo"), trainMode=row.get("trainMode"),
                priceSource=row.get("priceSource"))
            label = f"{day:%A %-d %B %Y}, 04:00 to 04:00"
        return [_tile(tile) for tile in figures.summaryTiles(scoped)], board, label


def build(region, resultsDir):
    runs = {}

    def loader(name):
        if name not in runs:
            runs[name] = data.loadRun(name, resultsDir)
        return runs[name]

    regions = data.availableRegions(resultsDir) or [region]
    first = loader(region)
    day = str(pd.Timestamp(first.days.testDay.max()).date())
    app = Dash(__name__, title=f"{region} rolling backtest")
    app.layout = layout(regions, region, day)
    register(app, loader)
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
