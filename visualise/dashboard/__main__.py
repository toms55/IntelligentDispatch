"""The dashboard. Two screens, picked by the tabs at the top.

    uv run python -m visualise.dashboard

It reads the one run under `results/`. Pass `--results DIR` when that directory holds
more than one run.

Backtest. Pick a model and a region at the top. Below them a date field, a stepper and
a calendar button drive one board, which shows either the month or one day's dispatch,
never both. The stepper moves what the board shows, a month or a day, and the label
above it names which. Click a day in the month to open it.

Current forecast. Pick a model and a region, and the board draws the AEMO price path
published right now, with the dispatch that model would run over it. This screen runs
the agent itself, so its model list is what `trainedPolicies/` holds on this machine,
not what the backtest scored. See `visualise/dashboard/live.py` for what the two
screens do not share.
"""
import argparse
import calendar as calendarModule

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from visualise.dashboard import data, figures, live

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
    # A tile whose value is words rather than a number sets them smaller, so a phrase
    # like "Refreshed every 4 hours" fits the width a dollar figure does. An accent is
    # the colour of the chart line the tile names.
    accent = tile.get("accent")
    style = dict(tileStyle, borderLeft=f"4px solid {accent}") if accent else tileStyle
    return html.Div([
        html.Div(tile["label"], style={"color": figures.mutedInk, "fontSize": "12px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.04em"}),
        html.Div(tile["value"], style={"color": accent or figures.ink,
                                       "fontSize": "26px" if tile.get("raw") is not None
                                       else "18px",
                                       "fontWeight": 600, "marginTop": "4px"}),
        html.Div(tile["sub"], style={"color": figures.secondaryInk,
                                     "fontSize": "12px", "marginTop": "2px"}),
    ], style=style)


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


def clickedDay(click, month):
    """The date a click on the calendar landed on, or None when it hit no cell.

    The cell is found from the row and the column, not from the cell's own date. Dash
    reads a point's `customdata` out of the trace by the point number, and a heatmap
    point number is a [row, column] pair rather than one integer, so Dash drops the
    date the cell carries. The row and the column always arrive, and the board draws
    the month of `month`, which is enough to name the day.

    A click on an untested cell never reaches here: `hoverongaps=False` stops the
    figure from reporting a cell that holds no profit.
    """
    point = click["points"][0]
    if point.get("customdata"):
        return str(point["customdata"][0])
    row, column = point.get("y"), point.get("x")
    if row is None or column is None:
        return None
    first = pd.Timestamp(month).normalize().replace(day=1)
    # Rounded, because the date and the volatility dot of a cell sit a fraction of a
    # row above and below the centre of it.
    number = round(row) * 7 + round(column) - int(first.dayofweek) + 1
    if not 1 <= number <= calendarModule.monthrange(first.year, first.month)[1]:
        return None
    return str(first.replace(day=number).date())


def stepUnit(view):
    """What the arrows move, which is what the board shows. The calendar draws a month,
    so they move a month. The dispatch view draws one day, so they move one day."""
    return "month" if view == "calendar" else "day"


def stepLabels(view):
    """The stepper's label and the two button tooltips, which all name the same unit.

    The arrows carry no unit of their own, so the screen names it. The label sits above
    them, where every other control keeps its label.
    """
    unit = stepUnit(view)
    return f"Step {unit}", f"Previous {unit}", f"Next {unit}"


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
        # `DateOffset` clamps the day to the month it lands in, so a month step off
        # 31 January reaches 28 February and does not run into March.
        unit = f"{stepUnit(state['view'])}s"
        moved = pd.Timestamp(state["day"]) + pd.DateOffset(**{unit: step})
        state["day"] = str(moved.date())
    elif triggered == "day-text":
        parsed = parseDay(text)
        if parsed is not None:
            state["day"] = parsed
    elif triggered == "board" and click and state["view"] == "calendar":
        chosen = clickedDay(click, state["day"])
        if chosen is not None:
            state["day"] = chosen
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
            html.Div([html.Label(id="step-label", style=labelStyle),
                      html.Div([html.Button("<", id="step-prev", n_clicks=0,
                                            style=buttonStyle),
                                html.Button(">", id="step-next", n_clicks=0,
                                            style=buttonStyle)],
                               style={"display": "flex", "gap": "6px"})]),
            html.Div([html.Label(" ", style=labelStyle),
                      html.Button("Calendar", id="show-calendar", n_clicks=0,
                                  style=buttonStyle)],
                     id="calendar-control", style={"display": "none"}),
            html.Div(id="board-label",
                     style={"color": figures.secondaryInk, "fontSize": "15px",
                            "alignSelf": "flex-end", "paddingBottom": "7px"}),
        ], style={"display": "flex", "gap": "14px", "alignItems": "flex-end",
                  "flexWrap": "wrap", "marginBottom": "18px"}),
    ])


def _liveControls(regions, region):
    """The forecast screen's own controls. The date field, the stepper and the calendar
    button have no meaning here, so the box in their place names the run instead."""
    return html.Div([
        html.Div([
            # Bottom aligned with the dropdowns, and as tall as one (Dash's is 36px).
            html.Button("Refresh", id="forecast-refresh", n_clicks=0,
                        style={**buttonStyle, "height": "36px"}),
            html.Div([html.Label("Model", style=labelStyle),
                      dcc.Dropdown(id="pick-live-model", clearable=False,
                                   style={"minWidth": "260px"})]),
            html.Div([html.Label("Region", style=labelStyle),
                      dcc.Dropdown(id="pick-live-region", value=region, clearable=False,
                                   options=[{"label": name, "value": name}
                                            for name in regions],
                                   style={"minWidth": "120px"})]),
        ], style={"display": "flex", "gap": "14px", "justifyContent": "flex-end",
                  "alignItems": "flex-end", "marginBottom": "20px", "flexWrap": "wrap"}),
        html.Div([
            html.Div(id="forecast-start", style={**panelStyle, "minWidth": "300px"}),
            # The cards take the rest of the row, which is empty space otherwise.
            html.Div(id="forecast-tiles",
                     style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                            "flex": "1 1 320px"}),
        ], style={"display": "flex", "gap": "14px", "alignItems": "stretch",
                  "flexWrap": "wrap", "marginBottom": "18px"}),
    ])


def layout(regions, region, day):
    return html.Div([
        dcc.Store(id="state", data={"region": region, "day": day, "view": "calendar",
                                    "model": None}),
        dcc.Tabs(id="screen", value="backtest", style={"marginBottom": "20px"},
                 children=[dcc.Tab(label="Backtest", value="backtest"),
                           dcc.Tab(label="Current forecast", value="live")]),
        # Both screens stay mounted and one is hidden, so a tab change costs no rebuild
        # and the day a reader opened is still there when they come back.
        html.Div(id="backtest-screen", children=[
            _controls(regions, region),
            html.Div(id="tiles", style={"display": "flex", "gap": "12px",
                                        "flexWrap": "wrap", "marginBottom": "18px"}),
            html.Div([dcc.Graph(id="board", config={"displayModeBar": False})],
                     style=panelStyle),
        ]),
        dcc.Store(id="live-state",
                  data={"screen": "backtest", "region": region, "model": None,
                        "clicks": 0}),
        html.Div(id="live-screen", style={"display": "none"}, children=[
            _liveControls(regions, region),
            html.Div([dcc.Graph(id="forecast-board",
                                config={"displayModeBar": False})],
                     style=panelStyle),
        ]),
    ], style={"maxWidth": "1180px", "margin": "0 auto", "padding": "28px 20px",
              "background": "#f9f9f7", "minHeight": "100vh",
              "fontFamily": 'system-ui, -apple-system, "Segoe UI", sans-serif',
              "color": figures.ink})


def liveModels(region, policyRoot=live.defaultCheckpoints):
    """The forecast screen's model list: what `trainedPolicies/` holds for `region`.

    Not the backtest screen's list. That one names runs already scored and needs no
    weights, so it offers models this screen cannot load. A machine that holds no
    agent for `region` gets an empty list, and the board then draws the prices alone.
    """
    return sorted(live.findCheckpoints(region, policyRoot))


def forecastBoard(region, model, policyRoot):
    """The forecast figure, the box above it and its two cards.

    Returns (figure, box, tiles).

    The two halves fail apart. The prices are AEMO's and need no agent, so a model
    whose checkpoint is not on this machine still gets its price path, with the reason
    the blocks are missing written over it. Only a failed fetch empties the board, and
    that is a network fault: AEMO has not published the run, or NEMWeb is unreachable.
    """
    empty = pd.DataFrame()
    try:
        path = live.forecastPath(region)
    except Exception as error:  # noqa: BLE001 - the message is the output
        return (figures.forecastFigure(empty, note=str(error)), _liveBox(None),
                figures.forecastTiles(empty))
    note = None
    try:
        frame = live.suggest(path, model, region, root=policyRoot)
    except (FileNotFoundError, ValueError) as error:
        frame, note = live.priceFrame(path), str(error)
    figure = figures.forecastFigure(frame, live.blocks(frame), region=region,
                                    model=model, note=note)
    return (figure, _liveBox(None if frame.empty else frame.settlementdate.iloc[0]),
            figures.forecastTiles(frame))


def _liveBox(start):
    """The box in the calendar button's place: when the forecast path starts."""
    if start is None:
        return [html.Div("Forecast", style=labelStyle),
                html.Div("no run", style={"fontSize": "20px", "fontWeight": 600})]
    start = pd.Timestamp(start)
    return [
        html.Div("Forecast begins", style=labelStyle),
        html.Div(start.strftime(f"{dateFormat} %H:%M"),
                 style={"fontSize": "20px", "fontWeight": 600}),
        html.Div("NEM time", style={"color": figures.mutedInk, "fontSize": "12px",
                                    "marginTop": "2px"}),
    ]


def register(app, loader, policyRoot=live.defaultCheckpoints):
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

    # The stepper is named in a callback of its own, so `redraw` keeps the three
    # outputs the board needs and does not redraw a board for the sake of a label.
    @app.callback(Output("step-label", "children"), Output("step-prev", "title"),
                  Output("step-next", "title"), Output("calendar-control", "style"),
                  Input("state", "data"))
    def nameTheStep(state):
        # The calendar button only leads back from a day; on the calendar it does nothing.
        shown = {} if state["view"] == "day" else {"display": "none"}
        return (*stepLabels(state["view"]), shown)

    @app.callback(Output("tiles", "children"), Output("board", "figure"),
                  Output("board-label", "children"), Input("state", "data"))
    def redraw(state):
        run = loader(state["region"])
        chosen = data.modelRows(run.days, state["model"])
        day = pd.Timestamp(state["day"]) + pd.Timedelta(hours=data.dayStartHour)
        # The tiles report the selected day in both views, so stepping the day or
        # clicking a calendar cell always moves them.
        scoped = data.scope(chosen, day=day)
        row = chosen.iloc[0] if not chosen.empty else {}
        # The forecast comes from the run's own actions export, so the line on the
        # chart and the refresh tile both follow --results.
        forecast = (data.forecastPath(run.resultsDir, state["region"], day,
                                      algo=row.get("algo"), trainMode=row.get("trainMode"),
                                      seed=row.get("seed"))
                    if row.get("priceSource") == "forecast" else None)
        # Verified against the export by `data.loadRun`, so the tile states it as fact.
        every = run.manifest.get("forecastEvery")
        if state["view"] == "calendar":
            board = figures.monthFigure(data.monthScope(chosen, day), day,
                                        selected=state["day"])
            label = f"{day:%B %Y}"
        else:
            # `live.blocks` turns the actions into blocks for both screens, so a
            # backtest day and a live forecast read the same way.
            path = data.dispatchPath(
                data.trajectories(run, day),
                algo=row.get("algo"), trainMode=row.get("trainMode"),
                seed=row.get("seed"), priceSource=row.get("priceSource"))
            board = figures.dayFigure(path, live.blocks(path), day,
                                      region=state["region"], model=state["model"],
                                      forecast=forecast, every=every)
            label = f"{day:%A %-d %B %Y}, 04:00 to 04:00"
        return [_tile(tile) for tile in figures.summaryTiles(scoped, every)], board, label

    @app.callback(Output("backtest-screen", "style"), Output("live-screen", "style"),
                  Input("screen", "value"))
    def showScreen(screen):
        hidden, shown = {"display": "none"}, {}
        return (hidden, shown) if screen == "live" else (shown, hidden)

    # One callback settles the region and the model together, as the backtest screen
    # does. Two callbacks would let the board fetch a model against a region that does
    # not hold it, which is one wasted NEMWeb read per region change.
    @app.callback(Output("live-state", "data"), Output("pick-live-model", "options"),
                  Output("pick-live-model", "value"),
                  Input("screen", "value"), Input("pick-live-region", "value"),
                  Input("pick-live-model", "value"),
                  Input("forecast-refresh", "n_clicks"))
    def setLiveState(screen, region, model, clicks):
        names = liveModels(region, policyRoot)
        model = model if model in names else (names[0] if names else None)
        return ({"screen": screen, "region": region, "model": model, "clicks": clicks},
                [{"label": name, "value": name} for name in names], model)

    # The forecast is fetched over the network, so it is drawn when the screen is
    # opened and on demand, never on every callback the other screen fires.
    @app.callback(Output("forecast-board", "figure"),
                  Output("forecast-start", "children"),
                  Output("forecast-tiles", "children"),
                  Input("live-state", "data"))
    def redrawForecast(state):
        if state["screen"] != "live":
            raise PreventUpdate
        figure, box, tiles = forecastBoard(state["region"], state["model"], policyRoot)
        return figure, box, [_tile(tile) for tile in tiles]


def build(region, resultsDir, policyRoot=live.defaultCheckpoints):
    runs = {}

    def loader(name):
        if name not in runs:
            runs[name] = data.loadRun(name, resultsDir)
        return runs[name]

    regions = data.availableRegions(resultsDir) or [region]
    # A run the agent never trained on NSW1 for still opens, on a region it holds.
    region = region if region in regions else regions[0]
    first = loader(region)
    day = str(pd.Timestamp(first.days.testDay.max()).date())
    app = Dash(__name__, title=f"{region} rolling backtest")
    app.layout = layout(regions, region, day)
    register(app, loader, policyRoot)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="visualise.dashboard",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--region", default="NSW1")
    parser.add_argument("--results", default=None,
                        help="the backtest records to read (default: the one run "
                             f"under {data.resultsRoot}/)")
    parser.add_argument("--policies", default=str(live.defaultCheckpoints),
                        help="where training published its final agents "
                             f"(default: {live.defaultCheckpoints})")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)
    resultsDir = data.findResults(args.results)
    print(f"reading {resultsDir}")
    build(args.region, resultsDir, args.policies).run(port=args.port)


if __name__ == "__main__":
    main()
