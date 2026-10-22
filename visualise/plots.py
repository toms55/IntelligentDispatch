"""Build matplotlib figures from parsed results/<REGION>.md sections.

Three views, one figure: per-seed agent profit against the oracle, per-seed captured
share, and a mean +/- SD summary across sections (e.g. realised prices vs the AEMO
forecast). Matplotlib is imported lazily and switched to Agg before any pyplot use
when `destination` is set, matching cli.py's own `_plot`.
"""


def _barOffsets(groupCount, width=0.8):
    """x offsets for `groupCount` bars sharing one tick, centred on that tick."""
    import numpy as np

    groupWidth = width / groupCount
    offsets = (np.arange(groupCount) - (groupCount - 1) / 2) * groupWidth
    return offsets, groupWidth


def plotPerSeedProfit(sections, ax):
    """Grouped bars: agent profit per seed, one group per section, oracle as a line."""
    offsets, barWidth = _barOffsets(len(sections))
    oracleProfit = sections[0].oracleProfit
    for offset, section in zip(offsets, sections):
        seeds = [s.seed for s in section.seeds]
        profits = [s.agentProfit for s in section.seeds]
        ax.bar([x + offset for x in seeds], profits, width=barWidth, label=section.heading)
    ax.axhline(oracleProfit, color="black", linestyle="--", linewidth=1,
               label=f"oracle profit ({oracleProfit:.0f})")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("seed")
    ax.set_ylabel("agent profit")
    ax.set_title("Per-seed profit vs. the oracle")
    ax.set_xticks(seeds)
    ax.legend(fontsize="small")


def plotPerSeedCaptured(sections, ax):
    """Grouped bars: captured share of the oracle's profit, per seed per section."""
    offsets, barWidth = _barOffsets(len(sections))
    for offset, section in zip(offsets, sections):
        seeds = [s.seed for s in section.seeds]
        captured = [s.captured * 100 for s in section.seeds]
        ax.bar([x + offset for x in seeds], captured, width=barWidth, label=section.heading)
    ax.axhline(100, color="black", linestyle="--", linewidth=1, label="oracle (100%)")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("seed")
    ax.set_ylabel("captured (%)")
    ax.set_title("Per-seed share of oracle profit captured")
    ax.set_xticks(seeds)
    ax.legend(fontsize="small")


def plotSummary(sections, ax):
    """One bar per section: mean agent profit +/- SD across seeds, oracle as a line."""
    import numpy as np

    labels = [section.heading for section in sections]
    means = [section.agentProfitMean for section in sections]
    sds = [section.agentProfitSd for section in sections]
    x = np.arange(len(sections))
    ax.bar(x, means, yerr=sds, capsize=4, width=0.5)
    ax.axhline(sections[0].oracleProfit, color="black", linestyle="--", linewidth=1,
               label=f"oracle profit ({sections[0].oracleProfit:.0f})")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("agent profit (mean +/- SD across seeds)")
    ax.set_title("Mean profit by price source")
    ax.legend(fontsize="small")


def buildFigure(sections, region, algo):
    """One figure, three panels, for one region's results file."""
    import matplotlib.pyplot as plt

    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(15, 4.5))
    plotPerSeedProfit(sections, left)
    plotPerSeedCaptured(sections, middle)
    plotSummary(sections, right)
    figure.suptitle(f"{algo} on {region}")
    figure.tight_layout()
    return figure


def saveFigure(sections, region, algo, destination):
    """Render buildFigure to PATH. Switches to the Agg backend first, matching
    cli.py's _plot, since the default backend needs a display that may not exist."""
    import matplotlib
    matplotlib.use("Agg")

    figure = buildFigure(sections, region, algo)
    figure.savefig(destination, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(figure)


def _bestAcrossSections(sections):
    """The (section, seed) pair with the single highest agent profit, over every
    section -- e.g. picks the better of a realised-price and a forecast run."""
    candidates = [(section, section.bestSeed()) for section in sections]
    return max(candidates, key=lambda pair: pair[1].agentProfit)


def _shortTimestamp(isoTimestamp):
    """'2025-01-01T04:00:00' -> '2025-01-01 04:00'."""
    date, _, time = isoTimestamp.partition("T")
    return f"{date} {time[:5]}" if time else date


def plotBestSeedVsOracle(sections, ax, source=None):
    """Two bars: the best seed's profit against the oracle's, for the same window.

    `source` picks one section by a substring of its heading (e.g. "realised" or
    "forecast"); omitted, it is the best seed across every section in the file.
    Returns the (section, seed) it plotted, so the caller can build a title from it.
    """
    if source is not None:
        section = next(s for s in sections if source.lower() in s.heading.lower())
        seed = section.bestSeed()
    else:
        section, seed = _bestAcrossSections(sections)

    labels = [f"seed {seed.seed}\n({section.heading})", "oracle"]
    values = [seed.agentProfit, section.oracleProfit]
    bars = ax.bar(labels, values, color=["tab:blue", "black"], width=0.5)
    ax.bar_label(bars, fmt="%.0f")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_ylabel("profit")
    ax.set_title(f"best seed captures {seed.captured:.1%} of the oracle's profit")
    return section, seed


def _windowTitle(section, region, algo, trainWindow=None):
    """The three-line header over a best-seed panel: the algo and region, the test
    window from the results file, and the train window if the caller gives one. The
    results file never records what a checkpoint trained on, only what it was scored
    on, so the caller has to supply that half.

    `trainWindow` is a (start, end) pair for one fixed split. It is a plain string for
    a rolling backtest, which retrains before every test week and so has no one window
    to print.
    """
    testRange = (
        f"{_shortTimestamp(section.windowStart)} to {_shortTimestamp(section.windowEnd)}"
    )
    title = f"{algo} on {region}\ntest: {testRange}"
    if isinstance(trainWindow, str):
        title += f"\ntrain: {trainWindow}"
    elif trainWindow:
        trainStart, trainEnd = trainWindow
        title += f"\ntrain: {_shortTimestamp(trainStart)} to {_shortTimestamp(trainEnd)}"
    return title


def buildBestSeedFigure(sections, region, algo, trainWindow=None, source=None):
    """One figure, one panel: the single best seed against the oracle."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(5, 5.5))
    section, seed = plotBestSeedVsOracle(sections, ax, source=source)
    figure.suptitle(_windowTitle(section, region, algo, trainWindow))
    figure.tight_layout()
    return figure


def buildSourceComparisonFigure(sections, region, algo, trainWindow=None,
                                sources=("realised", "forecast")):
    """One figure, one panel per price source, side by side: each source's best seed
    against the oracle. Every panel keeps its own header and its own captured share,
    so the realised-price run and the forecast run read as the same chart twice.

    Each panel is a subfigure rather than a plain subplot, because a panel needs two
    stacked titles -- the window header above the captured share -- and only a
    subfigure carries a suptitle of its own.
    """
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(5 * len(sources), 5.5))
    panels = figure.subfigures(1, len(sources))
    if len(sources) == 1:
        panels = [panels]
    for panel, source in zip(panels, sources):
        ax = panel.subplots()
        section, _ = plotBestSeedVsOracle(sections, ax, source=source)
        panel.suptitle(_windowTitle(section, region, algo, trainWindow), y=0.98,
                       verticalalignment="top")
        # tight_layout cannot see a subfigure suptitle, so the axes is placed by hand:
        # the header needs the top fifth, and the y label the left margin.
        panel.subplots_adjust(top=0.80, bottom=0.12, left=0.17, right=0.97)
    return figure


def saveSourceComparisonFigure(sections, region, algo, destination, trainWindow=None,
                               sources=("realised", "forecast")):
    """Render buildSourceComparisonFigure to PATH. Agg backend first, as saveFigure
    does."""
    import matplotlib
    matplotlib.use("Agg")

    figure = buildSourceComparisonFigure(
        sections, region, algo, trainWindow=trainWindow, sources=sources
    )
    figure.savefig(destination, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(figure)


def saveBestSeedFigure(sections, region, algo, destination, trainWindow=None, source=None):
    """Render buildBestSeedFigure to PATH. Agg backend first, as saveFigure does."""
    import matplotlib
    matplotlib.use("Agg")

    figure = buildBestSeedFigure(
        sections, region, algo, trainWindow=trainWindow, source=source
    )
    figure.savefig(destination, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(figure)
