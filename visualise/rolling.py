"""Turn a rolling backtest result set into the Sections the plots already draw.

`visualise/parse.py` reads a `results/<REGION>.md` file, which `cli eval` writes for one
train/test split. The rolling backtest writes per-day rows instead, so a region that only
ever ran `cli rolling` has no such file. This module pools those rows into the same
`Section` shape, which lets one chart serve both experiments.

Pooling follows `backtest/metrics.py`: captured is a ratio of sums, never a mean of
ratios, and the oracle figure is `oracleDualBound`, never an incumbent `agentProfit`.
Seeds are kept apart here rather than averaged, because the chart names one seed.
"""
import json
from statistics import mean, pstdev

from backtest import records
from visualise.parse import Section, SeedResult

# The rolling rows name a price source; the results file names a section. One name per
# thing, so the headings here are the headings cli.py writes.
sourceHeadings = {"realised": "Realised prices", "forecast": "AEMO forecast"}


def _isoDay(timestamp):
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S")


def sectionsFromRolling(resultsDir, region, algo=None, trainMode=None):
    """One Section per price source in the region's folds, best seed included.

    `algo` and `trainMode` narrow the rows when a run holds more than one. A run with
    several of either and no filter raises, because pooling two algos into one bar
    reports a number no agent achieved.
    """
    frame = records.readDays(resultsDir, region)
    if frame.empty:
        raise FileNotFoundError(f"no rolling results for {region} in {resultsDir}")
    if algo is not None:
        frame = frame[frame.algo == algo]
    if trainMode is not None:
        frame = frame[frame.trainMode == trainMode]
    if frame.empty:
        raise ValueError(f"no rows left for {region} after filtering")
    for column in ("algo", "trainMode"):
        held = sorted(frame[column].unique().tolist())
        if len(held) > 1:
            raise ValueError(f"{region} holds several {column} values ({held}); "
                             f"pass {column} to pick one")

    algoName = frame.algo.iloc[0]
    sections = []
    for source in sorted(frame.priceSource.unique().tolist()):
        rows = frame[frame.priceSource == source]
        sections.append(_section(rows, region, algoName, source))
    return sections


def _section(rows, region, algo, source):
    """One price source's rows, pooled over every fold and day, split by seed."""
    # The oracle bound belongs to the day, not the seed, so count each day once.
    perDay = rows.groupby("testDay").oracleDualBound.first()
    oracleProfit = float(perDay.sum())

    seeds = []
    for seed in sorted(rows.seed.unique().tolist()):
        agentProfit = float(rows[rows.seed == seed].agentProfit.sum())
        seeds.append(SeedResult(
            seed=int(seed),
            agentProfit=agentProfit,
            regret=oracleProfit - agentProfit,
            captured=agentProfit / oracleProfit if oracleProfit else float("nan"),
        ))

    profits = [s.agentProfit for s in seeds]
    regrets = [s.regret for s in seeds]
    captured = [s.captured for s in seeds]
    return Section(
        heading=sourceHeadings.get(source, source),
        region=region,
        algo=algo,
        windowStart=_isoDay(rows.testDay.min()),
        windowEnd=_isoDay(rows.testDay.max()),
        oracleProfit=oracleProfit,
        agentProfitMean=mean(profits),
        agentProfitSd=pstdev(profits),
        regretMean=mean(regrets),
        regretSd=pstdev(regrets),
        capturedMean=mean(captured),
        capturedSd=pstdev(captured),
        seeds=seeds,
    )


def trainNote(resultsDir, region):
    """How a rolling run trained, in one line for a chart title. The run retrains before
    every test week, so there is no one train window to print. Empty when the manifest
    does not say."""
    path = records.manifestPath(resultsDir, region)
    if not path.exists():
        return ""
    manifest = json.loads(path.read_text())
    weeks = manifest.get("trainWeeks")
    if not weeks:
        return ""
    return f"rolling {weeks} weeks before each test week"
