"""Parse a results/<REGION>.md file into structured data for plotting.

The file format is owned by cli.py's `_resultSection`/`_spliceSection`: one
`## heading` section per price source (e.g. "Realised prices", "AEMO forecast"),
each holding a few `- key: value` lines, a one-row-per-metric summary table, and a
per-seed table. This module only reads that format; it never runs an eval itself.
"""
import re
from dataclasses import dataclass


@dataclass
class SeedResult:
    seed: int
    agentProfit: float
    regret: float
    captured: float  # fraction, e.g. 0.1173 for "11.73%"


@dataclass
class Section:
    heading: str  # "Realised prices" or "AEMO forecast"
    region: str
    algo: str
    windowStart: str  # ISO timestamp, as written -- this is the window scored on
    windowEnd: str
    oracleProfit: float
    agentProfitMean: float
    agentProfitSd: float
    regretMean: float
    regretSd: float
    capturedMean: float  # fraction
    capturedSd: float
    seeds: list[SeedResult]

    def bestSeed(self):
        """The seed with the highest agent profit."""
        return max(self.seeds, key=lambda s: s.agentProfit)


_KV = re.compile(r"^- ([\w ]+): (.+)$")
_WINDOW = re.compile(r"^(\S+) \.\. (\S+)$")
_METRIC_ROW = re.compile(r"^\| (oracle profit|agent profit|regret|captured) \| (.+) \|$")
_SEED_ROW = re.compile(r"^\| (-?\d+) \| (-?[\d.]+) \| (-?[\d.]+) \| (-?[\d.]+)% \|$")
_MEAN_SD = re.compile(r"^(-?[\d.]+)%? \+/- (-?[\d.]+)%?$")


def _percent(number_text):
    return float(number_text) / 100.0


def parseSections(text):
    """One Section per `## heading` block. Raises if a section is missing a field
    the plots need -- a half-written results file should fail loudly, not plot gaps."""
    blocks = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    sections = []
    for block in blocks:
        heading, _, body = block.partition("\n")
        sections.append(_parseSection(heading.strip(), body))
    return sections


def _parseSection(heading, body):
    region = algo = windowStart = windowEnd = None
    oracleProfit = agentMean = agentSd = regretMean = regretSd = None
    capturedMean = capturedSd = None
    seeds = []

    for line in body.splitlines():
        kv = _KV.match(line)
        if kv:
            key, value = kv.groups()
            if key == "region":
                region = value
            elif key == "algo":
                algo = value
            elif key == "window":
                match = _WINDOW.match(value)
                if match:
                    windowStart, windowEnd = match.groups()
            continue

        metric = _METRIC_ROW.match(line)
        if metric:
            name, value = metric.groups()
            if name == "oracle profit":
                oracleProfit = float(value)
            elif name == "agent profit":
                agentMean, agentSd = (float(v) for v in _MEAN_SD.match(value).groups())
            elif name == "regret":
                regretMean, regretSd = (float(v) for v in _MEAN_SD.match(value).groups())
            elif name == "captured":
                capturedMean, capturedSd = (
                    _percent(v) for v in _MEAN_SD.match(value).groups()
                )
            continue

        seedRow = _SEED_ROW.match(line)
        if seedRow:
            seed, profit, regret, captured = seedRow.groups()
            seeds.append(SeedResult(
                seed=int(seed), agentProfit=float(profit), regret=float(regret),
                captured=_percent(captured),
            ))

    missing = [
        name for name, value in [
            ("region", region), ("algo", algo),
            ("window", windowStart and windowEnd), ("oracle profit", oracleProfit),
            ("agent profit", agentMean), ("regret", regretMean),
            ("captured", capturedMean),
        ] if value is None
    ]
    if missing or not seeds:
        raise ValueError(
            f"section {heading!r} is missing {missing or 'a per-seed table'}; "
            "half-written results file?"
        )

    return Section(
        heading=heading, region=region, algo=algo,
        windowStart=windowStart, windowEnd=windowEnd, oracleProfit=oracleProfit,
        agentProfitMean=agentMean, agentProfitSd=agentSd,
        regretMean=regretMean, regretSd=regretSd,
        capturedMean=capturedMean, capturedSd=capturedSd, seeds=seeds,
    )


def loadResultsFile(path):
    with open(path) as handle:
        return parseSections(handle.read())
