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
    oracleProfit: float
    agentProfitMean: float
    agentProfitSd: float
    regretMean: float
    regretSd: float
    capturedMean: float  # fraction
    capturedSd: float
    seeds: list[SeedResult]


_KV = re.compile(r"^- ([\w ]+): (.+)$")
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
    region = algo = None
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
            ("region", region), ("algo", algo), ("oracle profit", oracleProfit),
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
        heading=heading, region=region, algo=algo, oracleProfit=oracleProfit,
        agentProfitMean=agentMean, agentProfitSd=agentSd,
        regretMean=regretMean, regretSd=regretSd,
        capturedMean=capturedMean, capturedSd=capturedSd, seeds=seeds,
    )


def loadResultsFile(path):
    with open(path) as handle:
        return parseSections(handle.read())
