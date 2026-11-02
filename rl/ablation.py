"""The MDP ablation study: one record naming what a run changes from the baseline.

Every field defaults to the baseline problem, so `Ablation()` is the environment every
published agent trained in. A variant moves one field, and `manifestOf` records the record
only when it is not the default, so a published result set still resumes under a build
that knows about the study.

Two kinds of field, and the difference is what the oracle has to do:

- Agent-side (`lookahead`, `timeEncoding`, `history`, `hideMask`, `shapingBonus`, and the
  price transform that lives in `PriceScaler`): what the agent sees or is rewarded for.
  The problem is the same, so the oracle corpus still bounds it. Profit, revenue and
  degradation stay raw dollars; the shaping bonus enters the reward only, as the
  out-of-bounds penalty does (CLAUDE.md, "Two algorithms, one problem", rule 4).
- Problem-side: changes the dispatch problem itself, so the oracle has to be solved under
  the same change. See `degradation` and `actionLevels` below, which the oracle receives
  through `rl.baseline.cacheKey`.
"""
from typing import NamedTuple

import config

timeEncodings = ("clock", "sincos", "none")
degradations = ("cycling", "full", "flat", "none")


class Ablation(NamedTuple):
    # How many intervals ahead the agent may read. Price slots whose window ends past
    # this are zero, so the observation keeps its size and one network shape serves every
    # variant. 1 is the spot price alone. The baseline is the whole trading day.
    lookahead: int = config.observation_intervals
    # What the clock and episode slots carry. "clock" is the fraction of the trading day
    # left, repeated in the episode slot (an episode is one trading day, so they are the
    # same number). "sincos" puts the sine and cosine of the time of day there, mapped to
    # [0, 1]. "none" holds both at zero, so the agent cannot read the hour at all.
    timeEncoding: str = "clock"
    # Half-hour means of the realised prices already settled in this episode, newest
    # first, appended after the baseline layout. Zero before the episode's own start, so
    # nothing crosses a region join in the pooled training set and nothing leaks across an
    # episode boundary. 0 is no history slots.
    history: int = 0
    # How the observation maps a price: "asinh" with the knee below (the baseline),
    # "linear" (price / cap), or "clip" (clamped to config.price_clip_bounds, then asinh).
    # Observation only. The settled price the agent is paid is never transformed.
    priceTransform: str = "asinh"
    priceKnee: float = config.price_knee
    # MaskablePPO is trained and scored without the mask. `BessEnv.action_masks` then
    # reads all-legal, while the out-of-bounds count still reads the true mask.
    hideMask: bool = False
    # Reward units added when the agent charges in the cheapest quarter of the trading
    # day's prices or discharges in the dearest quarter (the buy-low/sell-high bonus of
    # Li et al. 2024). Reward only: it enters no reported dollar figure.
    shapingBonus: float = 0.0
    # The cost the reward and the oracle charge for wear. "cycling" is the Schade cycling
    # term alone, which is what every published backtest ran under (calendar=False).
    # "full" adds the calendar term, "flat" charges `flatCost` $ per MWh removed cell-side
    # (the Schade price of a full discharge per MWh when `flatCost` is zero, so only the
    # shape differs), and "none" charges nothing. Problem-side: the oracle is solved under
    # the same cost, and the backtest turns the calendar term on for "full".
    degradation: str = "cycling"
    flatCost: float = 0.0
    # Power levels per direction. 1 is the three-action baseline (charge, idle,
    # discharge). 2 adds a half-rate step each way, and so on. Problem-side.
    actionLevels: int = 1

    def isDefault(self):
        return self == Ablation()

    def validate(self):
        if self.lookahead < 1 or self.lookahead > config.observation_intervals:
            raise ValueError(f"lookahead {self.lookahead} is outside "
                             f"1..{config.observation_intervals}")
        if self.timeEncoding not in timeEncodings:
            raise ValueError(f"timeEncoding {self.timeEncoding!r} is not one of "
                             f"{timeEncodings}")
        if self.priceTransform not in ("asinh", "linear", "clip"):
            raise ValueError(f"unknown priceTransform {self.priceTransform!r}")
        if self.history < 0:
            raise ValueError("history cannot be negative")
        if self.shapingBonus < 0:
            raise ValueError("shapingBonus cannot be negative")
        if self.degradation not in degradations:
            raise ValueError(f"unknown degradation {self.degradation!r}")
        if self.actionLevels < 1:
            raise ValueError("actionLevels must be at least 1")
        return self

    def asManifest(self):
        """Only the fields that differ from the baseline, so a named variant reads as
        what it changes. Empty for the baseline."""
        base = Ablation()
        return {name: value for name, value in self._asdict().items()
                if value != getattr(base, name)}

    @classmethod
    def fromManifest(cls, entry):
        return cls(**(entry or {})).validate()

    def costing(self, model, dtHours):
        """What the oracle and the environment charge in place of the Schade terms, as
        (mode, $ per MWh removed cell-side), or None when the Schade terms stand."""
        if self.degradation in ("cycling", "full"):
            return None
        if self.degradation == "none":
            return ("none", 0.0)
        if self.flatCost:
            return ("flat", self.flatCost)
        qCd, _, _ = model.step(1.0, 0.0, dtHours)
        return ("flat", model.cost(qCd) / model.MWh)

    def scaler(self, cap):
        """The price scaler for a market cap, under this ablation's transform."""
        from rl.features import PriceScaler
        bounds = {}
        if self.priceTransform == "clip":
            bounds = {"clipLow": config.price_clip_bounds[0],
                      "clipHigh": config.price_clip_bounds[1]}
        return PriceScaler(cap, knee=self.priceKnee, transform=self.priceTransform,
                           **bounds)


def addAblationArguments(parser):
    """The flags that name an ablation, shared by every command that trains an agent. A
    run that passes none is the baseline problem."""
    group = parser.add_argument_group("MDP ablation (rl/ablation.py)")
    group.add_argument("--lookahead", type=int, default=config.observation_intervals,
                       metavar="N", help="intervals of price the agent may read")
    group.add_argument("--time-encoding", choices=timeEncodings, default="clock")
    group.add_argument("--history", type=int, default=0, metavar="N",
                       help="half-hour means of this episode's settled prices to append")
    group.add_argument("--price-transform", choices=("asinh", "linear", "clip"),
                       default="asinh")
    group.add_argument("--price-knee", type=float, default=config.price_knee)
    group.add_argument("--hide-mask", action="store_true",
                       help="train and score MaskablePPO without the action mask")
    group.add_argument("--shaping-bonus", type=float, default=0.0, metavar="R",
                       help="reward for charging cheap or discharging dear")
    group.add_argument("--degradation", choices=degradations, default="cycling")
    group.add_argument("--flat-cost", type=float, default=0.0, metavar="$/MWh")
    group.add_argument("--action-levels", type=int, default=1, metavar="N")


def ablationOf(args):
    return Ablation(
        lookahead=args.lookahead, timeEncoding=args.time_encoding, history=args.history,
        priceTransform=args.price_transform, priceKnee=args.price_knee,
        hideMask=args.hide_mask, shapingBonus=args.shaping_bonus,
        degradation=args.degradation, flatCost=args.flat_cost,
        actionLevels=args.action_levels).validate()
