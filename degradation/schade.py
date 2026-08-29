"""
code modelled off paper:
    C. Schade and R. Egging-Bratseth, "Battery degradation: Impact on economic
    dispatch," Energy Storage, vol. 6, no. 2, e588, 2024. doi:10.1002/est2.588
    Open access. Data: https://zenodo.org/doi/10.5281/zenodo.10255080

Every item below names the paper section and equation to implement from.
Section numbers are the paper's own (§3.2 = "Cycle degradation", etc).

The paper is an hourly MILP economic-dispatch model. I need two things out of
it: (A) an LP-embeddable degradation cost for the perfect-foresight oracle
(TODO.txt Step 1), and (B) a causal per-step cost usable as an RL reward
(Step 2).
"""
import numpy as np
from utils import nem_interval_hours

class SchadeDegradation:
    def __init__(self, MW, MWh, replacementCost):
        self.chargePower = MW
        self.dischargePower = MW
        self.MWh = MWh
        self.replacementCost = replacementCost  # cost to buy back lost capacity, per MWh
        self.maxCapacityLoss = 0.0004519  # fraction of capacity lost by one full 100% cycle
        self.SoC_stress = 0.000085  # extra loss for cycling away from 50% SOC, penalised equally high or low
        self.fatigueStrengthExponent = 0.4926 # how much worse deep, full drains are for the battery's physical structure compared to small, shallow uses.
        self.chargeEfficiency = 0.95
        self.dischargeEfficiency = 0.95
        self.SoC_breakpointVal = [j/16 for j in range(17)]  # where to cut the cycle-depth curve into 16 straight segments
        self.calendarAgingBreakpointVal = [0.00, 0.30, 0.60, 0.70, 1.00]  # SOC levels where the measured calendar curve changes shape
        self.y = [3.75e-7, 8.76e-7, 10.01e-7, 18.41e-7, 22.34e-7]  # capacity lost per hour just sitting at each of those SOC levels
        self.segmentCosts = self._segmentCosts()
        self.segmentCapacities = self._segmentCapacities()
        self.segmentMarginalCosts = [
            cost / cap for cost, cap in zip(self.segmentCosts, self.segmentCapacities)
        ]

    def cycleDegradation(self, CD: float) -> float:
        "Eq. 1 capacity lost by one cycle of depth CD."
        return self.maxCapacityLoss * CD ** (1 / self.fatigueStrengthExponent)

    def _segmentCosts(self) -> list[float]:
        "Eq. 2 cost of fully discharging each segment, lowest SOC first."
        scale = self.MWh * self.replacementCost
        edges = self.SoC_breakpointVal
        return [
            scale * (self.cycleDegradation(hi) - self.cycleDegradation(lo))
            for lo, hi in zip(edges, edges[1:])
        ]

    def _segmentCapacities(self) -> list[float]:
        "Eq. 5 energy held by each segment, i.e. the LP's per-segment cap."
        edges = self.SoC_breakpointVal
        return [self.MWh * (hi - lo) for lo, hi in zip(edges, edges[1:])]

    def lpObjectiveCoefficients(self) -> list[float]:
        "Eq. 3 objective row: $ per MWh SOLD out of each segment."
        # self.segmentMarginalCosts prices cell-side energy (see totalCycleDegradation),
        # but the LP's D_{j,h} is meter-side, and cell-side is 1/nu_D larger.
        return [c / self.dischargeEfficiency for c in self.segmentMarginalCosts]

    def totalCycleDegradation(self, socPrev: float, socNew: float) -> float:
        "Eq. 11 capacity lost discharging from socPrev down to socNew."
        if socNew >= socPrev:
            return 0.0
        total = 0.0
        edges = self.SoC_breakpointVal
        for lo, hi in zip(edges, edges[1:]):
            overlap = max(0.0, min(socPrev, hi) - max(socNew, lo))
            total += (overlap / (hi - lo)) * (self.cycleDegradation(hi) - self.cycleDegradation(lo))
        # cell side degradation, if you wanted meter side multiply by self.dischargeEfficiency 
        return total

    def calendarAging(self, SoC: float) -> float:
        return np.interp(SoC, self.calendarAgingBreakpointVal, self.y) * nem_interval_hours 
    
    def cost(self, Q: float) -> float:
        "Eq. 10 convert capacity loss fraction to $"
        return self.replacementCost * self.MWh * Q

    def step(self, socPrev, socNew, dt_hours):
        q_cd  = self.totalCycleDegradation(socPrev, socNew)
        q_cal = self.calendarAging(socNew)
        return q_cd, q_cal, self.cost(q_cd + q_cal)

