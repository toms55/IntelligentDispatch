# Methodology

## Overview

One diagram for the whole method. Realised prices make the problem. The oracle and the
agent solve the same problem. The difference between them is the result.

```mermaid
flowchart TB
    AEMO[AEMO dispatch prices via nemosis]
    CLEAN[Clean 5-min RRP, one region]
    FOLDS[Rolling folds, train then test, 04:00 boundaries]

    DP[Discrete oracle: optimisation/dp.py]
    ENV[BessEnv: 3 actions, Schade degradation]
    AGENT[DQN and MaskablePPO, equal step budget, 5 seeds]

    SCORE[Score each test day twice: realised prices, then AEMO forecast]
    OUT[Profit, regret, captured fraction, out-of-bounds rate]

    AEMO --> CLEAN --> FOLDS
    FOLDS --> DP
    FOLDS --> ENV --> AGENT
    DP --> SCORE
    AGENT --> SCORE
    SCORE --> OUT
```

The oracle and the environment are the same problem stated twice. Same three actions,
same top-up rule at the SoC boundary, same episode, same degradation. That is what makes
the oracle a bound.

## Detail

Three more diagrams. The first shows the whole pipeline. The second shows one
environment step. The third shows how the agent is scored.

### Pipeline

Data enters at the top. Every window is solved twice: once by the oracle, once by the
agent.

```mermaid
flowchart TB
    AEMO[AEMO NEM data]
    NEMOSIS[nemosis]
    NEMSEER[nemseer]
    REAL[Realised RRP]
    FCST[Forecast RRP]
    CLEAN[Clean 5-min series]
    SPLIT[Split at 04:00]
    TRAIN[Train 2024]
    VAL[Validate Nov-Dec 2024]
    TEST[Test 2025]
    ORACLE[MILP oracle]
    BASE[Reward baseline]
    BOUND[Oracle bound]
    ENV[BessEnv]
    AGENT[MaskablePPO, 5 seeds]
    CKPT[Checkpoint + scaler.json]
    EVAL[Evaluation]
    OUT[Regret and profit]

    AEMO --> NEMOSIS
    AEMO --> NEMSEER
    NEMOSIS --> REAL
    NEMSEER --> FCST
    REAL --> CLEAN
    CLEAN --> SPLIT
    SPLIT --> TRAIN
    SPLIT --> VAL
    SPLIT --> TEST

    TRAIN --> ORACLE
    VAL --> ORACLE
    TEST --> ORACLE
    ORACLE --> BASE
    ORACLE --> BOUND

    TRAIN --> ENV
    BASE --> ENV
    ENV --> AGENT
    AGENT --> CKPT

    CKPT --> EVAL
    BOUND --> EVAL
    FCST --> EVAL
    VAL --> EVAL
    TEST --> EVAL
    EVAL --> OUT
```

The oracle runs in two modes. The continuous mode is cheap and gives the per-interval
reward baseline. The discrete mode uses the same three actions as the agent, so it is
the only fair bound. See `config.py`.

### One environment step

The environment and `solve(discretePower=True)` state the same problem. Same actions,
same top-up rule, same degradation terms.

```mermaid
flowchart TB
    OBS[Observation, 60 slots]
    SCALE[PriceScaler]
    POLICY[Policy]
    MASK[Action mask]
    ACT[Charge, hold or discharge]
    CLIP[Clip at SoC limit]
    SOC[Update SoC]
    DEG[Schade degradation]
    REV[Trade revenue]
    PROF[Profit]
    SUB[Subtract baseline]
    REW[Reward]

    SCALE --> OBS
    OBS --> POLICY
    MASK --> POLICY
    POLICY --> ACT
    ACT --> CLIP
    CLIP --> SOC
    SOC --> DEG
    SOC --> REV
    REV --> PROF
    DEG --> PROF
    PROF --> SUB
    SUB --> REW
    REW --> OBS
```

The observation holds the SoC, 12 five-minute prices, 46 half-hour bucket means and the
intervals remaining. `rl/features.py` is the single definition of this layout.

### Scoring

Each policy is scored twice on the same window. The prices it sees change. The prices
it is paid on do not.

```mermaid
flowchart TB
    POL[Trained policy]
    SEEN1[Sees realised prices]
    SEEN2[Sees AEMO forecast]
    PAY[Paid on realised prices]
    P1[Profit, realised]
    P2[Profit, forecast]
    OB[Oracle dual bound]
    REG[Regret]
    PLOT[results and visualise]

    POL --> SEEN1
    POL --> SEEN2
    SEEN1 --> PAY
    SEEN2 --> PAY
    PAY --> P1
    PAY --> P2
    P1 --> REG
    P2 --> REG
    OB --> REG
    REG --> PLOT
```

The gap between the two profits is the cost of forecast error. Quote
`Schedule.dualBound` as the bound, never `Schedule.profit`.
