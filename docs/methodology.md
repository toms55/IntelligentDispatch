# Methodology

## Overview

The method in one figure. Each stage is described in full below.

```mermaid
flowchart TB
    D[Collect historical market prices]
    P[Clean the price series and align it to trading days]
    S[Split into rolling train and test periods]
    M[Define the battery dispatch problem: three actions, one trading day, degradation cost]

    B[Build the benchmark: perfect-foresight optimum under the same constraints]
    T[Train two agents: DQN and PPO]

    E[Evaluate on held-out days, twice: with realised prices, then with the forecast available at the time]
    C[Compare each agent against the benchmark]
    R[Report profit, shortfall against the benchmark, and infeasible-action rate]

    D --> P --> S --> M
    M --> B
    M --> T
    B --> E
    T --> E
    E --> C --> R
```

The benchmark and the agents face the same problem. Same actions, same horizon, same
costs. This is what makes the benchmark an upper bound, so the gap to it measures the
policy and nothing else.

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
