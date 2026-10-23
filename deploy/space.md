---
title: NEM BESS Dispatch
emoji: 🔋
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# NEM battery dispatch: a rolling backtest, and the forecast now

Deep reinforcement learning for battery energy storage dispatch against the Australian
National Electricity Market. Two screens behind the tabs at the top.

**Backtest.** One year of rolling-origin folds, already scored. Pick a model and a
region. The calendar shows profit per day, and a click opens that day: the realised
price, and the charge and discharge runs the agent took against it. Every day is also
solved by a perfect-foresight MILP oracle, so the capture ratio on the cards is the
share of the achievable profit the agent took.

**Current forecast.** The price path AEMO publishes at this minute, and the dispatch a
trained agent would run over it. The agent runs here and now, so this screen needs the
network. The episode starts at the run time and reaches the end of the trading day,
because PREDISPATCH stops there. The state of charge starts at 50%, because the state
of a real battery is not in this repository.

This Space holds five published agents, one per NEM region, and one results directory.
The training code, the oracle and the data layer are not here. They live in the source
repository.

The first request after a quiet period wakes the Space, and that takes about a minute.
