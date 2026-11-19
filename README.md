# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wahyunp56/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wahyunp56/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [make.fun](https://make.fun) competition
**gta6-trophy-rarity-forecast-01**: forecast, for every one of GTA VI's 55
PlayStation 5 trophies, five quantiles of its Exophase earn rate (percent)
at the target instant. Scored by mean pinball loss on the logit scale.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every
`*.in` case file in the input directory and writes one `*.out` forecast
per case. Deterministic, ~0.03 s per case.

## Method

- Collapse the read history to distinct states (Exophase refresh points):
  identical consecutive reads are one stale page, not new information.
- Work in log time, `log(1 + hours since the first read)`, and extrapolate
  to an effective target shifted earlier by half the median refresh gap
  (the answer reflects the last refresh before the target).
- Median rate per trophy: a Theil–Sen fit of the logit's rate against log
  time, slope shrunk toward 0, read at the effective target. The logit of
  a rate is close to linear in log time; the Earners/Players decomposition
  is kept for the fallback path and for the spread.
- Spread on the logit scale: residual scatter of the trophy's logit fit
  plus residual scatter of the log-players fit (both vs their raw fits,
  growing slowly with extrapolation distance) plus a 0.15 refresh
  allowance, root-sum-of-squares, floored at 0.30 and capped at 1.80.
- Flat history (one distinct state, as in the practice cases): no trend
  is estimable, so the documented launch-night refresh is the prior —
  median = last rate / 6.06 × 1.15, sigma 0.85. A trophy at 100% stays
  at 100%.

Validated on the kit practice cases (valid, deterministic) and on a
14-scenario synthetic final-like suite (mean pinball loss 0.185, vs 0.966
for the house baseline and 0.344 for a logit-trend OLS baseline — better
than both on every scenario).
