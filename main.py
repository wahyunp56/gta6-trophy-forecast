"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PS5 trophies, five quantiles of its
Exophase earn rate (percent) at the case target.

Method (fitted fresh on each case file, nothing hard-coded from practice):
  * Collapse the read history to distinct states (refresh points): a new
    state starts when players-tracked or any rate changes. Identical
    consecutive reads are one stale page, not new information.
  * Work in log time, log(1 + hours since the first read).  The answer
    reflects the last refresh before the target, not the target instant
    itself, so extrapolate to an effective target shifted earlier by half
    the median gap between distinct states (capped at 12 hours).
  * Median rate: per trophy, a Theil-Sen fit of the logit's rate against
    log time over the distinct states, slope shrunk toward 0 with weight
    m/(m+1), read at the effective target.  The logit of a rate is close
    to linear in log time (a trophy's earners are the convolution of
    player arrivals with a time-to-earn distribution, which is roughly
    logistic in log time), while the log of earners itself is concave in
    log time -- a straight line fitted to log earners overshoots badly
    at a 4-5 day horizon (median 99.99% for mid trophies on the synthetic
    suite, loss 4.68 vs 0.19 for this fit).  The Earners/Players
    decomposition is kept for the fallback path and for the spread.
  * Spread on the logit scale combines the residual scatter of the
    trophy's logit fit, the residual scatter of the log-players fit
    (both measured against their raw, unshrunk fits and growing slowly
    with extrapolation distance), and a 0.15 refresh allowance,
    root-sum-of-squares, floored at 0.30 and capped at 1.80.
  * Flat history (one distinct state): no trend is estimable, so use the
    documented launch-night refresh as the prior -- players x6.06 with
    only modest earner growth -- median = last rate / 6.06 * 1.15,
    sigma 0.85.  A trophy at 100% stays at 100%; a trophy with zero
    earners in every read gets half the smallest positive last-read rate
    with a wide spread.

Python 3.13, standard library only. Deterministic, no network.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Flat-history prior: first refresh multiplied players 638 -> 3868.
FLAT_PLAYER_MULT = 6.06
FLAT_EARNER_GROWTH = 1.15
FLAT_SIGMA = 0.85
# Spread: refresh allowance, floor/cap, shrinkage, extrapolation growth.
REFRESH_ALLOWANCE = 0.15
SIGMA_FLOOR = 0.30
SIGMA_CAP = 1.80
SHRINK_K_PLAYERS = 3.0
SHRINK_K_LOGIT = 1.0
SHRINK_K_EARNERS = 3.0
EXTRAP_SCALE = 0.25


def parse_time(text):
    """ISO 8601 timestamp (with Z) -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the kit starter's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def theil_sen(xs, ys):
    """Return (slope, intercept) by Theil-Sen; intercept = median(y - slope*x)."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx != 0.0:
                slopes.append((ys[j] - ys[i]) / dx)
    slope = median(slopes) if slopes else 0.0
    intercept = median([y - slope * x for x, y in zip(xs, ys)])
    return slope, intercept


def rms_residual(xs, ys, slope, intercept):
    if len(xs) < 2:
        return 0.0
    sq = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(sq / len(xs))


def distinct_states(case):
    """Collapse history to refresh points: (first-seen time, players, rates)."""
    states = []
    prev_players = None
    prev_rates = None
    for read in case.get("history", []):
        players = read.get("players_tracked")
        rates = read.get("rates")
        if players is None or rates is None:
            continue
        if prev_players is not None and players == prev_players and rates == prev_rates:
            continue
        states.append((parse_time(read["at"]), players, list(rates)))
        prev_players = players
        prev_rates = list(rates)
    return states


def forecast_case(case):
    trophies = case["trophies"]
    n_troph = len(trophies)
    levels = case.get("levels") or LEVELS
    zvals = [Z[LEVELS.index(lv)] if lv in LEVELS else 0.0 for lv in levels]
    target = parse_time(case["target"])
    states = distinct_states(case)

    # --- Defensive fallback: no usable history at all. ---
    if not states:
        return {"trophies": [
            {"id": t["id"], "q": [round4(logistic(logit(1.0) + z * 1.5)) for z in zvals]}
            for t in trophies
        ]}

    last_time, last_players, last_rates = states[-1]
    first_time = states[0][0]

    def logt(when):
        hours = (when - first_time).total_seconds() / 3600.0
        return math.log(1.0 + max(0.0, hours))

    # --- Refresh lag: median gap between distinct states. ---
    if len(states) > 1:
        gaps = [
            (states[i + 1][0] - states[i][0]).total_seconds() / 3600.0
            for i in range(len(states) - 1)
        ]
        med_gap = median(gaps)
    else:
        med_gap = 5.0
    shift_hours = min(0.5 * med_gap, 12.0)
    eff_target = target - timedelta(hours=shift_hours)
    if eff_target < last_time:
        eff_target = last_time
    lt_eff = logt(eff_target)
    lt_last = logt(last_time)
    extrap = max(0.0, lt_eff - lt_last)

    last_rate = [r if r is not None else 0.0 for r in last_rates]
    positive_last = [r for r in last_rate if r > 0.0]
    smallest_positive = min(positive_last) if positive_last else 1.0

    rows = []

    # --- Flat-history regime: exactly one distinct state. ---
    if len(states) == 1:
        for i, trophy in enumerate(trophies):
            rate = last_rate[i]
            if rate >= 100.0:
                rows.append({"id": trophy["id"], "q": [100.0] * len(levels)})
                continue
            med = clamp_pct(rate / FLAT_PLAYER_MULT * FLAT_EARNER_GROWTH)
            qs = [round4(logistic(logit(med) + z * FLAT_SIGMA)) for z in zvals]
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}

    # --- Rich history. ---
    xs = [logt(t) for t, _, _ in states]
    n_states = len(states)

    # Players: Theil-Sen on log players vs log time, slope shrunk toward 0.
    # The shrunk fit forecasts players; the raw fit measures scatter.
    log_players = [math.log(max(1, p)) for _, p, _ in states]
    p_slope_raw, p_inter_raw = theil_sen(xs, log_players)
    p_slope = p_slope_raw * (n_states / (n_states + SHRINK_K_PLAYERS))
    p_inter = median([y - p_slope * x for x, y in zip(xs, log_players)])
    p_resid = rms_residual(xs, log_players, p_slope_raw, p_inter_raw)
    players_fc = max(float(last_players), math.exp(p_inter + p_slope * lt_eff))

    # Fallback data for trophies whose logit fit has too few points:
    # global earner slope (median over trophies) and global residual scale.
    earner_slopes = []
    earner_fits = []
    for i in range(n_troph):
        tx, ty = [], []
        for k, (_, players, rates) in enumerate(states):
            r = rates[i]
            if r is None:
                continue
            earners = int(round(r / 100.0 * players))
            if earners >= 1:
                tx.append(xs[k])
                ty.append(math.log(earners))
        if len(tx) >= 2:
            s_raw, _ = theil_sen(tx, ty)
            earner_slopes.append(s_raw)
            earner_fits.append((s_raw, tx, ty))
        else:
            earner_fits.append((None, tx, ty))
    global_e_slope = median(earner_slopes) if earner_slopes else 0.0
    _res = []
    for s_raw, tx, ty in earner_fits:
        if s_raw is not None and len(tx) >= 2:
            _a = median([y - s_raw * x for x, y in zip(tx, ty)])
            _res.append(rms_residual(tx, ty, s_raw, _a))
    global_resid = median(_res) if _res else 0.15

    scale = 1.0 + EXTRAP_SCALE * extrap

    for i, trophy in enumerate(trophies):
        rate = last_rate[i]
        if rate >= 100.0:
            rows.append({"id": trophy["id"], "q": [100.0] * len(levels)})
            continue

        # Primary path: Theil-Sen on logit(rate) vs log time.
        rx, ry = [], []
        for k, (_, _, rates) in enumerate(states):
            r = rates[i]
            if r is not None and 0.0 < r < 100.0:
                rx.append(xs[k])
                ry.append(logit(r))
        if len(rx) >= 2:
            s_raw, a_raw = theil_sen(rx, ry)
            m = len(rx)
            slope = s_raw * (m / (m + SHRINK_K_LOGIT))
            inter = median([y - slope * x for x, y in zip(rx, ry)])
            med = clamp_pct(logistic(inter + slope * lt_eff))
            sig_e = rms_residual(rx, ry, s_raw, a_raw) * scale
            sig_p = p_resid * scale
            sigma = math.sqrt(sig_e * sig_e + sig_p * sig_p + REFRESH_ALLOWANCE ** 2)
            sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
            qs = [round4(logistic(logit(med) + z * sigma)) for z in zvals]
            rows.append({"id": trophy["id"], "q": qs})
            continue

        # Fallback path: Earners / Players decomposition.
        s_raw, tx, ty = earner_fits[i]
        if s_raw is None and not ty:
            # Zero earners in every read: wide-spread low median.
            med = clamp_pct(0.5 * smallest_positive)
            qs = [round4(logistic(logit(med) + z * 1.5)) for z in zvals]
            rows.append({"id": trophy["id"], "q": qs})
            continue
        if s_raw is None:
            slope = global_e_slope
            inter = ty[-1] - slope * tx[-1]
            resid = global_resid
        else:
            m = len(tx)
            slope = global_e_slope + (s_raw - global_e_slope) * (m / (m + SHRINK_K_EARNERS))
            inter = median([y - slope * x for x, y in zip(tx, ty)])
            resid = rms_residual(tx, ty, slope, inter)
            if len(tx) < 3:
                resid = max(resid, global_resid)
        last_earners = int(round(rate / 100.0 * last_players))
        earn_fc = math.exp(inter + slope * lt_eff)
        earn_fc = max(float(last_earners), min(earn_fc, players_fc))
        med = clamp_pct(100.0 * earn_fc / players_fc)
        sig_e = resid * scale
        sig_p = p_resid * scale
        sigma = math.sqrt(sig_e * sig_e + sig_p * sig_p + REFRESH_ALLOWANCE ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        qs = [round4(logistic(logit(med) + z * sigma)) for z in zvals]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
