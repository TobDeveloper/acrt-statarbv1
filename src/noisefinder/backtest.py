import warnings

import numpy as np
import pandas as pd

from noisefinder import config, periods
from noisefinder.costs import Costs
from noisefinder.io import load_table

warnings.simplefilter("ignore")
TOP = 20
H = pd.Timedelta(hours=1)
MAKER, BUF = 0.0002, 0.0002
JUMP_Z, VOL_X, NEWS_H, END_H = 4.0, 5.0, 3, 72
TIME_STOP_HALF_LIVES = 3
MIN_EXPECTED_WIN = 0.0175
PER_HOUR = 1
P = config.PROCESSED
TAB = dict(
    close=load_table(P / "log_close_1h"),
    high=load_table(P / "log_high_1h"),
    low=load_table(P / "log_low_1h"),
    vol=load_table(P / "quote_volume_1h"),
    funding=load_table(P / "funding_1h"),
)
costs = Costs()
_PREP = {}


def kalman(y, X, warm, memory):

    n, k = X.shape
    xbar = X[:warm].mean(0)
    Xc = X - xbar
    A = np.column_stack([np.ones(warm), Xc[:warm]])
    coef, *_ = np.linalg.lstsq(A, y[:warm], rcond=None)
    r = float(((y[:warm] - A @ coef) ** 2).sum() / (warm - k - 1))
    P = r * np.linalg.inv(A.T @ A)
    Q = np.diag(np.r_[r / memory**2, r / memory**2 / Xc[:warm].var(0)])
    st = coef.copy()
    beta, spread = np.full((n, k), np.nan), np.full(n, np.nan)
    for t in range(warm, n):
        if np.isnan(y[t]) or np.isnan(Xc[t]).any():
            beta[t] = st[1:]
            continue
        P = P + Q
        h = np.r_[1.0, Xc[t]]
        e = y[t] - h @ st
        beta[t], spread[t] = st[1:], e
        S = h @ P @ h + r
        K = P @ h / S
        st = st + K * e
        P = P - np.outer(K, h @ P)
    return beta, spread


def prepare(r):
    start, end = periods.bounds(r.period)
    ph = PER_HOUR
    warm, band = config.KF_WARMUP_DAYS * 24 * ph, config.KF_BAND_DAYS * 24 * ph
    step = pd.Timedelta(minutes=60 // ph)
    hours = pd.date_range(start - step * (warm + band), end - step, freq=step)
    coins = r.coins.split(";")
    w = TAB["close"].reindex(index=hours, columns=coins).ffill()
    if w.iloc[0].isna().any():
        return None
    y = w[coins[0]].to_numpy()
    X = w[coins[1:]].mean(axis=1).to_numpy()[:, None] if r.kind == "sector" else w[coins[1:]].to_numpy()
    beta, spread = kalman(y, X, warm, config.KF_MEMORY_DAYS * 24 * ph)
    L = w.to_numpy()
    Hh = TAB["high"].reindex(index=hours, columns=coins).to_numpy()
    Ll = TAB["low"].reindex(index=hours, columns=coins).to_numpy()
    Hh, Ll = np.where(np.isnan(Hh), L, Hh), np.where(np.isnan(Ll), L, Ll)
    R = np.diff(L, axis=0, prepend=np.nan)
    V = TAB["vol"].reindex(index=hours, columns=coins).to_numpy()
    usual_move = pd.DataFrame(R).shift(1).rolling(240 * ph, min_periods=120 * ph).std().to_numpy()
    usual_vol = pd.DataFrame(V).shift(1).rolling(240 * ph, min_periods=120 * ph).median().to_numpy()
    recent = lambda f: pd.Series(f.astype(float)).rolling(NEWS_H * ph, min_periods=1).max().to_numpy() > 0
    month = periods.month(r.period)
    F = (
        TAB["funding"].reindex(index=hours, columns=coins).fillna(0.0).to_numpy()
        if "funding" in TAB
        else np.zeros_like(L)
    )
    return dict(
        period=r.period,
        kind=r.kind,
        coins=coins,
        beta=beta,
        spread=spread,
        first=warm + band,
        n=len(hours),
        F=F,
        hours=hours,
        L=L,
        Hh=Hh,
        Ll=Ll,
        taker=np.array([costs.per_side(month, c) for c in coins]),
        jump=recent((np.abs(R) > JUMP_Z * usual_move).any(axis=1)),
        volume=recent((V > VOL_X * usual_vol).any(axis=1)),
    )


def units(p, b):

    if p["kind"] == "sector":
        m = len(p["coins"]) - 1
        return np.r_[1.0, np.full(m, -b[0] / m)]
    return np.r_[1.0, -b]


def strategy(spread, first, window, entry, stop, block, max_hold=None):

    s = pd.Series(spread)
    ma, sd = s.shift(1).rolling(window).mean().to_numpy(), s.shift(1).rolling(window).std().to_numpy()
    sig, event = np.zeros(len(s)), [""] * len(s)
    pos, armed, t_in = 0, True, None
    for t in range(first, len(s)):
        v, m, d = spread[t], ma[t], sd[t]
        if t == len(s) - 1:
            if pos != 0:
                event[t] = "period end"
            pos = 0
        elif np.isnan(v) or np.isnan(m) or np.isnan(d):
            pass
        else:
            ue, le, us, ls = m + entry * d, m - entry * d, m + stop * d, m - stop * d
            if pos == 0:
                if not armed:
                    if le <= v <= ue:
                        armed = True
                elif v > us or v < ls:
                    armed = False
                elif block[t]:
                    pass
                elif v > ue:
                    pos, event[t], t_in = -1, "enter short", t
                elif v < le:
                    pos, event[t], t_in = 1, "enter long", t
            elif max_hold is not None and t - t_in >= max_hold:
                pos, armed, event[t] = 0, False, "time stop"
            elif pos == -1:
                if v > us:
                    pos, armed, event[t] = 0, False, "stop"
                elif v <= m:
                    pos, event[t] = 0, "exit at MA"
            elif v < ls:
                pos, armed, event[t] = 0, False, "stop"
            elif v >= m:
                pos, event[t] = 0, "exit at MA"
        sig[t] = pos
    return sig, event, sd


def spread_list(group="real", final=False, top=TOP):
    ve = pd.read_csv(config.RUN_DIR / "vecm.csv")
    ve = ve[ve.tradeable.astype(bool) & (ve.group == group) & ve.period.map(periods.checkable)]
    if not final:
        ve = ve[ve.period.str[:7] < config.BT_HOLDOUT_FROM]
    return ve.sort_values(["period", "half_life_h"]).groupby("period").head(top)


def prepared(r):
    k = (r.period, r.kind, r.coins)
    if k not in _PREP:
        _PREP[k] = prepare(r)
    return _PREP[k]


def trades_of(p, maker=True, end_rule=True, max_hold=None, entry=None, stop=None, block_extra=None):

    n, last = p["n"], p["n"] - 1
    block = np.zeros(n, bool)
    if end_rule:
        block[n - END_H :] = True
    if block_extra is not None:
        block |= block_extra
    sig, ev, sd = strategy(
        p["spread"],
        p["first"],
        config.KF_BAND_DAYS * 24,
        entry or config.KF_ENTRY,
        stop or config.KF_STOP,
        block,
        max_hold,
    )
    R = np.diff(p["L"], axis=0, prepend=np.nan)
    out, t0 = [], None
    for t in range(p["first"], n):
        if ev[t].startswith("enter"):
            t0 = t
        elif ev[t] and not ev[t].startswith("enter") and t0 is not None:
            how, side = ev[t], sig[t0]
            a0, a1 = t0, (last if how == "period end" else t)
            t0_ = t0
            t0 = None
            if a0 >= last:
                continue
            u = side * units(p, p["beta"][t0_])
            lo_, hi_ = a0, min(a1 + 1, last)
            span = hi_ - lo_ + 1
            g, c, f, e = (np.zeros(span) for _ in range(4))
            per_coin, pos_coin = {}, {}
            for j, coin in enumerate(p["coins"]):
                buy = u[j] > 0
                s0, c0, e1, c1 = a0, p["taker"][j], a1, p["taker"][j]
                if maker:
                    lim = p["L"][a0, j]
                    if (p["Ll"][a0 + 1, j] < lim - BUF) if buy else (p["Hh"][a0 + 1, j] > lim + BUF):
                        c0 = MAKER
                    else:
                        s0 = a0 + 1
                    if how == "exit at MA" and a1 < last:
                        lim = p["L"][a1, j]
                        if (p["Hh"][a1 + 1, j] > lim + BUF) if buy else (p["Ll"][a1 + 1, j] < lim - BUF):
                            c1 = MAKER
                        else:
                            e1 = a1 + 1
                e1 = max(e1, s0)
                leg = np.zeros(span)
                leg[s0 + 1 - lo_ : e1 + 1 - lo_] = u[j] * R[s0 + 1 : e1 + 1, j]
                g += leg
                c[s0 - lo_] += abs(u[j]) * c0
                c[e1 - lo_] += abs(u[j]) * c1
                f[s0 + 1 - lo_ : e1 + 1 - lo_] += u[j] * p["F"][s0 + 1 : e1 + 1, j]
                e[s0 + 1 - lo_ : e1 + 1 - lo_] += abs(u[j])
                per_coin[coin] = per_coin.get(coin, 0) + leg
                held = np.zeros(span)
                held[s0 + 1 - lo_ : e1 + 1 - lo_] = u[j]
                pos_coin[coin] = pos_coin.get(coin, 0) + held
            out.append(
                dict(
                    period=p["period"],
                    kind=p["kind"],
                    coins=";".join(p["coins"]),
                    side=int(side),
                    how=how,
                    t_open=p["hours"][a0],
                    t_close=p["hours"][a1],
                    hours=a1 - a0,
                    std=float(sd[a0]),
                    start=p["hours"][lo_],
                    gross=g,
                    cost=c,
                    funding=f,
                    exposure=e,
                    per_coin=per_coin,
                    pos_coin=pos_coin,
                )
            )
    return out


def hour_index(final=False):
    first = pd.Timestamp(config.FIRST_UNIVERSE_MONTH + "-01")
    end = pd.Timestamp(config.DATA_END) if final else pd.Timestamp(config.BT_HOLDOUT_FROM + "-01")
    return pd.date_range(first, end - H, freq="h")


def book(trades, weights=None, idx=None, per_spread=1 / TOP):

    idx = idx if idx is not None else hour_index()
    pos = {t: i for i, t in enumerate(idx)}
    cols = {k: np.zeros(len(idx)) for k in ("gross", "cost", "funding", "exposure")}
    for i, tr in enumerate(trades):
        w = per_spread * (1.0 if weights is None else weights[i])
        if w == 0:
            continue
        s = pos.get(tr["start"])
        if s is None:
            continue
        m = min(len(tr["gross"]), len(idx) - s)
        for k in cols:
            cols[k][s : s + m] += w * tr[k][:m]
    b = pd.DataFrame(cols, index=idx)
    b["net"] = b.gross - b.cost - b.funding
    return b


def stats(net_daily, label=""):
    d = net_daily
    m = d.resample("MS").sum()
    eq = d.cumsum()
    return dict(
        label=label,
        total=d.sum(),
        ret=d.mean() * 365,
        vol=d.std() * np.sqrt(365),
        sharpe_d=d.mean() / d.std() * np.sqrt(365),
        sharpe_m=m.mean() / m.std() * np.sqrt(12),
        maxdd=(eq - eq.cummax()).min(),
        months_pos=(m > 0).mean(),
    )


def expected_win_ok(trade):
    gross_units = sum(float(np.abs(a).max()) for a in trade["pos_coin"].values())
    return 3 * trade["std"] / gross_units - 2 * MAKER >= MIN_EXPECTED_WIN


def run(group="real", final=True):
    trades = []
    for r in spread_list(group, final=final).itertuples(index=False):
        p = prepared(r)
        if p is not None:
            trades += trades_of(p, max_hold=int(round(TIME_STOP_HALF_LIVES * r.half_life_h)))
    weights = np.array([1.0 if expected_win_ok(t) else 0.0 for t in trades])
    daily = (
        book(trades, weights, idx=hour_index(final))[["gross", "cost", "funding", "net"]].resample("D").sum()
    )
    return [t for t, w in zip(trades, weights) if w > 0], daily
