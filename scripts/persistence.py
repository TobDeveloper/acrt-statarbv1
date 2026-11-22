import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from noisefinder import config, periods
from noisefinder.costs import Costs, legs
from noisefinder.io import load_table

warnings.simplefilter("ignore", FutureWarning)


def yardstick(z):

    trades, pos, j0 = [], 0, None
    for i in range(len(z) - 1):
        if pos == 0 and abs(z[i]) >= config.YARDSTICK_ENTRY_Z:
            pos, j0 = -np.sign(z[i]), i + 1
        elif pos != 0 and pos * z[i] >= 0:
            trades.append((pos, j0, i + 1))
            pos = 0
    if pos != 0:
        trades.append((pos, j0, len(z) - 1))
    return [t for t in trades if t[2] > t[1]]


def pull(x, s):

    dx, ds, L = np.diff(x, axis=0), np.diff(s), config.JOHANSEN_LAGS
    y = ds[L:]
    X = np.column_stack([np.ones(len(y)), s[L:-1]] + [dx[L - j : len(dx) - j] for j in range(1, L + 1)])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def check(r, x, fund_cum, costs):
    coins, wt = r.coins.split(";"), np.array([float(v) for v in r.weights.split(";")])
    s = x.to_numpy() @ wt
    z = (s - r.spread_mean) / r.swing
    b = pull(x.to_numpy(), s)
    hl = np.log(2) / -np.log1p(b) if -1 < b < 0 else np.inf
    adf_p = adfuller(s, regression="c", autolag="AIC")[1]
    held = legs(r.kind, coins, wt)
    exposure = sum(abs(v) for v in held.values())
    rt = costs.round_trip(periods.month(r.period), r.kind, coins, wt)
    hours = x.index
    bounce = cost = fund = 0.0
    trades = yardstick(z)
    for pos, j0, j1 in trades:
        bounce += pos * (s[j1] - s[j0])
        cost += rt
        paid = fund_cum.loc[hours[j1]] - fund_cum.loc[hours[j0]]
        fund += pos * sum(v * paid.get(c, 0.0) for c, v in held.items())
    net = bounce - cost - fund
    hl_ok = bool(config.MIN_HALF_LIFE_H <= hl <= config.MAX_HALF_LIFE_H)
    return dict(
        n_hours_next=len(x),
        adf_p_next=adf_p,
        half_life_next=hl,
        drift_next=float(np.mean(z)),
        half_life_ok=hl_ok,
        still_leashed=bool(hl_ok and adf_p < config.SIGNIFICANCE),
        trades=len(trades),
        bounce=bounce / exposure,
        costs=cost / exposure,
        funding=fund / exposure,
        net=net / exposure,
        made_money=bool(net > 0),
    )


def top_n(ck, score, ascending):

    out = {}
    for g in ["real", "random"]:
        d = ck[ck.group == g].sort_values(score, ascending=ascending).groupby("period").head(config.TOP_N)
        out[g] = d.groupby("period").agg(net=("net", "mean"), leash=("half_life_ok", "mean"))
    return out["real"], out["random"]


def main():
    ve = pd.read_csv(config.RUN_DIR / "vecm.csv")
    ve = ve[ve.tradeable]
    lp = load_table(config.PROCESSED / "log_close_1h")
    try:
        fund = load_table(config.PROCESSED / "funding_1h")
    except FileNotFoundError:
        print("no funding_1h yet (run scripts/pull_funding.py): funding counted as zero")
        fund = pd.DataFrame(index=lp.index)
    costs = Costs()
    print(f"run {config.RUN}", flush=True)
    rows = []
    for m, part in ve.groupby("period", sort=True):
        if not periods.checkable(m):
            rows += [dict(r._asdict(), checked=False) for r in part.itertuples(index=False)]
            continue
        w = periods.test_window(lp, m)
        fc = fund.reindex(w.index).fillna(0).cumsum()
        for r in part.itertuples(index=False):
            res = dict(r._asdict(), checked=True)
            x = periods.prices(w, r.kind, r.coins.split(";"))
            if len(x) < 48:
                res.update(
                    n_hours_next=len(x),
                    half_life_ok=False,
                    still_leashed=False,
                    trades=0,
                    bounce=0.0,
                    costs=0.0,
                    funding=0.0,
                    net=0.0,
                    made_money=False,
                )
            else:
                res.update(check(r, x, fc, costs))
            rows.append(res)
        d = [x for x in rows[-len(part) :] if x["group"] == "real"]
        print(
            f"  {m}: half-life still ok for {sum(x['half_life_ok'] for x in d)}/{len(d)} real leashes",
            flush=True,
        )
    pe = pd.DataFrame(rows)
    pe.to_csv(config.RUN_DIR / "persistence.csv", index=False, float_format="%.6g")

    ck = pe[pe.checked].copy()
    ck["strength"] = ck.trace / ck.trace_5pct
    live = pe[~pe.checked & (pe.group == "real")]
    unit = "week" if config.REPICK == "W" else "month"
    print(
        f"\nchecked {len(ck):,} leashes ({(ck.group == 'real').sum():,} real, {(ck.group == 'random').sum():,} random) "
        f"over {ck.period.nunique()} {unit}s, each on the {unit} after it was found, weights frozen; "
        f"{len(live)} real leashes in {', '.join(sorted(live.period.unique())) or '-'} are live (no data yet)"
    )
    g = ck.groupby("group")
    tab = pd.DataFrame(
        {
            "half-life still 6h-10d": g.half_life_ok.mean().map("{:.1%}".format),
            "  and ADF agrees (on a leash)": g.still_leashed.mean().map("{:.1%}".format),
            "drifted away (>1 swing)": g.drift_next.apply(lambda v: (v.abs() > 1).mean()).map(
                "{:.1%}".format
            ),
            "yardstick trades": g.trades.mean().map("{:.1f}".format),
            "bounce per $": g.bounce.mean().map("{:+.2%}".format),
            "- fees and slippage": g.costs.mean().map("{:.2%}".format),
            "- funding": g.funding.mean().map("{:+.2%}".format),
            "= net per $": g.net.mean().map("{:+.2%}".format),
            "  95% range of that average": g.net.apply(lambda v: 1.96 * v.std() / np.sqrt(len(v))).map(
                "+-{:.2%}".format
            ),
            "made money": g.made_money.mean().map("{:.1%}".format),
            "half-life ok AND money": (ck.half_life_ok & ck.made_money)
            .groupby(ck.group)
            .mean()
            .map("{:.1%}".format),
            "on a leash AND money": (ck.still_leashed & ck.made_money)
            .groupby(ck.group)
            .mean()
            .map("{:.1%}".format),
        }
    ).T[["real", "random"]]
    print(f"\naverage per leash, the {unit} after (real vs random):")
    print(tab.to_string())
    cols = pd.MultiIndex.from_product([["leash", "net"], ["real", "random"]])
    print("\nby kind (half-life still ok / average net per $):")
    kt = (
        ck.groupby(["kind", "group"])
        .agg(leash=("half_life_ok", "mean"), net=("net", "mean"))
        .unstack()
        .reindex(columns=cols)
    )
    for k in kt.index:
        print(
            f"  {k:6s}  real {kt.at[k, ('leash', 'real')]:5.1%} / {kt.at[k, ('net', 'real')]:+.2%}    "
            f"random {kt.at[k, ('leash', 'random')]:5.1%} / {kt.at[k, ('net', 'random')]:+.2%}"
        )
    print(f"\nby month (half-life still ok: real vs random | average net per $ per {unit}: real vs random):")
    pm = (
        ck.groupby([ck.period.str[:7], "group"])
        .agg(leash=("half_life_ok", "mean"), net=("net", "mean"))
        .unstack()
        .reindex(columns=cols)
    )
    for m in pm.index:
        print(
            f"  {m}:  {pm.at[m, ('leash', 'real')]:5.1%} vs {pm.at[m, ('leash', 'random')]:5.1%}   |  "
            f"{pm.at[m, ('net', 'real')]:+.2%} vs {pm.at[m, ('net', 'random')]:+.2%}"
        )
    pp = ck.groupby(["period", "group"]).net.mean().unstack().dropna()
    print(
        f"{unit}s where real leashes earned more than random ones: {(pp.real > pp.random).sum()} of {len(pp)}"
    )

    print(
        f"\nonly the top {config.TOP_N} leashes per {unit} by an in-sample score "
        f"(average net per $ per {unit}, 95% range):"
    )
    for name, score, asc in [
        ("strongest Johansen", "strength", False),
        ("fastest half-life", "half_life_h", True),
    ]:
        r, x = top_n(ck, score, asc)
        both = r.join(x, lsuffix="_r", rsuffix="_x", how="inner")
        print(
            f"  {name:19s} real {r.net.mean():+.2%} (+-{1.96 * r.net.std() / np.sqrt(len(r)):.2%}), "
            f"positive in {(r.net > 0).sum()}/{len(r)} {unit}s, half-life ok {r.leash.mean():.0%}  |  "
            f"random top {config.TOP_N}: {x.net.mean():+.2%}, half-life ok {x.leash.mean():.0%}  |  "
            f"real beat random in {(both.net_r > both.net_x).sum()}/{len(both)}"
        )

    good = ck[(ck.group == "real") & ck.half_life_ok & ck.made_money]
    last = ck.period.max()
    short = lambda s: ";".join(x.replace("USDT", "") for x in s.split(";"))
    print(f"\nnewest checked {unit} ({last}): real leashes whose half-life held AND that made money")
    for r in good[good.period == last].sort_values("net", ascending=False).head(12).itertuples():
        print(
            f"  {r.kind:6s} {short(r.coins):38s} net {r.net:+.2%}  trades {int(r.trades)}  "
            f"half-life {r.half_life_h:.0f}h -> {r.half_life_next:.0f}h"
        )
    print(f"\nsaved persistence.csv in {config.RUN_DIR}")


if __name__ == "__main__":
    main()
