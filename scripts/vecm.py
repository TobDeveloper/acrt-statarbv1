import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM

from noisefinder import config, periods
from noisefinder.costs import Costs
from noisefinder.io import load_table

warnings.simplefilter("ignore", FutureWarning)


def fit(x, round_trip):

    res = VECM(x, k_ar_diff=config.JOHANSEN_LAGS, coint_rank=1, deterministic="co").fit()
    b0 = res.beta[0, 0]
    wt, alpha = res.beta[:, 0] / b0, res.alpha[:, 0] * b0
    pull = float(wt @ alpha)
    hl = np.log(2) / -np.log1p(pull) if -1 < pull < 0 else np.inf
    spread = x @ wt
    swing, cost = float(spread.std()), float(round_trip(wt))
    if pull >= 0:
        why = "no pull back"
    elif hl < config.MIN_HALF_LIFE_H:
        why = "too fast"
    elif hl > config.MAX_HALF_LIFE_H:
        why = "too slow"
    elif swing < config.MIN_SWING_TO_COST * cost:
        why = "too small for costs"
    else:
        why = ""
    return dict(
        weights=";".join(f"{v:.4f}" for v in wt),
        alpha=";".join(f"{v:.5f}" for v in alpha),
        alpha_t=";".join(f"{v:.2f}" for v in res.tvalues_alpha[:, 0]),
        pull=pull,
        half_life_h=hl,
        spread_mean=float(spread.mean()),
        swing=swing,
        cost=cost,
        swing_to_cost=swing / cost,
        tradeable=why == "",
        why_not=why,
    )


def main():
    jo = pd.read_csv(config.RUN_DIR / "johansen.csv")
    lp = load_table(config.PROCESSED / "log_close_1h")
    passed = jo[jo.passed]
    costs = Costs()
    rows, errors = [], 0
    print(f"run {config.RUN}", flush=True)
    for m, part in passed.groupby("period", sort=True):
        w = periods.formation(lp, m)
        for r in part.itertuples(index=False):
            res = dict(r._asdict())
            try:
                coins = r.coins.split(";")
                res.update(
                    fit(
                        periods.prices(w, r.kind, coins).to_numpy(),
                        lambda wt: costs.round_trip(periods.month(m), r.kind, coins, wt),
                    )
                )
            except Exception as e:
                errors += 1
                res.update(tradeable=False, why_not="fit failed: " + str(e)[:60])
            rows.append(res)
        t = [x for x in rows[-len(part) :] if x["group"] == "real"]
        print(f"  {m}: {sum(x['tradeable'] for x in t)}/{len(t)} real leashes are tradeable", flush=True)
    ve = pd.DataFrame(rows)
    ve.to_csv(config.RUN_DIR / "vecm.csv", index=False, float_format="%.6g")

    real, rand = ve[ve.group == "real"], ve[ve.group == "random"]
    print(
        f"\nVECM fitted on {len(ve):,} step 4 passes ({len(real):,} real, {len(rand):,} random)"
        + (f"; {errors} could not be fitted" if errors else "")
    )
    print(
        f"\ntradeable (half-life {config.MIN_HALF_LIFE_H}h to {config.MAX_HALF_LIFE_H / 24:.0f} days, "
        f"swing >= {config.MIN_SWING_TO_COST}x cost), as a share of ALL candidates from step 3:"
    )
    n_all = jo.groupby(["kind", "group"]).size()
    n_ok = ve[ve.tradeable].groupby(["kind", "group"]).size().reindex(n_all.index, fill_value=0)
    tab = (n_ok / n_all).unstack()
    tab.loc["all"] = (
        ve[ve.tradeable].groupby("group").size().reindex(["random", "real"], fill_value=0)
        / jo.groupby("group").size()
    )
    cnt = n_ok.unstack()
    cnt.loc["all"] = cnt.sum()
    out = pd.DataFrame(
        {
            "real": [f"{cnt.at[k, 'real']:,} ({tab.at[k, 'real']:.1%})" for k in tab.index],
            "random": [f"{cnt.at[k, 'random']:,} ({tab.at[k, 'random']:.1%})" for k in tab.index],
        },
        index=tab.index,
    )
    print(out.to_string())
    luck = tab.at["all", "random"]
    n_real = int(cnt.at["all", "real"])
    print(
        f"\nrandom candidates end up tradeable {luck:.1%} of the time (the luck line). Real: {n_real:,} tradeable; "
        f"at the luck rate about {luck * (jo.group == 'real').sum():,.0f} would be there anyway"
    )
    print("\nwhy step 4 passes were dropped here (real / random):")
    print(
        pd.concat([real.why_not.value_counts(), rand.why_not.value_counts()], axis=1, keys=["real", "random"])
        .drop("", errors="ignore")
        .fillna(0)
        .astype(int)
        .to_string()
    )
    ok = real[real.tradeable]
    if len(ok):
        q = ok.half_life_h.quantile([0.25, 0.5, 0.75])
        print(
            f"\nhalf-life of the real tradeable ones: median {q[0.5]:.0f}h (middle half {q[0.25]:.0f}h to {q[0.75]:.0f}h); "
            f"swing median {ok.swing.median():.1%}, {ok.swing_to_cost.median():.0f}x the cost"
        )
    print("\nby month (tradeable real / tradeable random; weekly runs: all weeks of the month):")
    t = ve[ve.tradeable]
    pm = (
        t.groupby([t.period.str[:7], "group"])
        .size()
        .unstack(fill_value=0)
        .reindex(sorted(jo.period.str[:7].unique()), fill_value=0)
    )
    for m, r in pm.iterrows():
        print(f"  {m}: {r.get('real', 0):4d} / {r.get('random', 0):4d}")
    last = jo.period.max()
    top = ok[ok.period == last].sort_values("half_life_h")
    short = lambda s: ";".join(x.replace("USDT", "") for x in s.split(";"))
    print(f"\nnewest period ({last}), real tradeable noises, fastest first:")
    for r in top.head(12).itertuples():
        print(
            f"  {r.kind:6s} {short(r.coins):38s} half-life {r.half_life_h:5.0f}h  swing {r.swing:5.1%}  "
            f"{r.swing_to_cost:4.0f}x cost  weights {r.weights}"
        )
    print(f"\nsaved vecm.csv in {config.RUN_DIR}")


if __name__ == "__main__":
    main()
