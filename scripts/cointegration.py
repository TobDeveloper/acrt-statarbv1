import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from noisefinder import config, periods
from noisefinder.io import load_table

warnings.simplefilter("ignore")


def test(x):
    k = x.shape[1]
    jo = coint_johansen(x, 0, config.JOHANSEN_LAGS)
    rank = 0
    while rank < k and jo.lr1[rank] > jo.cvt[rank, 1]:
        rank += 1
    wt = jo.evec[:, 0].real
    wt = wt / wt[0]
    out = dict(
        n_hours=len(x),
        rank=rank,
        trace=jo.lr1[0],
        trace_5pct=jo.cvt[0, 1],
        weights=";".join(f"{v:.4f}" for v in wt),
        adf_p=np.nan,
        eg_p=np.nan,
        passed=False,
    )
    if 1 <= rank < k:
        out["adf_p"] = adfuller(x @ wt, regression="c", autolag="AIC")[1]
        out["eg_p"] = coint(x[:, 0], x[:, 1:], trend="c", autolag="aic")[1]
        out["passed"] = bool(out["eg_p"] < config.SIGNIFICANCE)
    return out


def main(budget_s=None):
    cand = pd.read_csv(config.RUN_DIR / "candidates.csv")
    lp = load_table(config.PROCESSED / "log_close_1h")
    cache = config.RUN_DIR / "cointegration_cache"
    cache.mkdir(exist_ok=True)
    t0 = time.time()
    for m, part in cand.groupby("period", sort=True):
        f = cache / f"{m}.csv"
        if f.exists():
            continue
        if budget_s and time.time() - t0 > budget_s:
            break
        w = periods.formation(lp, m)
        rows = []
        for r in part.itertuples(index=False):
            res = dict(r._asdict())
            try:
                res.update(test(periods.prices(w, r.kind, r.coins.split(";")).to_numpy()))
            except Exception as e:
                res.update(passed=False, error=str(e)[:80])
            rows.append(res)
        pd.DataFrame(rows).to_csv(f, index=False)
        print(
            f"  {m}: {sum(r['passed'] for r in rows if r['group'] == 'real')} real candidates pass",
            flush=True,
        )
    left = [m for m in cand.period.unique() if not (cache / f"{m}.csv").exists()]
    if left:
        raise SystemExit(f"{len(left)} months left: run again")
    jo = pd.concat([pd.read_csv(cache / f"{m}.csv") for m in sorted(cand.period.unique())])
    jo.to_csv(config.RUN_DIR / "johansen.csv", index=False, float_format="%.6g")
    print("pass rate by group:", jo.groupby("group").passed.mean().round(3).to_dict())


if __name__ == "__main__":
    import sys

    main(float(sys.argv[1]) if len(sys.argv) > 1 else None)
