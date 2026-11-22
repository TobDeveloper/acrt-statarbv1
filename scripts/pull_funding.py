import concurrent.futures as cf
import time

import pandas as pd

from noisefinder import config
from noisefinder.binance import fetch_funding_month, funding_file
from noisefinder.io import save_table


def split_old_files():

    n = 0
    for f in sorted((config.RAW / "funding").glob("*.csv")):
        d = pd.read_csv(f, usecols=["funding_time", "funding_rate"], parse_dates=["funding_time"])
        for m, part in d.groupby(d.funding_time.dt.strftime("%Y-%m")):
            out = funding_file(f.stem, m)
            last_day = pd.Period(m, "M").end_time.normalize()
            if not out.exists() and part.funding_time.max() >= last_day:
                out.parent.mkdir(parents=True, exist_ok=True)
                part.to_csv(out, index=False)
                n += 1
    return n


def main():
    print(f"copied coin files cut into {split_old_files():,} monthly files", flush=True)
    uni = pd.read_csv(config.PROCESSED / "universe_monthly.csv")
    last = (pd.Timestamp(config.DATA_END) - pd.Timedelta(hours=1)).strftime("%Y-%m")
    jobs = sorted({(r.symbol, r.month) for r in uni[uni.in_universe & (uni.month <= last)].itertuples()})
    todo = [(s, m) for s, m in jobs if not funding_file(s, m).exists()]
    print(
        f"{len(jobs):,} coin-months in the step 0 lists, {len(jobs) - len(todo):,} already here, "
        f"{len(todo):,} to download",
        flush=True,
    )
    done, t0 = {"ok": 0, "cached": 0, "missing": 0, "error": 0}, time.time()
    with cf.ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(fetch_funding_month, s, m): (s, m) for s, m in todo}
        for n, fu in enumerate(cf.as_completed(futs), 1):
            try:
                done[fu.result()] += 1
            except Exception as e:
                done["error"] += 1
                print("  error", futs[fu], e, flush=True)
            if n % 500 == 0 or n == len(todo):
                print(f"  {n:,}/{len(todo):,} {done}  {(time.time() - t0) / 60:.1f} min", flush=True)

    parts = {}
    for s in sorted({s for s, _ in jobs}):
        fs = sorted((config.RAW / "funding_monthly" / s).glob("*.csv"))
        if fs:
            d = pd.concat([pd.read_csv(f, parse_dates=["funding_time"]) for f in fs])
            d["hour"] = d.funding_time.dt.floor("h")
            parts[s] = d.groupby("hour").funding_rate.sum()
    idx = pd.date_range(config.DATA_START, pd.Timestamp(config.DATA_END) - pd.Timedelta(hours=1), freq="h")
    fund = pd.DataFrame(parts).reindex(idx).rename_axis("hour")
    save_table(fund, config.PROCESSED / "funding_1h")
    have = sum(funding_file(s, m).exists() for s, m in jobs)
    print(
        f"\nfunding found for {have:,} of {len(jobs):,} coin-months ({have / len(jobs):.0%}); "
        f"the rest count as zero funding in step 6",
        flush=True,
    )
    rates = fund.stack().dropna()
    print(
        f"saved funding_1h: {len(fund.columns)} coins, {len(rates):,} payments, average rate {rates.mean():.4%} "
        f"per payment (positive: longs pay shorts)",
        flush=True,
    )


if __name__ == "__main__":
    main()
