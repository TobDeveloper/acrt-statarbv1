import numpy as np
import pandas as pd

from noisefinder import config
from noisefinder.io import load_table, save_table


def fill_short_gaps(s, max_hours):

    gap = s.isna()
    run_id = (gap != gap.shift()).cumsum()
    run_len = gap.groupby(run_id).transform("sum")
    started = s.ffill().notna()
    fill = gap & (run_len <= max_hours) & started
    out = s.copy()
    out[fill] = s.ffill()[fill]
    return out, int(fill.sum())


def main():
    close = load_table(config.PROCESSED / "close_1h")
    qv = load_table(config.PROCESSED / "quote_volume_1h")
    coins = pd.read_csv(config.PROCESSED / "coins.csv", parse_dates=["listed"]).set_index("symbol")
    print(f"cleaning {close.shape[1]} symbols x {close.shape[0]:,} hours ...", flush=True)

    had_price = close.notna()

    no_trade = had_price & qv.reindex_like(close).fillna(0).le(0)
    lp = np.log(close.where(~no_trade))

    listed = coins.listed.reindex(lp.columns)
    before = pd.DataFrame(
        lp.index.values[:, None] < listed.values[None, :], index=lp.index, columns=lp.columns
    )
    before &= lp.notna()
    lp = lp.where(~before)

    med = lp.rolling(
        config.SPIKE_WINDOW_HOURS, center=True, min_periods=config.SPIKE_WINDOW_HOURS // 2
    ).median()
    dev = lp - med
    del med
    spike = dev.abs() > config.SPIKE_LOG
    r, c = np.nonzero(spike.to_numpy())
    sp = pd.DataFrame(
        {"hour": lp.index[r], "symbol": lp.columns[c], "log_dev": dev.to_numpy()[r, c].round(3)}
    )
    del dev
    sp["price_vs_median"] = np.exp(sp.log_dev).round(3)
    sp.sort_values(["symbol", "hour"]).to_csv(config.PROCESSED / "spikes.csv", index=False)
    lp = lp.where(~spike)

    filled = {}
    for c in lp.columns:
        lp[c], filled[c] = fill_short_gaps(lp[c], config.MAX_FILL_HOURS)
    save_table(lp, config.PROCESSED / "log_close_1h")

    life = lp.notna().cumsum().gt(0) & lp[::-1].notna().cumsum()[::-1].gt(0)
    rep = pd.DataFrame(
        {
            "hours_with_price": had_price.sum(),
            "no_trade_hours": no_trade.sum(),
            "before_listing_hours": before.sum(),
            "spike_hours": spike.sum(),
            "gap_hours_filled": pd.Series(filled),
            "coverage_in_life": (lp.notna() & life).sum() / life.sum().replace(0, np.nan),
        }
    ).rename_axis("symbol")
    rep.round(4).to_csv(config.PROCESSED / "cleaning_report.csv")

    uni = pd.read_csv(config.PROCESSED / "universe_monthly.csv")
    cov = []
    for m in sorted(uni.month.unique()):
        start = pd.Period(m, "M").start_time
        w = lp.loc[start - pd.Timedelta(days=config.LOOKBACK_DAYS) : start - pd.Timedelta(hours=1)]
        cov.append(
            pd.DataFrame(
                {
                    "month": m,
                    "symbol": lp.columns,
                    "coverage_30d": (w.notna().sum() / (config.LOOKBACK_DAYS * 24)).values,
                }
            )
        )
    cov = pd.concat(cov)
    us = uni.merge(cov, on=["month", "symbol"], how="left")
    us["usable"] = us.in_universe & (us.coverage_30d >= config.MIN_COVERAGE)
    us[["month", "symbol", "base", "in_universe", "coverage_30d", "usable"]].round(4).to_csv(
        config.PROCESSED / "usable_monthly.csv", index=False
    )

    print(
        f"\nhours blanked: {no_trade.values.sum():,} no-trade, {before.values.sum():,} before a relisting, "
        f"{spike.values.sum():,} spikes;  gap hours filled: {sum(filled.values()):,}"
    )
    top = rep.spike_hours.sort_values(ascending=False)
    print(
        f"spikes found in {(top > 0).sum()} coins; most: "
        + ", ".join(f"{s.replace('USDT', '')} {int(v)}" for s, v in top.head(8).items())
    )
    if len(sp):
        big = sp.loc[sp.log_dev.abs().idxmax()]
        print(
            f"biggest spike: {big.symbol} at {big.hour:%Y-%m-%d %H:00}, {big.price_vs_median}x its local median"
        )
    print(
        f"coverage over each coin's life: median {rep.coverage_in_life.median():.2%}, "
        f"lowest {rep.coverage_in_life.min():.2%} ({rep.coverage_in_life.idxmin()})"
    )
    per = us.groupby("month").agg(in_list=("in_universe", "sum"), usable=("usable", "sum"))
    print("\ncoins per month: in the step 0 list -> usable after cleaning")
    print(per.to_string())
    lost = us[us.in_universe & ~us.usable]
    if len(lost):
        print(
            f"\nlost to coverage: {len(lost)} coin-months, e.g. "
            + ", ".join(f"{r.base} {r.month} ({r.coverage_30d:.0%})" for r in lost.head(6).itertuples())
        )
    print(f"\nsaved log_close_1h, cleaning_report.csv, spikes.csv, usable_monthly.csv in {config.PROCESSED}")


if __name__ == "__main__":
    main()
