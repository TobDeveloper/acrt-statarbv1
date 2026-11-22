import numpy as np
import pandas as pd

from noisefinder import config
from noisefinder.io import load_table

RULES = ["real coin", "listed 1 year+", "actually trading", "liquid"]


def main():
    coins = pd.read_csv(config.PROCESSED / "coins.csv", parse_dates=["listed"]).set_index("symbol")
    qv = load_table(config.PROCESSED / "quote_volume_1h")
    coins = coins.loc[coins.index.intersection(qv.columns)]
    stable = coins.base.isin(config.STABLECOINS) | coins.categories.fillna("").str.lower().str.contains(
        "|".join(config.STABLE_WORDS)
    )
    index_like = coins.base.isin(config.NOT_COINS) | coins.underlying_type.astype(object).fillna("").isin(
        ["INDEX", "PREMARKET"]
    )
    real_coin = ~(stable | index_like)

    last = (pd.Timestamp(config.DATA_END)).to_period("M")
    months = pd.period_range(config.FIRST_UNIVERSE_MONTH, last, freq="M")
    hours = config.LOOKBACK_DAYS * 24
    rows = []
    for m in months:
        start = m.start_time
        win = qv.loc[
            start - pd.Timedelta(days=config.LOOKBACK_DAYS) : start - pd.Timedelta(hours=1), coins.index
        ]
        traded = (win > 0).sum() / hours
        daily = win.fillna(0).resample("1D").sum()
        median_vol = daily.median()
        listed_days = (start - coins.listed).dt.days
        ok = pd.DataFrame(
            {
                "real coin": real_coin,
                "listed 1 year+": listed_days >= config.MIN_LISTED_DAYS,
                "actually trading": traded >= config.MIN_TRADED_SHARE,
                "liquid": median_vol >= config.MIN_VOLUME_USD,
            }
        )
        first_fail = ok[RULES].apply(lambda r: next((k for k in RULES if not r[k]), ""), axis=1)
        alive = listed_days >= 0
        rows.append(
            pd.DataFrame(
                {
                    "month": str(m),
                    "symbol": coins.index,
                    "base": coins.base,
                    "listed_days": listed_days,
                    "traded_share": traded.round(3),
                    "median_daily_volume_usd": median_vol.round(0),
                    "in_universe": ok.all(axis=1),
                    "failed_rule": first_fail,
                }
            )[alive]
        )
    uni = pd.concat(rows, ignore_index=True)
    uni.to_csv(config.PROCESSED / "universe_monthly.csv", index=False)
    latest = uni[(uni.month == str(last)) & uni.in_universe].sort_values(
        "median_daily_volume_usd", ascending=False
    )
    latest.merge(coins[["name", "categories", "chain_names"]], left_on="symbol", right_index=True).to_csv(
        config.PROCESSED / "universe_latest.csv", index=False
    )

    per_month = uni[uni.in_universe].groupby("month").size()
    print(
        f"{len(coins)} symbols with data; coin lists built for {len(months)} months ({months[0]} to {months[-1]})\n"
    )
    print("coins allowed per month:")
    print(per_month.to_string())
    print(f"\nwhy coins were left out, newest month ({last}), first rule each one failed:")
    print(uni[uni.month == str(last)].failed_rule.replace("", "-> allowed").value_counts().to_string())
    sets = uni[uni.in_universe].groupby("month").symbol.apply(set)
    churn = [(len(b - a), len(a - b)) for a, b in zip(sets.iloc[:-1], sets.iloc[1:])]
    print(
        f"\nchange per month: on average {np.mean([c[0] for c in churn]):.1f} coins join and "
        f"{np.mean([c[1] for c in churn]):.1f} leave"
    )
    always = set.intersection(*sets)
    print(f"coins allowed in every month: {len(always)}")
    dead = uni[(uni.failed_rule == "actually trading") & (uni.median_daily_volume_usd == 0)].symbol.nunique()
    print(f"dead coins caught by the trading rule (listed, but no trades at all in a month): {dead}")
    rel = coins[coins.relisted].index.tolist()
    print(
        f"relisted symbols (same coin back, or a new coin on an old name like LIT): {len(rel)}; "
        f"only data from the current listing counts"
    )
    print(
        f"\nnewest list ({last}): {len(latest)} coins, largest by volume: " + ", ".join(latest.base.head(15))
    )
    print(f"\nsaved universe_monthly.csv and universe_latest.csv in {config.PROCESSED}")


if __name__ == "__main__":
    main()
