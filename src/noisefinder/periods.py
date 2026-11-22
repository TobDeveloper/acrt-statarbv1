import pandas as pd

from noisefinder import config


def bounds(label):

    if len(label) == 7:
        start = pd.Timestamp(label + "-01")
        return start, start + pd.DateOffset(months=1)
    start = pd.Timestamp(label)
    return start, start + pd.Timedelta(days=7)


def all_periods():

    first = max(
        pd.Timestamp(config.DATA_START) + pd.Timedelta(days=config.FORMATION_DAYS),
        pd.Period(config.FIRST_UNIVERSE_MONTH, "M").start_time,
    )
    end = pd.Timestamp(config.DATA_END)
    if config.REPICK == "M":
        return [
            str(m)
            for m in pd.period_range(first.to_period("M"), end.to_period("M"), freq="M")
            if m.start_time >= first
        ]
    n = int((end - first) / pd.Timedelta(days=7))
    return [(end - pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d") for k in range(n, -1, -1)]


def month(label):

    return label[:7]


def formation(table, label):

    start, _ = bounds(label)
    return table.loc[start - pd.Timedelta(days=config.FORMATION_DAYS) : start - pd.Timedelta(hours=1)]


def test_window(table, label):

    start, end = bounds(label)
    return table.loc[start : end - pd.Timedelta(hours=1)]


def checkable(label):

    return bounds(label)[1] <= pd.Timestamp(config.DATA_END)


def prices(w, kind, coins):

    if kind == "sector":
        x = pd.concat([w[coins[0]], w[coins[1:]].mean(axis=1, skipna=False)], axis=1)
    else:
        x = w[coins]
    return x.dropna()
