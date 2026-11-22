import pandas as pd


def save_table(df, path):

    try:
        df.to_parquet(path.with_suffix(".parquet"))
    except ImportError:
        df.to_pickle(path.with_suffix(".pkl"))


def load_table(path):
    p = path.with_suffix(".parquet")
    return pd.read_parquet(p) if p.exists() else pd.read_pickle(path.with_suffix(".pkl"))


def split_symbol(symbol):

    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    for prefix, mult in (("1000000", 1_000_000), ("1000", 1000), ("1M", 1_000_000)):
        if base.startswith(prefix) and len(base) > len(prefix) + 1:
            return base[len(prefix) :], mult
    return base, 1
