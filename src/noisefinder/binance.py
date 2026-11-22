import io
import re
import time
import zipfile

import pandas as pd
import requests

from noisefinder import config

S = requests.Session()
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]


def get(url, params=None, headers=None, tries=6):

    for k in range(tries):
        try:
            r = S.get(url, params=params, headers=headers, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 451:
                raise RuntimeError("HTTP 451: Binance refuses this location (check your VPN)")
            if r.status_code in (403, 418, 429) or r.status_code >= 500:
                time.sleep(min(60, 5 * 2**k))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            time.sleep(min(60, 5 * 2**k))
    raise RuntimeError(f"failed after {tries} tries: {url}")


def list_bulk(prefix, delimiter=None):

    out, marker = [], ""
    while True:
        p = {"prefix": prefix, "marker": marker}
        if delimiter:
            p["delimiter"] = delimiter
        x = get(config.S3_LIST, p).text
        keys = (
            re.findall(r"<Prefix>(.*?)</Prefix>", x)[1:] if delimiter else re.findall(r"<Key>(.*?)</Key>", x)
        )
        out += keys
        if "<IsTruncated>true</IsTruncated>" not in x or not keys:
            return out
        marker = keys[-1]


def read_zip(content):

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    d = pd.read_csv(io.StringIO(raw), header=0 if raw[:1].isalpha() else None).iloc[:, :11]
    d.columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_base",
        "taker_quote",
    ]
    t = pd.to_numeric(d.open_time)
    d["open_time"] = pd.to_datetime(t, unit="us" if t.iloc[0] > 1e14 else "ms")
    return d[COLUMNS]


def month_file(symbol, month, interval="1h"):
    return config.RAW / f"binance_{interval}" / symbol / f"{month}.csv"


def funding_file(symbol, month):
    return config.RAW / "funding_monthly" / symbol / f"{month}.csv"


def read_funding_zip(content):

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    d = pd.read_csv(io.StringIO(raw), header=0 if raw[:1].isalpha() else None)
    t = pd.to_numeric(d.iloc[:, 0])
    return pd.DataFrame(
        {
            "funding_time": pd.to_datetime(t, unit="us" if t.iloc[0] > 1e14 else "ms"),
            "funding_rate": pd.to_numeric(d.iloc[:, -1], errors="coerce"),
        }
    )


def fetch_funding_month(symbol, month):

    f = funding_file(symbol, month)
    if f.exists():
        return "cached"
    r = get(
        f"{config.BINANCE_BULK}/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
    )
    if r is None:
        return "missing"
    f.parent.mkdir(parents=True, exist_ok=True)
    read_funding_zip(r.content).to_csv(f, index=False)
    return "ok"


def fetch_month(symbol, month, interval="1h"):

    f = month_file(symbol, month, interval)
    if f.exists():
        return "cached"
    r = get(
        f"{config.BINANCE_BULK}/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    )
    if r is None:
        return "missing"
    f.parent.mkdir(parents=True, exist_ok=True)
    read_zip(r.content).to_csv(f, index=False)
    return "ok"
