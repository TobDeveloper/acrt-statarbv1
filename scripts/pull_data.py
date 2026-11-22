import argparse
import concurrent.futures as cf
import json
import re
import time

import pandas as pd

from noisefinder import config
from noisefinder.binance import fetch_month, get, list_bulk, month_file
from noisefinder.io import save_table, split_symbol

MANUAL_IDS = {
    "LUNA2": "terra-luna-2",
    "FTT": "ftx-token",
    "BTCST": "btc-standard-hashrate-token",
    "BNX": "binaryx-2",
    "XEM": "nem",
    "OMG": "omisego",
    "LUNC": "terra-luna",
    "RAY": "raydium",
    "ALPHA": "alpha-finance",
    "ATA": "automata",
    "LIT": "lighter",
    "LINA": "linear",
    "BEAMX": "beam-2",
    "DODOX": "dodo",
    "RAYSOL": "raydium",
}


def symbols(refresh):
    f = config.REFERENCE / "symbols.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f).dropna(subset=["months"])
    f.parent.mkdir(parents=True, exist_ok=True)
    syms = [p.rstrip("/").split("/")[-1] for p in list_bulk("data/futures/um/monthly/klines/", "/")]
    syms = [s for s in syms if s.endswith("USDT") and "_" not in s]
    print(f"listing months for {len(syms)} symbols ...", flush=True)

    def months(s):
        m = sorted(
            set(
                re.findall(
                    r"-1h-(\d{4}-\d{2})\.zip$",
                    "\n".join(list_bulk(f"data/futures/um/monthly/klines/{s}/1h/")),
                    flags=re.M,
                )
            )
        )
        return dict(
            symbol=s,
            first_month=m[0] if m else None,
            last_month=m[-1] if m else None,
            n_months=len(m),
            months=" ".join(m),
        )

    with cf.ThreadPoolExecutor(12) as ex:
        df = pd.DataFrame(list(ex.map(months, syms))).sort_values("symbol")
    df.to_csv(f, index=False)
    return df.dropna(subset=["months"])


def download(sy):
    lo, hi = config.DATA_START[:7], (pd.Timestamp(config.DATA_END) - pd.Timedelta(hours=1)).strftime("%Y-%m")
    jobs = [(r.symbol, m) for r in sy.itertuples() for m in r.months.split() if lo <= m <= hi]
    todo = [(s, m) for s, m in jobs if not month_file(s, m).exists()]
    print(
        f"{len(jobs):,} symbol-months in the window, {len(jobs) - len(todo):,} already here, {len(todo):,} to download",
        flush=True,
    )
    done, t0 = {"ok": 0, "cached": 0, "missing": 0, "error": 0}, time.time()
    with cf.ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(fetch_month, s, m): (s, m) for s, m in todo}
        for n, fu in enumerate(cf.as_completed(futs), 1):
            try:
                done[fu.result()] += 1
            except Exception as e:
                done["error"] += 1
                print("  error", futs[fu], e, flush=True)
            if n % 500 == 0 or n == len(todo):
                print(f"  {n:,}/{len(todo):,} {done}  {(time.time() - t0) / 60:.1f} min", flush=True)
    return done["ok"]


def build_tables(sy):
    idx = pd.date_range(config.DATA_START, pd.Timestamp(config.DATA_END) - pd.Timedelta(hours=1), freq="h")
    close, qv, t0 = {}, {}, time.time()
    for n, s in enumerate(sy.symbol, 1):
        fs = [
            f
            for f in sorted((config.RAW / "binance_1h" / s).glob("*.csv"))
            if config.DATA_START[:7] <= f.stem < config.DATA_END[:7]
        ]
        if fs:
            d = pd.concat([pd.read_csv(f, parse_dates=["open_time"]) for f in fs]).drop_duplicates(
                "open_time"
            )
            d = d.set_index("open_time").reindex(idx)
            if d.close.notna().any():
                close[s], qv[s] = d.close.astype("float64"), d.quote_volume.astype("float64")
        if n % 100 == 0:
            print(f"  read {n}/{len(sy)} symbols  {(time.time() - t0) / 60:.1f} min", flush=True)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    save_table(pd.DataFrame(close).rename_axis("hour"), config.PROCESSED / "close_1h")
    save_table(pd.DataFrame(qv).rename_axis("hour"), config.PROCESSED / "quote_volume_1h")
    print(f"saved close_1h and quote_volume_1h: {len(idx):,} hours x {len(close)} symbols", flush=True)
    return list(close)


def reference(online):

    from noisefinder.coingecko import get as cg

    config.REFERENCE.mkdir(parents=True, exist_ok=True)
    f_list, f_plat = (
        config.REFERENCE / "coingecko_coins_list.json",
        config.REFERENCE / "coingecko_asset_platforms.json",
    )
    f_mk = config.REFERENCE / "coingecko_markets.csv"
    if f_list.exists() and f_plat.exists() and f_mk.exists():
        return
    if not online:
        raise SystemExit(
            "the CoinGecko reference lists are missing: run once without --build-only (needs internet)"
        )
    if not config.COINGECKO_API_KEY:
        raise SystemExit(
            "COINGECKO_API_KEY is empty: put it in .env (COINGECKO_API_KEY=... and COINGECKO_PLAN=pro)"
        )
    print("downloading CoinGecko reference lists (coins, chains, top 5,000 by market cap) ...", flush=True)
    if not f_list.exists():
        f_list.write_text(json.dumps(cg("/coins/list", {"include_platform": "true"})), encoding="utf-8")
    if not f_plat.exists():
        f_plat.write_text(json.dumps(cg("/asset_platforms")), encoding="utf-8")
    if not f_mk.exists():
        rows = []
        for page in range(1, 21):
            got = (
                cg(
                    "/coins/markets",
                    {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page},
                )
                or []
            )
            rows += [
                {k: c.get(k) for k in ("id", "symbol", "name", "market_cap_rank", "market_cap")} for c in got
            ]
            if len(got) < 250:
                break
        pd.DataFrame(rows).to_csv(f_mk, index=False)
    print("  done", flush=True)


def exchange_info(online):
    f = config.REFERENCE / "exchange_info.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    if online:
        print("refreshing Binance exchange info (listing dates, which symbols trade today) ...", flush=True)
        try:
            r = get(f"{config.BINANCE_FAPI}/fapi/v1/exchangeInfo", tries=2)
            if r is not None:
                f.write_text(r.text, encoding="utf-8")
                print("  done", flush=True)
        except Exception as e:
            print(
                f"  Binance did not answer ({e})"
                + (
                    f"; using the saved copy from " f"{pd.Timestamp(f.stat().st_mtime, unit='s'):%Y-%m-%d}"
                    if f.exists()
                    else ""
                ),
                flush=True,
            )
    if not f.exists():
        print("  no exchange info: listing date = first month of data, 'still trading' unknown", flush=True)
        return pd.DataFrame(
            columns=["status", "contractType", "underlyingType", "listed"], index=pd.Index([], name="symbol")
        )
    s = pd.DataFrame(json.loads(f.read_text(encoding="utf-8"))["symbols"])
    s["listed"] = pd.to_datetime(s.onboardDate, unit="ms")
    return s.set_index("symbol")[["status", "contractType", "underlyingType", "listed"]]


def coin_facts(syms, sy, ex, online):
    from noisefinder.coingecko import coin

    lst = json.loads((config.REFERENCE / "coingecko_coins_list.json").read_text(encoding="utf-8"))
    plats = json.loads((config.REFERENCE / "coingecko_asset_platforms.json").read_text(encoding="utf-8"))
    mk = pd.read_csv(config.REFERENCE / "coingecko_markets.csv").dropna(subset=["symbol"])
    f_v2 = config.REFERENCE / "v2_universe.csv"
    v2 = (
        pd.read_csv(f_v2) if f_v2.exists() else pd.DataFrame(columns=["symbol", "coingecko_id", "categories"])
    ).set_index("symbol")
    by_id = {c["id"]: c for c in lst}
    first_id = {}
    for c in lst:
        first_id.setdefault(c["symbol"].lower(), c["id"])
    native = {}
    for p in plats:
        if p.get("native_coin_id"):
            native.setdefault(p["native_coin_id"], set()).add(p["id"])
    pname = {p["id"]: p.get("name") or p["id"] for p in plats}
    best = mk.sort_values("market_cap_rank").drop_duplicates("symbol").set_index("symbol").id
    months = sy.set_index("symbol")
    print(
        f"coin facts for {len(syms)} symbols (CoinGecko; coins looked up before are read from the cache) ...",
        flush=True,
    )
    rows, t0 = [], time.time()
    for n, s in enumerate(syms, 1):
        base, mult = split_symbol(s)
        cid = (
            MANUAL_IDS.get(base)
            or (v2.coingecko_id.get(s) if s in v2.index else None)
            or best.get(base.lower())
            or first_id.get(base.lower())
        )
        cats, plat, d = [], dict((by_id.get(cid) or {}).get("platforms") or {}), {}
        if cid and online:
            d = coin(cid)
            cats, plat = d.get("categories") or [], d.get("platforms") or plat
        elif cid and (config.RAW / "coingecko" / f"{cid}.json").exists():
            d = coin(cid)
            cats, plat = d.get("categories") or [], d.get("platforms") or plat
        elif s in v2.index:
            cats = [
                x.strip() for x in str(v2.at[s, "categories"]).split(";") if x.strip() and x.strip() != "nan"
            ]
        chains = {k for k in plat if k} | native.get(cid, set())
        first = pd.Timestamp(months.at[s, "first_month"] + "-01")
        listed = ex.listed.get(s) if s in ex.index else first
        rows.append(
            dict(
                symbol=s,
                base=base,
                multiplier=mult,
                listed=listed,
                relisted=bool(listed > first + pd.Timedelta(days=31)),
                trading_now=bool(s in ex.index and ex.at[s, "status"] == "TRADING"),
                underlying_type=ex.underlyingType.get(s, "") if s in ex.index else "",
                first_month=months.at[s, "first_month"],
                last_month=months.at[s, "last_month"],
                coingecko_id=cid,
                name=(by_id.get(cid) or {}).get("name"),
                categories="; ".join(cats),
                home_chain=d.get("asset_platform_id") or "",
                home_chain_name="",
                chains=";".join(sorted(chains)),
                chain_names=", ".join(sorted(pname.get(c, c) for c in chains)),
            )
        )
        if online and n % 100 == 0:
            print(f"  CoinGecko {n}/{len(syms)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    out = pd.DataFrame(rows)

    used = out.home_chain[out.home_chain != ""].value_counts()

    def own_chain(cid):
        nat = sorted(native.get(cid, set()))
        if not nat:
            return ""
        p = max(nat, key=lambda p: (used.get(p, 0), p == cid))
        return p if used.get(p, 0) else cid

    own = (out.home_chain == "") & out.coingecko_id.notna()
    out.loc[own, "home_chain"] = out.loc[own, "coingecko_id"].map(own_chain)
    out["home_chain_name"] = out.home_chain.map(lambda h: pname.get(h, h) if h else "")
    out.to_csv(config.PROCESSED / "coins.csv", index=False)
    print(
        f"saved coins.csv: {len(out)} symbols, {out.coingecko_id.notna().sum()} matched to CoinGecko, "
        f"{(out.categories != '').sum()} with categories, {out.trading_now.sum()} still trading today, "
        f"{out.relisted.sum()} relisted symbols: {', '.join(out[out.relisted].symbol)}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-symbols", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    online = not a.build_only
    sy = symbols(a.refresh_symbols and online)
    lo, hi = config.DATA_START[:7], (pd.Timestamp(config.DATA_END) - pd.Timedelta(hours=1)).strftime("%Y-%m")
    sy = sy[(sy.first_month <= hi) & (sy.last_month >= lo)]
    print(f"{len(sy)} USDT perpetuals traded at some point between {lo} and {hi}", flush=True)
    new_files = download(sy) if online else 0
    tables = config.PROCESSED / "close_1h.parquet"
    if new_files == 0 and tables.exists():
        syms = list(pd.read_parquet(tables).columns)
        print("no new files: hourly tables kept as they are", flush=True)
    else:
        syms = build_tables(sy)
    reference(online)
    coin_facts(syms, sy, exchange_info(online), online)


if __name__ == "__main__":
    main()
