import json
import time

from noisefinder import config
from noisefinder.binance import get as _get

BASE = (
    "https://pro-api.coingecko.com/api/v3"
    if config.COINGECKO_PLAN == "pro"
    else "https://api.coingecko.com/api/v3"
)
HEADER = {
    "x-cg-pro-api-key" if config.COINGECKO_PLAN == "pro" else "x-cg-demo-api-key": config.COINGECKO_API_KEY
}


def get(path, params=None):
    r = _get(f"{BASE}{path}", params, HEADER)
    time.sleep(config.COINGECKO_PAUSE)
    return None if r is None else r.json()


def coin(cid):

    f = config.RAW / "coingecko" / f"{cid}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    d = (
        get(
            f"/coins/{cid}",
            {
                "localization": "false",
                "tickers": "false",
                "market_data": "false",
                "community_data": "false",
                "developer_data": "false",
            },
        )
        or {}
    )
    keep = {k: d.get(k) for k in ("id", "symbol", "name", "categories", "platforms", "asset_platform_id")}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(keep), encoding="utf-8")
    return keep
