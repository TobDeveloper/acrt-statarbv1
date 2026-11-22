import time

import numpy as np
import pandas as pd

from noisefinder import config
from noisefinder.io import load_table, save_table

lp = load_table(config.PROCESSED / "log_close_1h").astype(float)
qv = (
    load_table(config.PROCESSED / "quote_volume_1h").reindex(index=lp.index, columns=lp.columns).astype(float)
)
first, end = config.DATA_START[:7], config.DATA_END[:7]
hi, lo, t0 = {}, {}, time.time()
for n, s in enumerate(lp.columns, 1):
    fs = [f for f in sorted((config.RAW / "binance_1h" / s).glob("*.csv")) if first <= f.stem < end]
    if fs:
        d = pd.concat(
            [pd.read_csv(f, usecols=["open_time", "high", "low"], parse_dates=["open_time"]) for f in fs]
        )
        d = d.drop_duplicates("open_time").set_index("open_time").reindex(lp.index)
        hi[s], lo[s] = np.log(d.high.astype(float)), np.log(d.low.astype(float))
    if n % 100 == 0:
        print(f"  read {n}/{lp.shape[1]} symbols  {(time.time() - t0) / 60:.1f} min", flush=True)
hi = pd.DataFrame(hi, index=lp.index).reindex(columns=lp.columns)
lo = pd.DataFrame(lo, index=lp.index).reindex(columns=lp.columns)
ok = (qv.fillna(0) > 0) & lp.notna()
hi, lo = hi.where(ok, lp), lo.where(ok, lp)
hi = hi.where(hi - lp <= config.SPIKE_LOG, lp)
lo = lo.where(lp - lo <= config.SPIKE_LOG, lp)
hi, lo = np.maximum(hi, lp), np.minimum(lo, lp)
save_table(hi.astype("float32"), config.PROCESSED / "log_high_1h")
save_table(lo.astype("float32"), config.PROCESSED / "log_low_1h")
print(
    f"saved log_high_1h and log_low_1h ({lp.shape[0]:,} hours x {lp.shape[1]} symbols); "
    f"hours with a wick below the close: {float(((lp - lo) > 0).sum().sum() / lp.notna().sum().sum()):.0%}"
)
