import pandas as pd

from noisefinder import config


def legs(kind, coins, weights):

    if kind == "sector":
        return {coins[0]: weights[0], **{c: weights[1] / (len(coins) - 1) for c in coins[1:]}}
    return dict(zip(coins, weights))


class Costs:
    def __init__(self):
        u = pd.read_csv(
            config.PROCESSED / "universe_monthly.csv", usecols=["month", "symbol", "median_daily_volume_usd"]
        )
        self.volume = u.set_index(["month", "symbol"]).median_daily_volume_usd.to_dict()

    def per_side(self, month, symbol):

        v = self.volume.get((month, symbol), 0.0)
        return config.FEE_PER_SIDE + next(bp for floor, bp in config.SLIPPAGE_BP if v >= floor) / 1e4

    def round_trip(self, month, kind, coins, weights):

        return 2 * sum(abs(n) * self.per_side(month, s) for s, n in legs(kind, coins, weights).items())
