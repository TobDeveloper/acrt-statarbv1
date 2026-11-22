import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noisefinder import backtest as bt
from noisefinder import config

HOLDOUT = pd.Timestamp(config.BT_HOLDOUT_FROM + "-01")
OUT = config.RESULTS / "backtest"


def summary(net):
    rows = []
    for name, sel in [
        ("Development", net.index < HOLDOUT),
        ("Hold-out", net.index >= HOLDOUT),
        ("All months", net.index >= net.index[0]),
    ]:
        s = bt.stats(net[sel])
        rows.append(
            dict(
                period=name,
                total=s["total"],
                ret_yr=s["ret"],
                vol_yr=s["vol"],
                sharpe=s["sharpe_d"],
                max_dd=s["maxdd"],
            )
        )
    return pd.DataFrame(rows).set_index("period")


def equity_chart(daily, path):
    net = daily.net
    s = bt.stats(net)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={"height_ratios": [2, 1]})
    a1.axvspan(HOLDOUT, net.index[-1], color="#000000", alpha=0.05, lw=0)
    a1.text(
        HOLDOUT, 0.97, "  hold-out", transform=a1.get_xaxis_transform(), va="top", fontsize=8, color="#555555"
    )
    a1.plot(net.index, daily.gross.cumsum() * 100, color="#9a9a9a", lw=1, label="before costs and funding")
    a1.plot(net.index, net.cumsum() * 100, color="#1f4e79", lw=1.6, label="after costs and funding")
    a1.axhline(0, color="#777777", lw=0.7)
    a1.set_ylabel("cumulative P&L, % of capital")
    a1.set_title(
        f"after costs: {s['ret']:+.1%}/yr, vol {s['vol']:.1%}, Sharpe {s['sharpe_d']:.2f}, max drawdown {s['maxdd']:+.1%}",
        loc="left",
        fontsize=8,
        color="#555555",
        pad=6,
    )
    a1.annotate(
        f"Backtest: {net.sum():+.1%} after costs over {len(net)} days",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 22),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
    )
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    mo = net.resample("MS").sum() * 100
    x = np.arange(len(mo))
    a2.bar(x, mo.values, color=["#1f4e79" if v > 0 else "#b03a2e" for v in mo.values])
    a2.set_xticks(x)
    a2.set_xticklabels([q.strftime("%y-%m") for q in mo.index], rotation=90, fontsize=7)
    a2.axhline(0, color="#777777", lw=0.7)
    a2.set_ylabel("net P&L per month, %")
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e3e3e3", lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def random_chart(real, rand, path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axvspan(HOLDOUT, real.index[-1], color="#000000", alpha=0.05, lw=0)
    ax.plot(
        real.index,
        real.cumsum() * 100,
        color="#1f4e79",
        lw=1.8,
        label=f"real spreads (Sharpe {bt.stats(real)['sharpe_d']:.2f})",
    )
    ax.plot(
        rand.index,
        rand.cumsum() * 100,
        color="#9a9a9a",
        lw=1.3,
        label=f"random control spreads, same rules (Sharpe {bt.stats(rand)['sharpe_d']:.2f})",
    )
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set_ylabel("cumulative P&L after costs, % of capital")
    ax.set_title("Real vs random spreads", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e3e3e3", lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    real_trades, real = bt.run("real")
    rand_trades, rand = bt.run("random")
    real.to_csv(OUT / "daily_real.csv", float_format="%.8g")
    rand.to_csv(OUT / "daily_random.csv", float_format="%.8g")
    table = pd.concat({"real": summary(real.net), "random": summary(rand.net)})
    table.to_csv(OUT / "summary.csv", float_format="%.6g")
    exits = pd.Series([t["how"] for t in real_trades]).value_counts()
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(real_trades)} trades; exits: {exits.to_dict()}")
    equity_chart(real, OUT / "equity.png")
    random_chart(real.net, rand.net, OUT / "real_vs_random.png")
    print(f"saved in {OUT}")


if __name__ == "__main__":
    main()
