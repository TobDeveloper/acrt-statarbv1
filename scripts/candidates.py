from itertools import combinations

import numpy as np
import pandas as pd

from noisefinder import config, periods
from noisefinder.io import load_table


def mean_pairwise(c, members):
    return float(np.mean([c.at[a, b] for a, b in combinations(members, 2)]))


def residual_moves(w):

    r = w.diff()
    mkt = r.mean(axis=1)
    beta = r.apply(lambda s: s.cov(mkt) / mkt[s.notna()].var())
    return r - np.outer(mkt.to_numpy(), beta.to_numpy())


def links(coins, eligible):

    c = coins.reindex(eligible)
    home = c.home_chain.fillna("").astype(str).to_dict()
    home_name = c.home_chain_name.fillna("").astype(str).to_dict()
    cats = {
        s: [x.strip() for x in str(c.at[s, "categories"]).split(";") if x.strip() and x.strip() != "nan"]
        for s in eligible
    }
    size = pd.Series([x for s in eligible for x in cats[s]], dtype=object).value_counts()
    ok = {
        x
        for x, n in size.items()
        if n <= config.MAX_CATEGORY_SIZE and not any(w in x.lower() for w in config.NOT_A_REASON)
    }
    cats = {s: set(v) & ok for s, v in cats.items()}

    def link(a, b):
        ca = cats[a] & cats[b]
        if ca:
            return "category: " + sorted(ca, key=lambda x: (size[x], x))[0]
        if home[a] and home[a] == home[b]:
            return "chain: " + (home_name[a] or home[a])
        return ""

    return link


def period_candidates(m, lp, usable, coins, rng):

    w = periods.formation(lp, m)
    cov = w.notna().mean()
    eligible = [s for s in usable if s in w.columns and cov[s] >= config.MIN_FORMATION_COVERAGE]
    w = w[eligible]
    need = int(config.MIN_FORMATION_COVERAGE * len(w))
    c_lp = w.corr(min_periods=need)
    c_res = residual_moves(w).corr(min_periods=need)
    link = links(coins, eligible)
    rows = []

    pairs, tally = [], dict(all=0, logp=0, resid=0, link=0, all3=0)
    for a, b in combinations(eligible, 2):
        p_lp, p_res, why = (
            c_lp.at[a, b] > config.MIN_CORR_LOGP,
            c_res.at[a, b] >= config.MIN_RESID_CORR,
            link(a, b),
        )
        tally["all"] += 1
        tally["logp"] += p_lp
        tally["resid"] += p_res
        tally["link"] += bool(why)
        if p_lp and p_res and why:
            tally["all3"] += 1
            pairs.append((c_lp.at[a, b], c_res.at[a, b], a, b, why))
    pairs.sort(reverse=True)
    for lpc, rc, a, b, why in pairs[: config.MAX_PAIRS]:
        rows.append(dict(kind="pair", group="real", coins=f"{a};{b}", link=why, corr_logp=lpc, corr_resid=rc))

    seen, baskets = set(), []
    for _, _, a, b, why in pairs[: 3 * config.MAX_PAIRS]:
        members = [a, b]
        for _ in range(2):
            pool = [x for x in eligible if x not in members]
            pool = [x for x in pool if c_lp.loc[x, members].min() > config.MIN_CORR_LOGP]
            if not pool:
                break
            worst = c_res.loc[pool, members].min(axis=1)
            best = worst.idxmax()
            if not worst[best] >= config.MIN_RESID_CORR:
                break
            members = members + [best]
            key = frozenset(members)
            if key not in seen:
                seen.add(key)
                baskets.append(
                    (mean_pairwise(c_lp, members), mean_pairwise(c_res, members), sorted(members), why)
                )
    baskets.sort(key=lambda t: t[0], reverse=True)
    for ml, mr, members, why in baskets[: config.MAX_BASKETS]:
        rows.append(
            dict(kind="basket", group="real", coins=";".join(members), link=why, corr_logp=ml, corr_resid=mr)
        )

    sectors = []
    for s in eligible:
        sim = c_res[s].drop(s).dropna().nlargest(config.LOOKALIKES)
        if len(sim) < config.LOOKALIKES or sim.mean() < config.MIN_RESID_CORR:
            continue
        sector = w[sim.index].mean(axis=1)
        sectors.append((sim.mean(), float(w[s].corr(sector)), s, list(sim.index)))
    sectors.sort(reverse=True)
    for rc, lpc, s, peers in sectors[: config.MAX_SECTOR]:
        rows.append(
            dict(
                kind="sector",
                group="real",
                coins=";".join([s] + peers),
                link="5 look-alikes",
                corr_logp=lpc,
                corr_resid=rc,
            )
        )

    n_pair = min(len(pairs), config.MAX_PAIRS)
    n_basket = {k: sum(1 for b in baskets[: config.MAX_BASKETS] if len(b[2]) == k) for k in (3, 4)}
    n_sector = min(len(sectors), config.MAX_SECTOR)
    rand = (
        [("pair", 2)] * n_pair
        + [("basket", 3)] * n_basket[3]
        + [("basket", 4)] * n_basket[4]
        + [("sector", 1 + config.LOOKALIKES)] * n_sector
    )
    for kind, k in rand:
        pick = list(rng.choice(eligible, k, replace=False))
        if kind == "sector":
            ml = float(w[pick[0]].corr(w[pick[1:]].mean(axis=1)))
            mr = float(c_res.loc[pick[0], pick[1:]].mean())
        else:
            ml, mr = mean_pairwise(c_lp, pick), mean_pairwise(c_res, pick)
        rows.append(
            dict(kind=kind, group="random", coins=";".join(pick), link="", corr_logp=ml, corr_resid=mr)
        )

    out = pd.DataFrame(rows, columns=["kind", "group", "coins", "link", "corr_logp", "corr_resid"])
    out.insert(0, "period", m)
    return out, len(eligible), tally


def main():
    lp = load_table(config.PROCESSED / "log_close_1h")
    us = pd.read_csv(config.PROCESSED / "usable_monthly.csv")
    coins = pd.read_csv(config.PROCESSED / "coins.csv").set_index("symbol")
    if "home_chain" not in coins.columns:
        raise SystemExit("coins.csv has no home_chain column yet: run  python scripts/pull_data.py  first")
    rng = np.random.default_rng(config.SEED)
    months = [p for p in periods.all_periods() if periods.month(p) in set(us.month)]
    print(f"run {config.RUN}: {len(months)} periods, {months[0]} to {months[-1]}", flush=True)
    parts, info, tallies = [], [], []
    for m in months:
        usable = us[(us.month == periods.month(m)) & us.usable].symbol.tolist()
        out, n_elig, tally = period_candidates(m, lp, usable, coins, rng)
        parts.append(out)
        tallies.append(tally)
        info.append(
            dict(
                period=m,
                coins=n_elig,
                pairs_passing=tally["all3"],
                **out[out.group == "real"]
                .kind.value_counts()
                .reindex(["pair", "basket", "sector"], fill_value=0)
                .to_dict(),
            )
        )
        print(f"  {m}: {n_elig} coins, {tally['all3']} pairs pass the filters", flush=True)
    cand = pd.concat(parts, ignore_index=True)
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    cand.round(4).to_csv(config.RUN_DIR / "candidates.csv", index=False)

    inf = pd.DataFrame(info)
    inf["month"] = inf.period.str[:7]
    print("\nreal candidates kept per period, average by month (random controls: the same numbers):")
    print(inf.drop(columns="period").groupby("month").mean().round(0).astype(int).to_string())
    t = pd.DataFrame(tallies).sum()
    print(
        f"\nof all {int(t['all']):,} possible pairs over all periods (= what a random pair looks like), share passing:"
    )
    print(f"  log-price corr > {config.MIN_CORR_LOGP}:            {t['logp'] / t['all']:6.1%}")
    print(f"  market-removed corr >= {config.MIN_RESID_CORR}:       {t['resid'] / t['all']:6.1%}")
    print(f"  linked (category or home chain):  {t['link'] / t['all']:6.1%}")
    print(f"  all three:                        {t['all3'] / t['all']:6.1%}")
    real = cand[cand.group == "real"]
    print("\nwhat links the real pairs:")
    print(real[real.kind == "pair"].link.str.split(":").str[0].value_counts().to_string())
    print("most common links:")
    print(real[real.kind == "pair"].link.value_counts().head(8).to_string())
    print("\naverage correlations, real vs random (log price / market-removed hourly moves):")
    print(cand.groupby(["kind", "group"])[["corr_logp", "corr_resid"]].mean().round(2).to_string())
    last = months[-1]
    show = real[real.period == last]
    short = lambda s: ";".join(x.replace("USDT", "") for x in s.split(";"))
    print(f"\nnewest period ({last}), strongest of each kind (log-price corr / market-removed corr):")
    for kind in ["pair", "basket", "sector"]:
        top = show[show.kind == kind].head(5)
        print(
            f"  {kind}: "
            + " | ".join(f"{short(r.coins)} ({r.corr_logp:.2f}/{r.corr_resid:.2f})" for r in top.itertuples())
        )
    print(f"\nsaved candidates.csv ({len(cand):,} rows) in {config.RUN_DIR}")


if __name__ == "__main__":
    main()
