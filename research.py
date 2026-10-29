"""Holdout research: baseline vs benchmark-residualised vs benchmark-neutralised.

Scores each candidate with the event formula (FIT + 3*UNQ - B_FIT; INOV skipped,
its core feature set is not published) on a time-ordered, embargoed holdout.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

DATA = "hackathon-example-scripts/futures_train.parquet"
BENCH = "hackathon-example-scripts/benchmark_futures_train.parquet"
TARGET = "target_everest"
EMBARGO = 20      # expeds between fit and holdout (target is a ~20-day forward return)
HOLDOUT_FRAC = 0.15
B = 0.6366197723675814

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, max_depth=6,
                  colsample_bytree=0.1, subsample=0.7, subsample_freq=1,
                  min_child_samples=200, n_jobs=-1, verbose=-1)


def load():
    df = pd.read_parquet(DATA)
    bench = pd.read_parquet(BENCH)
    df = df.join(bench[["v1_sherpa"]], how="left")
    feats = [c for c in df.columns if c.startswith("feature")]
    df[feats] = df[feats].replace(-1, np.nan).astype("float32")
    return df, feats


def gauss_rank(s):
    r = s.rank(method="average")
    return pd.Series(norm.ppf((r - 0.5) / len(r)), index=s.index)


def neutralize(p, ref, proportion=1.0):
    X = np.column_stack([ref, np.ones(len(ref))])
    beta = np.linalg.lstsq(X, p, rcond=None)[0]
    return p - proportion * (X @ beta)


def per_exped_neutral(df, col, ref_col, proportion=1.0):
    out = pd.Series(index=df.index, dtype="float64")
    for _, g in df.groupby("exped"):
        out.loc[g.index] = neutralize(gauss_rank(g[col]).to_numpy(), gauss_rank(g[ref_col]).to_numpy(), proportion)
    return out


def score(df, col):
    fits, unqs = [], []
    for _, g in df.groupby("exped"):
        t = g[TARGET] - g[TARGET].mean()
        p = gauss_rank(g[col]).to_numpy()
        b = gauss_rank(g["v1_sherpa"]).to_numpy()
        fits.append(np.mean(p * t))
        unqs.append(np.mean(neutralize(p, b) * t))
    return np.mean(fits), np.mean(unqs)


def residual_target(df):
    out = pd.Series(index=df.index, dtype="float64")
    for _, g in df.groupby("exped"):
        t = (g[TARGET] - g[TARGET].mean()).to_numpy()
        out.loc[g.index] = neutralize(t, gauss_rank(g["v1_sherpa"]).to_numpy())
    return out


def main():
    t0 = time.time()
    df, feats = load()
    expeds = sorted(df["exped"].unique())
    n_hold = int(len(expeds) * HOLDOUT_FRAC)
    hold_e = set(expeds[-n_hold:])
    fit_e = set(expeds[: -(n_hold + EMBARGO)])
    tr, ho = df[df["exped"].isin(fit_e)], df[df["exped"].isin(hold_e)].copy()
    print(f"{len(expeds)} expeds; fit {len(fit_e)} ({len(tr):,} rows), holdout {n_hold} ({len(ho):,}), "
          f"bench coverage {ho['v1_sherpa'].notna().mean():.3f}", flush=True)
    tr = tr[tr[TARGET].notna()]

    m_a = lgb.LGBMRegressor(**LGB_PARAMS).fit(tr[feats], tr[TARGET])
    ho["A"] = m_a.predict(ho[feats]); print(f"A fitted {time.time()-t0:.0f}s", flush=True)

    m_rep = lgb.LGBMRegressor(**LGB_PARAMS).fit(tr[feats], tr["v1_sherpa"])
    ho["bench_rep"] = m_rep.predict(ho[feats])
    print(f"bench replica fitted {time.time()-t0:.0f}s; corr with real bench "
          f"{np.corrcoef(ho['bench_rep'], ho['v1_sherpa'])[0,1]:.3f}", flush=True)

    m_b = lgb.LGBMRegressor(**LGB_PARAMS).fit(tr[feats], residual_target(tr))
    ho["B"] = m_b.predict(ho[feats]); print(f"B fitted {time.time()-t0:.0f}s", flush=True)

    for prop in (0.5, 1.0):
        ho[f"C{prop}"] = per_exped_neutral(ho, "A", "bench_rep", prop)
        ho[f"B_C{prop}"] = per_exped_neutral(ho, "B", "bench_rep", prop)
    ho["AB"] = gauss_rank(ho["A"]) + gauss_rank(ho["B"])

    bfit, _ = score(ho, "v1_sherpa")
    print(f"\nbenchmark FIT on holdout: {bfit:.5f}")
    print(f"{'model':10} {'FIT':>9} {'UNQ':>9} {'S=FIT+3UNQ-BFIT':>16} {'score':>8}")
    for col in ["v1_sherpa", "A", "B", "C0.5", "C1.0", "B_C0.5", "B_C1.0", "AB"]:
        f, u = score(ho, col)
        s = f + 3 * u - bfit
        print(f"{col:10} {f:9.5f} {u:9.5f} {s:16.5f} {B*np.arctan(3*s/B):8.4f}")
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
