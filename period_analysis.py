"""Which period-level conditions predict good vs bad periods for the aux4 model?

Uses explore_per_exped.csv (per-period FIT of each idea on 4 test blocks) and period stats that
can also be computed on a live round (same-period rows only): basket count, missing share,
feature dispersion. Also measures persistence: does a good stretch predict the next one?
"""
import numpy as np
import pandas as pd

from research import load

pe = pd.read_csv("explore_per_exped.csv", index_col=0)
fit = pe["aux4"].dropna()
df, feats = load()
g = df[df["exped"].isin(fit.index)].groupby("exped")
stats = pd.DataFrame({
    "n_rows": g.size(),
    "missing_share": g[feats].apply(lambda x: x.isna().mean().mean()),
    "feat_dispersion": g[feats].std().mean(axis=1),
    "feat_mean": g[feats].mean().mean(axis=1),
}).loc[fit.index]
stats["fit"] = fit

print(f"{len(fit)} test periods, aux4 FIT mean {fit.mean():.5f}, share positive {(fit > 0).mean():.2f}")
print("\nSpearman correlation of each period stat with that period's FIT:")
print(stats.corr(method="spearman")["fit"].drop("fit").round(3).to_string())

# persistence: mean FIT of a ~20-period block vs the next block (a round is about 20-40 periods)
s = fit.sort_index()
blocks = s.groupby(np.arange(len(s)) // 20).mean()
print(f"\nblock-to-next-block correlation (20-period blocks): {blocks.autocorr():.3f}  (n={len(blocks)})")
print("block means:", blocks.round(4).tolist())

q = pd.qcut(stats["feat_dispersion"], 4, labels=["low", "mid-low", "mid-high", "high"])
print("\nFIT by feature-dispersion quartile:\n" + stats.groupby(q, observed=True)["fit"].agg(["mean", "count"]).round(5).to_string())
