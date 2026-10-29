"""Pre-stake risk gauge: share of a live round's periods with high feature dispersion.

High-dispersion periods (top quartile in the multi-period test) averaged FIT -0.012 vs about +0.003
otherwise. Uses only each period's own rows, so it is within the one-exped fair-play rule.
  python gauge.py <live parquet path>
"""
import sys

import numpy as np
import pandas as pd

from research import load

df, feats = load()
pe = pd.read_csv("explore_per_exped_run1.csv", index_col=0)
ref = df[df["exped"].isin(pe.index)].groupby("exped")[feats].std().mean(axis=1)
hi = ref.quantile(0.75)

live = pd.read_parquet(sys.argv[1])
era = "exped" if "exped" in live else "era"
x = live[feats].astype("float32").mask(live[feats] == -1)
disp = x.groupby(live[era]).std().mean(axis=1)
share = (disp > hi).mean()
print(f"{live[era].nunique()} periods in round; reference high-dispersion threshold {hi:.3f}")
print(f"round dispersion median {disp.median():.3f} (reference median {ref.median():.3f})")
print(f"share of round periods in the high-dispersion zone: {share:.0%} (baseline 25%)")
