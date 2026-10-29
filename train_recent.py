"""Starter-config models fitted on only the most recent slice of train (regime-shift test)."""
import sys
import time

import cloudpickle
import lightgbm as lgb
import numpy as np

from research import TARGET, load

PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
SLICES = {"R33": 1 / 3, "R16": 1 / 6, "R10": 1 / 10, "R20": 1 / 20, "R8": 1 / 8, "R12": 1 / 12, "R10N": 1 / 10}
NUMERAI = dict(n_estimators=858, learning_rate=0.023, max_depth=5, num_leaves=124, colsample_bytree=0.576,
               min_child_samples=204, reg_alpha=2, reg_lambda=0.01, subsample=0.858, subsample_freq=1, verbose=-1)

t0 = time.time()
df, feats = load()
df = df[df[TARGET].notna()]
with open("final_models.pkl", "rb") as f:
    models = cloudpickle.load(f)
expeds = sorted(df["exped"].unique())
for key in sys.argv[1:] or SLICES:
    if key == "RW":
        # full history, exponentially down-weighted with a half-life of 1/12 of the span
        pos = df["exped"].map({e: i for i, e in enumerate(expeds)}).to_numpy()
        w = 0.5 ** ((len(expeds) - 1 - pos) / (len(expeds) / 12))
        models[key] = lgb.LGBMRegressor(**PARAMS).fit(df[feats], df[TARGET], sample_weight=w)
    else:
        d = df[df["exped"].isin(expeds[-int(len(expeds) * SLICES[key]):])]
        models[key] = lgb.LGBMRegressor(**(NUMERAI if key.endswith("N") else PARAMS)).fit(d[feats], d[TARGET])
    print(f"{key} {time.time()-t0:.0f}s", flush=True)
with open("final_models.pkl", "wb") as f:
    cloudpickle.dump(models, f)
