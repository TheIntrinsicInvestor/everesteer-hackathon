"""LightGBM with the tuned Numerai-style params from the Tavily research (macalusojeff writeup)."""
import time

import cloudpickle
import lightgbm as lgb

from research import TARGET, load

t0 = time.time()
df, feats = load()
df = df[df[TARGET].notna()]
with open("final_models.pkl", "rb") as f:
    models = cloudpickle.load(f)
models["NM"] = lgb.LGBMRegressor(n_estimators=858, learning_rate=0.023, max_depth=5, num_leaves=124,
                                 colsample_bytree=0.576, min_child_samples=204, reg_alpha=2, reg_lambda=0.01,
                                 subsample=0.858, subsample_freq=1, n_jobs=-1, verbose=-1).fit(df[feats], df[TARGET])
print(f"NM {time.time()-t0:.0f}s", flush=True)
with open("final_models.pkl", "wb") as f:
    cloudpickle.dump(models, f)
