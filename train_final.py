"""Fit the final models on ALL of train and save them for submit_round.py."""
import time

import cloudpickle
import lightgbm as lgb

from research import LGB_PARAMS, TARGET, load, residual_target

t0 = time.time()
df, feats = load()
df = df[df[TARGET].notna()]
models = {"feats": feats}
models["A"] = lgb.LGBMRegressor(**LGB_PARAMS).fit(df[feats], df[TARGET]); print(f"A {time.time()-t0:.0f}s", flush=True)
models["REP"] = lgb.LGBMRegressor(**LGB_PARAMS).fit(df[feats], df["v1_sherpa"]); print(f"REP {time.time()-t0:.0f}s", flush=True)
models["B"] = lgb.LGBMRegressor(**LGB_PARAMS).fit(df[feats], residual_target(df)); print(f"B {time.time()-t0:.0f}s", flush=True)
for aux in ["target_Azgour", "target_Ighboula", "target_Outka", "target_Moussa"]:
    d = df[df[aux].notna()]
    models[aux] = lgb.LGBMRegressor(**LGB_PARAMS).fit(d[feats], d[aux]); print(f"{aux} {time.time()-t0:.0f}s", flush=True)
with open("final_models.pkl", "wb") as f:
    cloudpickle.dump(models, f)
print(f"saved final_models.pkl {time.time()-t0:.0f}s")
