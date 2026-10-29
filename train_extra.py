"""Add the starter-config model and the 'risky' (unstable) feature list to final_models.pkl."""
import time

import cloudpickle
import lightgbm as lgb

from research import TARGET, load

t0 = time.time()
df, feats = load()
df = df[df[TARGET].notna()]
with open("final_models.pkl", "rb") as f:
    models = cloudpickle.load(f)

# the starter's config scored best of ours on the out-of-sample practice board
models["S"] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).fit(df[feats], df[TARGET])
print(f"S {time.time()-t0:.0f}s", flush=True)

# risky = features whose relationship with the target changed most between the two halves of train
expeds = sorted(df["exped"].unique())
early = df[df["exped"].isin(expeds[: len(expeds) // 2])]
late = df[df["exped"].isin(expeds[len(expeds) // 2:])]
shift = (early[feats].corrwith(early[TARGET]) - late[feats].corrwith(late[TARGET])).abs()
models["risky"] = list(shift.sort_values(ascending=False).index[:20])
print("risky:", models["risky"][:5], "...")

with open("final_models.pkl", "wb") as f:
    cloudpickle.dump(models, f)
print(f"saved {time.time()-t0:.0f}s")
