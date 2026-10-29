"""Launch the hosted batch on Everesteer compute, keeping the queue full within the credit hold.

Each job's id and spec are appended to hosted_jobs.json as it launches, so a restart resumes.
"""
import json
import os
import time

from eiq import client

W10, W6 = "exped_6111", "exped_5660"     # starts of the last-tenth and last-sixth windows
LGB = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "seed": 1}
AUX = ["target_Adrar", "target_Azgour", "target_Ighboula", "target_Tighanimine", "target_Imaradene",
       "target_Bounaceur", "target_Outka", "target_Moussa", "target_Akioud", "target_Amlal", "target_Tizal",
       "target_Tinzar", "target_Siroua", "target_Toubkal", "target_Tidiquin", "target_Tazaghart",
       "target_Afantazi", "target_Tasserirt", "target_Ghat", "target_Awlim", "target_Ouaoulzat",
       "target_Tissouka", "target_Anghomar", "target_Tichki", "target_Tignousti", "target_Timgarine",
       "target_Adad", "target_Tazoult", "target_Yagour", "target_everest"]
CATBOOST = ("def build_model(params):\n"
            "    from catboost import CatBoostRegressor\n"
            "    return CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, verbose=0)\n")

SPECS = [
    {"key": "mlp_w10", "model": "mlp", "gpu": "T4", "target": "target_everest", "start": W10},
    {"key": "cat_w10", "model": "custom", "custom_model_fn": CATBOOST, "gpu": "CPU", "target": "target_everest", "start": W10},
    {"key": "xgb_w6", "model": "xgboost", "gpu": "CPU", "target": "target_everest", "start": W6},
] + [{"key": f"lgb_{t[7:]}_{w}", "model": "lightgbm", "gpu": "CPU", "target": t, "start": s, "params": LGB}
     for w, s in (("w10", W10), ("w6", W6)) for t in AUX]

STATE = "hosted_jobs.json"
state = json.load(open(STATE)) if os.path.exists(STATE) else {}
state = {k: v for k, v in state.items() if isinstance(v, dict)}   # drop the old pilot entry

for spec in SPECS:
    if spec["key"] in state:
        continue
    kw = {k: spec[k] for k in ("model", "gpu", "target", "custom_model_fn", "params") if k in spec}
    while True:
        try:
            j = client.train(features="all", train_filter={"exped": {"cutoff_gte": spec["start"]}}, **kw)
            break
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "credit" in msg.lower() or "insufficient" in msg.lower() or "402" in msg:
                time.sleep(20)          # hold is full: wait for running jobs to settle
                continue
            print("FAILED", spec["key"], msg[:200], flush=True)
            j = {"id": None, "error": msg[:300]}
            break
    state[spec["key"]] = {**spec, "job_id": j.get("id"), "error": j.get("error")}
    with open(STATE, "w") as f:
        json.dump(state, f, indent=1)
    print(time.strftime("%H:%M:%S"), "launched", spec["key"], j.get("id"), flush=True)
print("all launched:", len(state))
