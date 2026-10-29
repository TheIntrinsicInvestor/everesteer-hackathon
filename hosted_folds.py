"""Hosted forward test: train MLP and tree ideas on Everesteer compute over explore.py's 4 fold windows
(plus the true last window for the live fit), then score them locally on explore.py's test blocks.

  python hosted_folds.py launch    # launch every job, throttled by the credit hold (resumable)
  python hosted_folds.py collect   # download finished models, keep the fitted estimators
  python hosted_folds.py eval      # fold table vs aux4 (aux4 scores from explore_per_exped_auxtop.csv)

Delivered .pkls are 3.11 bytecode, which breaks when called under 3.12, so only the fitted estimator,
its feature list and fill value are kept (they sit in the wrapper's default arguments).
"""
import json
import os
import sys
import time

import cloudpickle
import pandas as pd

from eiq import client
from explore import AUX4, AUXTOP, BLOCK, EMBARGO, N_FOLDS, WINDOW, per_exped_fit, rank_blend
from research import DATA, TARGET

STATE, MODELS, PKL_DIR = "hosted_folds.json", "hosted_models.pkl", "hosted_pkls"
CATBOOST = ("def build_model(params):\n"
            "    from catboost import CatBoostRegressor\n"
            "    return CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, verbose=0)\n")
IDEAS = {
    "mlpaux": [("mlp", t, 1) for t in [TARGET] + AUX4 + AUXTOP],
    "mlpbag": [("mlp", TARGET, s) for s in range(1, 6)],
    "treetop": [("xgboost", t, 1) for t in AUXTOP] + [("catboost", t, 1) for t in AUXTOP],
}
JOBS = sorted({j for js in IDEAS.values() for j in js})


def expeds():
    d = pd.read_parquet(DATA, columns=["exped", TARGET])
    return sorted(d.loc[d[TARGET].notna(), "exped"].unique())


def windows(e):
    """Same windows as explore.py: fold k trains on n_win expeds ending EMBARGO before its test block."""
    n_win, n_blk = int(len(e) * WINDOW), int(len(e) * BLOCK)
    out = {}
    for k in range(N_FOLDS):
        cut = len(e) - k * n_blk - n_blk - EMBARGO
        out[f"f{k}"] = {"cutoff_gte": e[cut - n_win], "cutoff_lt": e[cut]}
    out["final"] = {"cutoff_gte": e[-n_win]}
    return out


def key(w, job):
    return "|".join([w, job[0], job[1], str(job[2])])


def launch():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    for w, filt in windows(expeds()).items():
        for model, target, seed in JOBS:
            k = key(w, (model, target, seed))
            if k in state:
                continue
            kw = dict(features="all", target=target, gpu="CPU", cv={"folds": 2}, train_filter={"exped": filt})
            if model == "catboost":
                kw.update(model="custom", custom_model_fn=CATBOOST)
            else:
                kw.update(model=model, params={"random_state": seed})
            while True:
                try:
                    j = client.train(**kw)
                    break
                except Exception as ex:  # noqa: BLE001
                    msg = str(ex)
                    if "credit" in msg.lower() or "insufficient" in msg.lower() or "402" in msg:
                        time.sleep(15)          # hold is full: wait for running jobs to settle
                        continue
                    print("FAILED", k, msg[:200], flush=True)
                    j = {"id": None, "error": msg[:300]}
                    break
            state[k] = {"job_id": j.get("id"), "error": j.get("error")}
            with open(STATE, "w") as f:
                json.dump(state, f, indent=1)
            print(time.strftime("%H:%M:%S"), "launched", k, j.get("id"), flush=True)
    print("all launched:", len(state))


def collect():
    state = json.load(open(STATE))
    models = cloudpickle.load(open(MODELS, "rb")) if os.path.exists(MODELS) else {}
    os.makedirs(PKL_DIR, exist_ok=True)
    pending = 0
    for k, v in state.items():
        if k in models or not v.get("job_id"):
            continue
        st = client.get_job_status(v["job_id"])
        if st.get("status") != "completed":
            pending += st.get("status") not in ("failed", "cancelled")
            if st.get("status") in ("failed", "cancelled"):
                print("job", st.get("status"), k, (st.get("error_message") or "")[:150])
            continue
        path = client.download_model(v["job_id"], os.path.join(PKL_DIR, v["job_id"] + ".pkl"))
        d = cloudpickle.load(open(path, "rb")).__defaults__
        est = next(x for x in d if hasattr(x, "predict"))
        feats = next(x for x in d if isinstance(x, list))
        fill = next((x for x in d if isinstance(x, float)), 0.0)
        models[k] = (est, feats, fill)
    with open(MODELS + ".tmp", "wb") as f:
        cloudpickle.dump(models, f)
    os.replace(MODELS + ".tmp", MODELS)
    print(f"collected {len(models)} of {len(state)}; still running {pending}")


def predict(m, x):
    est, feats, fill = m
    return est.predict(x.reindex(columns=feats).fillna(fill).astype("float32").to_numpy())


def evaluate():
    models = cloudpickle.load(open(MODELS, "rb"))
    raw = pd.read_parquet(DATA)
    raw = raw[raw[TARGET].notna()]
    e = sorted(raw["exped"].unique())
    n_blk = int(len(e) * BLOCK)
    aux4 = pd.read_csv("explore_per_exped_auxtop.csv", index_col=0).groupby("fold")["aux4"].mean()
    rows = []
    for k in range(N_FOLDS):
        end = len(e) - k * n_blk
        te = raw[raw["exped"].isin(e[end - n_blk:end])].copy()
        rows.append({"fold": k, "idea": "aux4", "fit": aux4[k]})
        for idea, jobs in IDEAS.items():
            ks = [key(f"f{k}", j) for j in jobs]
            if not all(x in models for x in ks):
                print(f"fold {k} {idea}: {sum(x in models for x in ks)}/{len(ks)} models, skipped")
                continue
            te[idea] = rank_blend([predict(models[x], te) for x in ks], te["exped"]).to_numpy()
            rows.append({"fold": k, "idea": idea, "fit": per_exped_fit(te, idea).mean()})
    piv = pd.DataFrame(rows).pivot(index="idea", columns="fold", values="fit")
    folds = [c for c in range(N_FOLDS) if c in piv]
    piv["wins_vs_aux4"] = piv[folds].gt(piv.loc["aux4", folds], axis=1).sum(axis=1)
    piv["mean"] = piv[folds].mean(axis=1)
    print(piv.sort_values("mean", ascending=False).round(5).to_string())


if __name__ == "__main__":
    {"launch": launch, "collect": collect, "eval": evaluate}[sys.argv[1]]()
