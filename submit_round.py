"""Predict the open round with every recipe and submit each as its own model.

Usage:  python submit_round.py            # open round only (event lane)
        python submit_round.py --practice # practice board, only when no round is open
"""
import json
import sys

import cloudpickle
import numpy as np
import pandas as pd
from scipy.stats import norm

from eiq import client

# name on the board -> recipe. Edit after research.py.
ROUND_LINEUP = ("bl-eq-aux-r12,bl-c-aux,bl-cross-blend,bl-recent10,bl-recent12,bl-c-bag,bl-c-cat,"
                "bl-recent10-nm,bl-recent-ens,bl-recent16,bl-recent-wt,bl-c-auxtop,bl-aux4-top,bl-aux-mw,"
                "bl-aux-w6,bl-aux-w4")
RECIPES = {"bl-starter": "S", "bl-aux": "D", "bl-starter-fn": "SN", "bl-aux-fn": "DN", "bl-blend": "SD",
           "bl-recent33": "R33", "bl-recent16": "R16", "bl-numerai": "NM",
           "bl-recent10": "R10", "bl-recent20": "R20", "bl-recent-wt": "RW",
           "bl-recent8": "R8", "bl-recent12": "R12", "bl-recent10-nm": "R10N", "bl-recent-ens": "RE",
           "bl-c-bag": "C_bag", "bl-c-xgb": "C_xgb", "bl-c-cat": "C_cat", "bl-c-ridge": "C_ridge",
           "bl-c-aux": "C_aux", "bl-cross-blend": "RX",
           "bl-eq-aux-r12": "BL:C_aux=1,R12=1",
           "bl-eq-aux-r12-ridge": "BL:C_aux=1,R12=1,C_ridge=1",
           "bl-eq-aux-r12-rw": "BL:C_aux=1,R12=1,RW=1",
           # explore.py forward test: auxtop won 3/4 folds vs aux4, aux4_top 4/4 (5:4 = equal weight per estimator)
           "bl-c-auxtop": "C_auxtop",
           "bl-aux4-top": "BL:C_aux=5,C_auxtop=4",
           "bl-aux-mw": "C_aux_mw",   # aux4 on 1/12, 1/10, 1/8 windows: won 3/4 folds vs aux4
           "bl-aux-w6": "C_aux_w6",   # aux4 on the last 1/6: won folds 0-2 vs aux4
           "bl-aux-w4": "C_aux_w4",   # aux4 on the last 1/4: won folds 0-2 vs aux4
           # weights = practice score minus 0.13 (the closest feasible 'optimised' weights)
           "bl-opt": "BL:C_aux=0.068,R12=0.050,R10=0.050,C_bag=0.041,C_cat=0.036,C_xgb=0.012,C_ridge=0.006"}
# the platform renames models, so pin them by id (new ones are created and recorded on first use)
IDS_FILE = "model_ids.json"
if "--only" in sys.argv:
    RECIPES = {k: v for k, v in RECIPES.items() if k in sys.argv[sys.argv.index("--only") + 1].split(",")}
PROPORTION = 0.5   # share of the risky-feature exposure SN/DN remove (C, unused now, also reads it)
TARGET = "target_everest"
PYVER = f"{sys.version_info.major}.{sys.version_info.minor}"


def build_predict(m, recipe, proportion):
    """Self-contained callable (cloudpickled by value): no imports from this repo."""
    feats = m["feats"]

    def predict(live_features, live_benchmark_models=None):
        x = live_features.reindex(columns=feats).astype("float32")
        x = x.mask(x == -1)
        if recipe in ("SN", "DN", "SD"):
            ranks = {}
            for k in ("S", "D"):
                ranks[k] = build_predict(m, k, proportion)(live_features)["prediction"].groupby(live_features["exped"]).rank(pct=True)
            p = (ranks["S"] + ranks["D"]) / 2 if recipe == "SD" else ranks[recipe[0]]
            if recipe != "SD":
                # partial neutralization per exped against the unstable features
                risky = x[m["risky"]].fillna(x[m["risky"]].mean())
                out = pd.Series(index=live_features.index, dtype="float64")
                for _, idx in live_features.groupby("exped").groups.items():
                    a = norm.ppf((p[idx].rank() - 0.5) / len(idx))
                    X = np.column_stack([risky.loc[idx].fillna(0).to_numpy(), np.ones(len(idx))])
                    out[idx] = a - proportion * (X @ np.linalg.lstsq(X, a, rcond=None)[0])
                p = out
            p = np.asarray(p, dtype="float64")
        elif recipe == "RE":
            # rank-average of the recent-slice models around the sweet spot
            keys = ["R8", "R10", "R12", "R16", "R10N"]
            era = live_features["exped"] if "exped" in live_features else live_features.get("era", pd.Series(0, index=live_features.index))
            p = sum(pd.Series(m[k].predict(x), index=live_features.index).groupby(era).rank(pct=True)
                    for k in keys).to_numpy() / len(keys)
        elif recipe.startswith("BL:"):
            # weighted rank blend, e.g. "BL:C_aux=1,R12=1"; list components are averaged first
            era = live_features["exped"] if "exped" in live_features else live_features.get("era", pd.Series(0, index=live_features.index))
            total, wsum = 0, 0.0
            for part in recipe[3:].split(","):
                k, w = part.split("=")
                ests = m[k] if isinstance(m[k], list) else [m[k]]
                r = sum(pd.Series(e.predict(x), index=live_features.index).groupby(era).rank(pct=True)
                        for e in ests) / len(ests)
                total, wsum = total + float(w) * r, wsum + float(w)
            p = (total / wsum).to_numpy()
        elif recipe == "RX":
            # equal-weight blend across windows and model families (drift research: diversity over picking)
            era = live_features["exped"] if "exped" in live_features else live_features.get("era", pd.Series(0, index=live_features.index))
            parts = []
            for k in ["R10", "R16", "RW", "C_bag", "C_xgb", "C_aux"]:
                ests = m[k] if isinstance(m[k], list) else [m[k]]
                parts.append(sum(pd.Series(e.predict(x), index=live_features.index).groupby(era).rank(pct=True)
                                 for e in ests) / len(ests))
            p = (sum(parts) / len(parts)).to_numpy()
        elif isinstance(m.get(recipe), list):
            # candidates.py stores a list of estimators: rank-average them per exped
            era = live_features["exped"] if "exped" in live_features else live_features.get("era", pd.Series(0, index=live_features.index))
            p = (sum(pd.Series(e.predict(x), index=live_features.index).groupby(era).rank(pct=True)
                     for e in m[recipe]) / len(m[recipe])).to_numpy()
        elif recipe in m:
            p = m[recipe].predict(x)
        elif recipe == "A":
            p = m["A"].predict(x)
        elif recipe == "B":
            p = m["B"].predict(x)
        elif recipe == "D":
            # equal-weight blend of the graded-target model and the aux-target models, by rank
            keys = ["A"] + [k for k in m if k.startswith("target_")]
            p = sum(pd.Series(m[k].predict(x), index=live_features.index).groupby(live_features["exped"]).rank(pct=True)
                    for k in keys).to_numpy() / len(keys)
        else:
            raw = pd.Series(m["A"].predict(x), index=live_features.index)
            rep = pd.Series(m["REP"].predict(x), index=live_features.index)
            p = pd.Series(index=live_features.index, dtype="float64")
            for _, idx in live_features.groupby("exped").groups.items():
                a = norm.ppf((raw[idx].rank() - 0.5) / len(idx))
                r = norm.ppf((rep[idx].rank() - 0.5) / len(idx))
                X = np.column_stack([r, np.ones(len(r))])
                p[idx] = a - proportion * (X @ np.linalg.lstsq(X, a, rcond=None)[0])
            p = p.to_numpy()
        return pd.DataFrame({"prediction": p}, index=live_features.index)

    return predict


def main():
    practice = "--practice" in sys.argv
    cad = client.get_started()["cadence"]
    window = cad.get("open_window")
    if practice:
        split = "validation"   # the practice board is not fenced, so it runs during a round too
    else:
        if not window or cad.get("intake_fenced"):
            sys.exit(f"No round taking predictions (phase={cad.get('phase')}, window={window}, "
                     f"fenced={cad.get('intake_fenced')}, next in {cad.get('seconds_until_next_phase')}s).")
        split = "live"

    live = pd.read_parquet(client.download_dataset(split=split))
    print(f"{split}: {len(live):,} rows, window={window}")
    with open("final_models.pkl", "rb") as f:
        m = cloudpickle.load(f)

    try:
        with open(IDS_FILE) as f:
            ids = json.load(f)
    except FileNotFoundError:
        ids = {}
    for name, recipe in RECIPES.items():
        fn = build_predict(m, recipe, PROPORTION)
        preds = fn(live)
        assert preds["prediction"].notna().all() and preds.index.is_unique
        out = preds.reset_index().rename(columns={preds.index.name or "index": "id"})
        pkl = f"pkl_{name}.pkl"
        with open(pkl, "wb") as f:
            cloudpickle.dump(fn, f)
        if name not in ids:
            ids[name] = client.create_model(name=name)["id"]
            with open(IDS_FILE, "w") as f:
                json.dump(ids, f, indent=1)
        model_id = ids[name]

        now = client.get_started()["cadence"]   # guard: rows still submittable?
        if not practice and (now.get("open_window") != window or now.get("intake_fenced")):
            sys.exit(f"Round moved on ({now.get('open_window')}) while predicting: re-run.")
        submit = client.submit_validation_diagnostics if practice else client.submit_event_predictions
        r = submit(model_id, out, target=TARGET, model_pkl=pkl, model_pkl_python_version=PYVER)
        print(f"{name:12} {recipe:2} -> {r.get('status')} upload={r.get('upload_id')} left={r.get('uploads_remaining')}")


if __name__ == "__main__":
    main()
