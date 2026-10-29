"""Recent-window candidate models: own out-of-time check, then final fit into final_models.pkl.

  python candidates.py check   # train on the window ending before the last 8% of train, score FIT there
  python candidates.py final   # train on the true last window, add to final_models.pkl

A candidate is a list of fitted estimators; submit_round rank-averages a list per exped.
"""
import os
import sys
import time

import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from research import TARGET, gauss_rank, load

WINDOW = 1 / 10        # share of expeds in the recent window (the practice-board sweet spot)
CHECK_TAIL = 0.08      # own check: score on the last 8% of train
EMBARGO = 20
STARTER = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
AUX = ["target_Azgour", "target_Ighboula", "target_Outka", "target_Moussa"]
AUXTOP = ["target_Tignousti", "target_Awlim", "target_Tasserirt", "target_Tinzar"]   # won 3/4 folds in explore.py


def fit_candidates(d, feats, only=None):
    X, y = d[feats], d[TARGET]
    c = {
        "C_lgb": lambda: [lgb.LGBMRegressor(**STARTER).fit(X, y)],
        "C_bag": lambda: [lgb.LGBMRegressor(**STARTER, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                            random_state=s).fit(X, y) for s in range(5)],
        "C_xgb": lambda: [xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                           colsample_bytree=0.8, tree_method="hist", n_jobs=-1).fit(X, y)],
        "C_cat": lambda: [CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, verbose=0,
                                            thread_count=-1).fit(X, y)],
        "C_ridge": lambda: [make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1000.0)).fit(X, y)],
        "C_aux": lambda: [lgb.LGBMRegressor(**STARTER).fit(X, y)] + [
            lgb.LGBMRegressor(**STARTER).fit(d.loc[d[a].notna(), feats], d.loc[d[a].notna(), a]) for a in AUX],
        "C_auxtop": lambda: [
            lgb.LGBMRegressor(**STARTER).fit(d.loc[d[a].notna(), feats], d.loc[d[a].notna(), a]) for a in AUXTOP],
    }
    out = {}
    for k, f in c.items():
        if only and k not in only:
            continue
        t0 = time.time()
        out[k] = f()
        print(f"  fitted {k} {time.time()-t0:.0f}s", flush=True)
    return out


def predict_list(models, x, era):
    return sum(pd.Series(m.predict(x), index=x.index).groupby(era).rank(pct=True) for m in models) / len(models)


def fit_by_exped(df, col):
    fits = []
    for _, g in df.groupby("exped"):
        t = g[TARGET] - g[TARGET].mean()
        fits.append(np.mean(gauss_rank(g[col]).to_numpy() * t.to_numpy()))
    return np.mean(fits), np.mean(fits) / np.std(fits)


def main():
    mode = sys.argv[1]
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    df, feats = load()
    df = df[df[TARGET].notna()]
    expeds = sorted(df["exped"].unique())
    n_win = int(len(expeds) * WINDOW)
    if mode == "check":
        n_tail = int(len(expeds) * CHECK_TAIL)
        test_e = expeds[-n_tail:]
        win_e = expeds[-(n_tail + EMBARGO + n_win):-(n_tail + EMBARGO)]
        tr, te = df[df["exped"].isin(win_e)], df[df["exped"].isin(test_e)].copy()
        print(f"check: train {len(win_e)} recent expeds ({len(tr):,} rows), test last {n_tail} ({len(te):,})")
        cands = fit_candidates(tr, feats, only)
        # reference: same model on the full history before the cutoff (practice board said recency wins)
        full = df[df["exped"].isin(expeds[: -(n_tail + EMBARGO)])]
        cands["REF_full"] = [lgb.LGBMRegressor(**STARTER).fit(full[feats], full[TARGET])]
        print(f"{'model':10} {'FIT':>8} {'sharpe':>7}")
        for k, ms in cands.items():
            te[k] = predict_list(ms, te[feats], te["exped"])
            f, s = fit_by_exped(te, k)
            print(f"{k:10} {f:8.5f} {s:7.3f}", flush=True)
        # blends (C_lgb stands in for the R10/R12 recent LightGBMs)
        blends = {"EQ_aux_lgb": {"C_aux": 1, "C_lgb": 1},
                  "EQ_aux_lgb_ridge": {"C_aux": 1, "C_lgb": 1, "C_ridge": 1},
                  "EQ_aux_lgb_full": {"C_aux": 1, "C_lgb": 1, "REF_full": 1},
                  "OPT": {"C_aux": 0.068, "C_lgb": 0.100, "C_cat": 0.036, "C_xgb": 0.012, "C_ridge": 0.006}}
        for b, w in blends.items():
            if all(k in te for k in w):
                te[b] = sum(wt * te[k] for k, wt in w.items()) / sum(w.values())
                f, s = fit_by_exped(te, b)
                print(f"{b:16} {f:8.5f} {s:7.3f}", flush=True)
    else:
        d = df[df["exped"].isin(expeds[-n_win:])]
        # C_aux on other window lengths (explore.py: aux4_mw won 3/4 folds, aux4_w6 and aux4_w4 won 3/3+)
        win_cands = {"C_aux_mw": (1 / 12, 1 / 10, 1 / 8), "C_aux_w6": (1 / 6,), "C_aux_w4": (1 / 4,)}
        rest = [k for k in only if k not in win_cands] if only else None
        cands = fit_candidates(d, feats, rest) if rest != [] else {}   # [] would mean "fit everything"
        for k, fracs in win_cands.items():
            if only and k in only:
                cands[k] = [e for f in fracs for e in fit_candidates(
                    df[df["exped"].isin(expeds[-int(len(expeds) * f):])], feats, ["C_aux"])["C_aux"]]
        with open("final_models.pkl", "rb") as f:
            m = cloudpickle.load(f)
        m.update(cands)
        # atomic swap: round_watch may be reading final_models.pkl at any moment
        with open("final_models.tmp", "wb") as f:
            cloudpickle.dump(m, f)
        os.replace("final_models.tmp", "final_models.pkl")
        print("saved", list(cands))


if __name__ == "__main__":
    main()
