"""Multi-period test: each idea is fitted on the recent window before 4 separate test blocks
at the end of train, and scored (per-exped FIT) on each block. Also saves per-exped scores
and period-level stats for the staking analysis.

  python explore.py               # all ideas
  python explore.py base,aux8     # a subset
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from research import TARGET, gauss_rank, load

WINDOW = 1 / 10
BLOCK = 0.05          # each test block = 5% of expeds
N_FOLDS = 4
EMBARGO = 20
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
P_REG = dict(P, num_leaves=15, min_child_samples=300, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
             reg_lambda=10.0)
P_REG2 = dict(P, n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=1000, colsample_bytree=0.3,
              subsample=0.7, subsample_freq=1, reg_lambda=20.0)
WIN_FRAC = {"w6": 1 / 6, "w4": 1 / 4, "w2": 1 / 2, "full": 1.0}
AUX4 = ["target_Azgour", "target_Ighboula", "target_Outka", "target_Moussa"]
AUXTOP = ["target_Tignousti", "target_Awlim", "target_Tasserirt", "target_Tinzar"]
AUX8 = ["target_Adrar", "target_Azgour", "target_Ighboula", "target_Tighanimine", "target_Imaradene",
        "target_Bounaceur", "target_Outka", "target_Moussa"]


def rank_target(d, col):
    r = d.groupby("exped")[col].rank(pct=True)
    n = d.groupby("exped")[col].transform("count")
    return pd.Series(norm.ppf((r * n - 0.5) / n), index=d.index)


def add_feats(d, feats):
    """Within-period engineered features: missing count and feature minus its period mean."""
    out = d[feats].copy()
    out["n_missing"] = d[feats].isna().sum(axis=1)
    dm = d[feats] - d.groupby("exped")[feats].transform("mean")
    dm.columns = [c + "_dm" for c in feats]
    return pd.concat([out, dm], axis=1)


def rank_blend(preds, era):
    return sum(pd.Series(p, index=era.index).groupby(era).rank(pct=True) for p in preds) / len(preds)


def fit_predict(idea, tr, te, feats, window=None):
    era = te["exped"]
    one = lambda X, y, Xt, **kw: lgb.LGBMRegressor(**P).fit(X, y, **kw).predict(Xt)
    aux = lambda targets, Xtr=tr[feats], Xte=te[feats]: [one(Xtr[tr[a].notna()], tr.loc[tr[a].notna(), a], Xte) for a in targets]
    aux4_preds = lambda d, fs, p=P: [lgb.LGBMRegressor(**p).fit(d.loc[d[t].notna(), fs], d.loc[d[t].notna(), t]).predict(te[fs])
                                     for t in [TARGET] + AUX4]
    if idea == "base":
        return one(tr[feats], tr[TARGET], te[feats])
    if idea == "aux4":       # current best
        return rank_blend([one(tr[feats], tr[TARGET], te[feats])] + aux(AUX4), era)
    if idea == "aux4_reg":   # aux4 with heavier regularisation (the recent window is small)
        return rank_blend(aux4_preds(tr, feats, P_REG), era)
    if idea == "aux4_mw":    # aux4 on three window lengths, equal-weight rank blend
        return rank_blend([p for f in (1 / 12, 1 / 10, 1 / 8) for p in aux4_preds(window(f), feats)], era)
    if idea.startswith("aux4_") and idea[5:] in WIN_FRAC:   # longer windows (round 1: 1/16 > 1/12 > 1/10)
        return rank_blend(aux4_preds(window(WIN_FRAC[idea[5:]]), feats), era)
    if idea.startswith("mix:"):  # equal rank blend of ideas already scored this fold, e.g. mix:aux4+aux4_w2
        return rank_blend([te[i].to_numpy() for i in idea[4:].split("+")], era)
    if idea == "aux4_reg2":  # much heavier regularisation
        return rank_blend(aux4_preds(tr, feats, P_REG2), era)
    if idea == "aux4_sim":   # periods from the last half whose mean features sit closest to the test block's
        hist = window(1 / 2)
        mu = hist.groupby("exped")[feats].mean()
        z = (mu - mu.mean()) / mu.std().replace(0, 1)
        target_mu = (te[feats].mean() - mu.mean()) / mu.std().replace(0, 1)
        dist = ((z - target_mu) ** 2).mean(axis=1)
        near = dist.nsmallest(tr["exped"].nunique()).index
        return rank_blend(aux4_preds(hist[hist["exped"].isin(near)], feats), era)
    if idea == "aux4_stable":  # aux4 without the 25% of features least stable across periods
        c = tr.groupby("exped").apply(lambda g: g[feats].corrwith(g[TARGET]))
        stab = (c.mean().abs() / c.std()).fillna(0)
        keep = list(stab[stab >= stab.quantile(0.25)].index)
        return rank_blend(aux4_preds(tr, keep), era)
    if idea == "auxtop":     # the targets that scored best in Everesteer's hosted CV
        return rank_blend(aux(AUXTOP), era)
    if idea == "aux4_top":
        return rank_blend([one(tr[feats], tr[TARGET], te[feats])] + aux(AUX4) + aux(AUXTOP), era)
    if idea == "aux8":
        return rank_blend([one(tr[feats], tr[TARGET], te[feats])] + aux(AUX8), era)
    if idea == "aux8w":      # aux8, weighting each target by its correlation with the graded one
        w = [1.0] + [tr[[a, TARGET]].corr().iloc[0, 1] for a in AUX8]
        preds = [one(tr[feats], tr[TARGET], te[feats])] + aux(AUX8)
        ranks = [pd.Series(p, index=era.index).groupby(era).rank(pct=True) for p in preds]
        return sum(wi * r for wi, r in zip(w, ranks)) / sum(w)
    if idea == "trank":      # target ranked within each period (matches the rank-based scoring)
        return one(tr[feats], rank_target(tr, TARGET), te[feats])
    if idea == "trank_aux4":
        return rank_blend([one(tr[feats], rank_target(tr, TARGET), te[feats])] +
                          [one(tr[feats][tr[a].notna()], rank_target(tr[tr[a].notna()], a), te[feats]) for a in AUX4], era)
    if idea == "feng":       # within-period engineered features
        return one(add_feats(tr, feats), tr[TARGET], add_feats(te, feats))
    if idea == "eraboost":   # refit with extra weight on the periods the first fit got wrong
        m = lgb.LGBMRegressor(**P).fit(tr[feats], tr[TARGET])
        p = pd.Series(m.predict(tr[feats]), index=tr.index)
        tc = tr[TARGET] - tr.groupby("exped")[TARGET].transform("mean")
        fit_e = (p.groupby(tr["exped"]).rank(pct=True) * tc).groupby(tr["exped"]).mean()
        bad = fit_e[fit_e < fit_e.median()].index
        w = np.where(tr["exped"].isin(bad), 2.0, 1.0)
        return one(tr[feats], tr[TARGET], te[feats], sample_weight=w)
    raise ValueError(idea)


def per_exped_fit(te, col):
    out = {}
    for e, g in te.groupby("exped"):
        t = g[TARGET] - g[TARGET].mean()
        out[e] = np.mean(gauss_rank(g[col]).to_numpy() * t.to_numpy())
    return pd.Series(out)


IDEAS = ["base", "aux4", "aux8", "aux8w", "trank", "trank_aux4", "feng", "eraboost"]


def main():
    ideas = sys.argv[1].split(",") if len(sys.argv) > 1 else IDEAS
    df, feats = load()
    df = df[df[TARGET].notna()]
    expeds = sorted(df["exped"].unique())
    n_win, n_blk = int(len(expeds) * WINDOW), int(len(expeds) * BLOCK)
    rows, per_exped = [], []
    for k in range(N_FOLDS):
        end = len(expeds) - k * n_blk
        test_e = expeds[end - n_blk:end]
        win_e = expeds[end - n_blk - EMBARGO - n_win:end - n_blk - EMBARGO]
        tr, te = df[df["exped"].isin(win_e)], df[df["exped"].isin(test_e)].copy()
        cut = end - n_blk - EMBARGO
        window = lambda f, cut=cut: df[df["exped"].isin(expeds[max(0, cut - int(len(expeds) * f)):cut])]
        for idea in ideas:
            t0 = time.time()
            te[idea] = np.asarray(fit_predict(idea, tr, te, feats, window))
            s = per_exped_fit(te, idea)
            rows.append({"fold": k, "idea": idea, "fit": s.mean(), "sharpe": s.mean() / s.std()})
            per_exped.append(s.rename(idea).to_frame().assign(fold=k))
            print(f"fold {k} {idea:11} FIT {s.mean():8.5f}  ({time.time()-t0:.0f}s)", flush=True)
    r = pd.DataFrame(rows)
    piv = r.pivot(index="idea", columns="fold", values="fit")
    if "aux4" in piv.index:
        piv["wins_vs_aux4"] = (piv[list(range(N_FOLDS))].gt(piv.loc["aux4", list(range(N_FOLDS))], axis=1)).sum(axis=1)
    piv["mean"] = r.groupby("idea")["fit"].mean()
    print("\n" + piv.sort_values("mean", ascending=False).round(5).to_string())
    pd.concat(per_exped).to_csv(sys.argv[2] if len(sys.argv) > 2 else "explore_per_exped.csv")


if __name__ == "__main__":
    main()
