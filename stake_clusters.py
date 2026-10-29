"""Stake equally per correlation cluster with a small live-score tilt (Brian's rule from round 3).

Each cluster gets TOTAL / n_clusters. Clusters are ranked by their members' mean live round score,
shrunk 50% toward 0 (a model with no live score counts as 0): the best gets +$1, the worst -$1.
Within a cluster the stake is split equally over its best two members (the $1 minimum rules out more).
Clusters come from per-period correlation of round 2 live predictions (members correlate 0.8 to 0.95).

  python stake_clusters.py round_1,round_2           # show the plan
  python stake_clusters.py round_1,round_2 --apply   # withdraw other drafts in the open window, stake the plan
"""
import json
import sys

from eiq import client

TOTAL = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--total=")), 15.0)
TILT = 1.0
CLUSTERS = {"eq1": ["bl-eq-aux-r12", "bl-recent12", "bl-cross-blend", "bl-c-bag", "bl-c-cat"],
            "long": ["bl-recent16", "bl-recent-wt"],
            "auxtop": ["bl-c-auxtop", "bl-aux4-top"],
            "mw": ["bl-aux-mw"],
            "longwin": ["bl-aux-w6", "bl-aux-w4"]}   # 1/6 and 1/4 windows, added round 3 as zero-stake trials
for a in sys.argv:   # --exclude=longwin leaves a cluster unstaked (e.g. zero-stake trials)
    if a.startswith("--exclude="):
        for c in a.split("=")[1].split(","):
            CLUSTERS.pop(c)

ids = json.load(open("model_ids.json"))
rev = {v: k for k, v in ids.items()}
name_to_id = {m["name"]: m["id"] for m in client.get_models()["models"]}
scores = {}
for w in sys.argv[1].split(","):
    mine = {rev.get(name_to_id.get(e["model_name"])): e["round_score"]
            for e in client.get_diagnostics_leaderboard(limit=500, scoring_window=w)["entries"] if e["is_self"]}
    med = sorted(mine.values())[len(mine) // 2]
    for n, s in mine.items():   # relative to my median that round, so models seen in different rounds compare fairly
        scores.setdefault(n, []).append(s - med)
live = {n: 0.5 * sum(v) / len(v) for n, v in scores.items() if n}

cl_score = {c: sum(live.get(n, 0.0) for n in ms) / len(ms) for c, ms in CLUSTERS.items()}
# only clusters with a live score can be tilted: an untested cluster must not win a tie at 0
order = sorted([c for c, ms in CLUSTERS.items() if any(n in live for n in ms)], key=cl_score.get)
base = TOTAL / len(CLUSTERS)
plan = {}
for c, ms in CLUSTERS.items():
    amt = base + (TILT if len(order) > 1 and c == order[-1] else -TILT if len(order) > 1 and c == order[0] else 0.0)
    best = sorted(ms, key=lambda n: live.get(n, 0.0), reverse=True)[:2]
    for n in best:
        plan[n] = int(amt / len(best) * 100) / 100   # round down so the total never exceeds TOTAL
    print(f"{c:7} shrunk score {cl_score[c]:+.4f} -> ${amt:.2f} on {best}")
print("plan", plan, "total", round(sum(plan.values()), 2))

if "--apply" in sys.argv:
    window = client.get_event_staking()["draft_window"]
    cur = [x for x in client.get_event_staking()["windows"] if x["window"] == window][0]["allocations"]
    for a in cur:
        if rev.get(a["model_id"]) not in plan:
            client.withdraw_stake_allocation(a["model_id"], window=window)
            print("withdrew", rev.get(a["model_id"]))
    for n, amt in plan.items():
        client.set_stake_allocation(ids[n], amount_usdc=str(amt), window=window)
    cur = [x for x in client.get_event_staking()["windows"] if x["window"] == window][0]
    print(window, {rev.get(a["model_id"]): a["amount_micro"] / 1e6 for a in cur["allocations"]},
          "total", cur["allocated_micro"] / 1e6)
