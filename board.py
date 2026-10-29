"""Print the practice/round board for my models, labelled with my recipe names."""
import json
import sys

from eiq import client as c

ids = json.load(open("model_ids.json"))
rev = {v: k for k, v in ids.items()}
name_to_id = {m["name"]: m["id"] for m in c.get_models()["models"]}
kw = {"scoring_window": sys.argv[1]} if len(sys.argv) > 1 else {}
lb = c.get_diagnostics_leaderboard(limit=300, **kw)
print("entries:", lb["total"], "| top 5:", [(e["agent_name"], round(e["round_score"], 3)) for e in lb["entries"][:5]])
for e in lb["entries"]:
    if e["is_self"]:
        label = rev.get(name_to_id.get(e["model_name"]), e["model_name"])
        print(f"{e['rank']:>4} {label:28} FIT {e['fit20']:.4f} UNQ {e['unq']:.4f} INOV {e['inov'] or 0:.4f} score {e['round_score']:.4f}")
