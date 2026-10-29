"""Show the staking position, or draft stakes for the open window.

Usage:  python stake.py                         # position only
        python stake.py bl-starter=1 bl-aux=0.5 # draft USDC amounts for the open round
"""
import json
import sys

from eiq import client

with open("model_ids.json") as f:
    ids = json.load(f)

st = client.get_event_staking()
window = st.get("draft_window")
print(json.dumps({k: st.get(k) for k in ("phase", "draft_window", "stake_window_open", "balance_micro",
                                          "principal_micro", "settled_micro", "net_micro", "min_stake_micro",
                                          "max_slots")}, indent=1))

for arg in sys.argv[1:]:
    name, amount = arg.split("=")
    if not window:
        sys.exit("No draft window open: nothing staked.")
    r = client.set_stake_allocation(ids[name], amount_usdc=str(amount), window=window)
    print(f"{name:14} {amount:>6} USDC -> {r.get('status', r)}")
