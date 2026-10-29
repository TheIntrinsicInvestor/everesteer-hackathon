"""Wait until a NEW round is taking predictions, then submit the lineup once and exit.

Usage: python round_watch.py round_1
"""
import subprocess
import sys
import time

from eiq import client
from submit_round import ROUND_LINEUP

want = sys.argv[1]
while True:
    cad = client.get_status()["cadence"]
    if cad.get("open_window") == want and not cad.get("intake_fenced"):
        break
    if cad.get("phase") == "done":
        sys.exit("event over")
    print(time.strftime("%H:%M:%S"), "phase", cad.get("phase"), "window", cad.get("open_window"),
          "next in", cad.get("seconds_until_next_phase"), flush=True)
    time.sleep(15)
print(time.strftime("%H:%M:%S"), f"{want} open: submitting", flush=True)
subprocess.run([sys.executable, "-u", "submit_round.py", "--only", ROUND_LINEUP], check=False)
