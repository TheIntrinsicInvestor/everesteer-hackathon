"""Shared client. Uses EIQ_API_KEY if set, else reads it from a profile outside the repo; never store it here."""
import os
import re

from everestapi import EverestAPI

PROFILE = os.path.expanduser(os.path.join("~", ".config", "everesteer", "ps_profile.ps1"))
BASE_URL = "https://hackathon.everesteer.ai"

if not os.environ.get("EIQ_API_KEY"):
    with open(PROFILE, encoding="utf-8-sig") as f:
        m = re.search(r"\$env:EIQ_API_KEY\s*=\s*'([^']+)'", f.read())
    os.environ["EIQ_API_KEY"] = m.group(1)
os.environ.setdefault("EIQ_BASE_URL", BASE_URL)

client = EverestAPI(api_key=os.environ["EIQ_API_KEY"], base_url=os.environ["EIQ_BASE_URL"])
