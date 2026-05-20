from __future__ import annotations

import json
from pathlib import Path

from .config import TEAM_ALIASES_JSON


def load_team_aliases(path: Path = TEAM_ALIASES_JSON) -> dict:
    if not path.exists():
        return {"teams": []}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_token(value: object) -> str:
    return "" if value is None else str(value).strip().lower().replace(" ", "")


def alias_lookup(path: Path = TEAM_ALIASES_JSON) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for team in load_team_aliases(path).get("teams", []):
        display = team.get("display_name") or team.get("short_name") or ""
        names = [display, team.get("short_name", ""), *team.get("aliases", [])]
        for name in names:
            key = normalize_token(name)
            if key:
                lookup[key] = display
    return lookup


def canonical_team_name(value: object, path: Path = TEAM_ALIASES_JSON) -> str:
    text = "" if value is None else str(value).strip()
    return alias_lookup(path).get(normalize_token(text), text)
