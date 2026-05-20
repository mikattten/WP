from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PBP_COLUMNS = [
    "game_id",
    "event_id",
    "period",
    "clock",
    "home_score",
    "away_score",
    "possession_team",
    "home_team",
    "away_team",
    "event_team",
    "event_type",
    "description",
]


def free_throw_key(row: pd.Series) -> tuple:
    description = str(row.get("description", ""))
    player = re.split(r"フリースロー|free throw", description, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return row.get("game_id"), row.get("period"), row.get("clock"), row.get("event_team"), player


def collapse_free_throw_trips(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.sort_values(["game_id", "event_id"], kind="stable").copy()
    pbp["_prev_home_score"] = pbp.groupby("game_id")["home_score"].shift(1).fillna(0)
    pbp["_prev_away_score"] = pbp.groupby("game_id")["away_score"].shift(1).fillna(0)
    rows = []
    trip = []
    key = None

    def is_ft(row: pd.Series) -> bool:
        event_type = str(row.get("event_type", "")).lower()
        desc = str(row.get("description", "")).lower()
        return "free_throw" in event_type or event_type == "ft" or "フリースロー" in desc or "free throw" in desc

    def flush():
        nonlocal trip
        if not trip:
            return
        if len(trip) == 1:
            rows.append(trip[0])
        else:
            first = trip[0]
            last = trip[-1].copy()
            start_home = float(first.get("_prev_home_score", first.get("home_score", 0)))
            start_away = float(first.get("_prev_away_score", first.get("away_score", 0)))
            made = int(float(last.get("home_score", 0)) - start_home + float(last.get("away_score", 0)) - start_away)
            last["event_type"] = "free_throw_trip"
            last["description"] = f"{first.get('event_team', '')} free_throw_trip {made}/{len(trip)}"
            rows.append(last)
        trip = []

    for _, row in pbp.iterrows():
        item = row.to_dict()
        if is_ft(row):
            row_key = free_throw_key(row)
            if trip and row_key == key:
                trip.append(item)
            else:
                flush()
                key = row_key
                trip = [item]
        else:
            flush()
            rows.append(item)
            key = None
    flush()

    out = pd.DataFrame(rows).drop(columns=["_prev_home_score", "_prev_away_score"], errors="ignore")
    out["event_id"] = out.groupby("game_id").cumcount() + 1
    return out


def import_v3_compatible_events(
    source_csv: Path,
    games_output: Path,
    pbp_output: Path,
    home_netrtg_pre: float = 0.0,
    away_netrtg_pre: float = 0.0,
    home_oftg_pre: float = 0.0,
    away_oftg_pre: float = 0.0,
    home_dftg_pre: float = 0.0,
    away_dftg_pre: float = 0.0,
    home_cs_exp: float = 0.0,
    away_cs_exp: float = 0.0,
    home_recent_form: float = 0.0,
    away_recent_form: float = 0.0,
    home_matchup_edge: float = 0.0,
    away_matchup_edge: float = 0.0,
    expected_pace: float = 70.0,
) -> tuple[Path, Path]:
    events = pd.read_csv(source_csv, encoding="utf-8-sig")
    if events.empty:
        raise ValueError(f"No events found in {source_csv}")
    events = events.sort_values(["period", "seconds_elapsed", "event_id"] if "seconds_elapsed" in events.columns else ["period", "event_id"], kind="stable")
    final = events.iloc[-1]
    neutral_site = int(pd.to_numeric(events.get("neutral_site", pd.Series([0])), errors="coerce").fillna(0).iloc[0])

    games = pd.DataFrame([{
        "game_id": final["game_id"],
        "season": events.get("season", pd.Series([""])).iloc[0],
        "date": events.get("game_date", pd.Series([""])).iloc[0],
        "home_team": events["home_team"].iloc[0],
        "away_team": events["away_team"].iloc[0],
        "home_score": int(final["home_score"]),
        "away_score": int(final["away_score"]),
        "home_win": int(final["home_score"] > final["away_score"]),
        "neutral_site": neutral_site,
        "home_netrtg_pre": home_netrtg_pre,
        "away_netrtg_pre": away_netrtg_pre,
        "home_oftg_pre": home_oftg_pre,
        "away_oftg_pre": away_oftg_pre,
        "home_dftg_pre": home_dftg_pre,
        "away_dftg_pre": away_dftg_pre,
        "home_cs_exp": home_cs_exp,
        "away_cs_exp": away_cs_exp,
        "home_recent_form": home_recent_form,
        "away_recent_form": away_recent_form,
        "home_matchup_edge": home_matchup_edge,
        "away_matchup_edge": away_matchup_edge,
        "expected_pace": expected_pace,
    }])

    pbp = events.copy()
    for column in PBP_COLUMNS:
        if column not in pbp.columns:
            pbp[column] = ""
    pbp = collapse_free_throw_trips(pbp[PBP_COLUMNS].copy())
    games_output.parent.mkdir(parents=True, exist_ok=True)
    pbp_output.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(games_output, index=False)
    pbp.to_csv(pbp_output, index=False)
    return games_output, pbp_output
