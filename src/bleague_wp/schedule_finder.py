from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CS_2025_26_GAMES_CSV
from .team_aliases import canonical_team_name


def load_cs_game_master(path: Path = CS_2025_26_GAMES_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"schedule_key": "Int64"})
    for col in ["home_team", "away_team"]:
        if col in df.columns:
            df[col] = df[col].map(canonical_team_name)
    return df


def matchup_label(row: pd.Series) -> str:
    return f"{row['home_team']} vs {row['away_team']}"


def filter_master(
    master: pd.DataFrame,
    season: str | None = None,
    competition: str | None = None,
    round_name: str | None = None,
) -> pd.DataFrame:
    df = master.copy()
    if season and "season" in df.columns:
        df = df[df["season"].astype(str) == season]
    if competition and "competition" in df.columns:
        df = df[df["competition"].astype(str) == competition]
    if round_name and "round" in df.columns:
        df = df[df["round"].astype(str) == round_name]
    return df


def resolve_schedule_key(master: pd.DataFrame, round_name: str, matchup: str, game_no: str) -> pd.Series | None:
    if master.empty:
        return None
    df = master[master["round"].astype(str).eq(round_name)].copy()
    df["matchup"] = df.apply(matchup_label, axis=1)
    hit = df[df["matchup"].eq(matchup) & df["game_no"].astype(str).eq(game_no)]
    if hit.empty:
        return None
    return hit.iloc[0]
