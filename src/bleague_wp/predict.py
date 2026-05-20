from __future__ import annotations

import pandas as pd

from .config import MIKAWA_NAMES, feature_columns_for, normalize_version
from .features import build_feature_table
from .model import predict_team_a_wp


def normalize_team_name(team: object) -> str:
    return "" if pd.isna(team) else str(team).strip()


def is_mikawa(team: object) -> bool:
    return normalize_team_name(team) in MIKAWA_NAMES


def add_wp_predictions(
    games: pd.DataFrame,
    pbp: pd.DataFrame,
    model_bundle,
    version: str = "v3.5",
    team_a: str = "三河",
) -> pd.DataFrame:
    if isinstance(model_bundle, dict):
        version = model_bundle.get("model_version", version)
    version = normalize_version(version)
    df = build_feature_table(games, pbp, version=version)
    df["team_A"] = team_a
    df["home_wp_balance"] = predict_team_a_wp(model_bundle, df)
    home_is_a = df["home_team"].map(lambda x: is_mikawa(x) or normalize_team_name(x) == normalize_team_name(team_a))
    away_is_a = df["away_team"].map(lambda x: is_mikawa(x) or normalize_team_name(x) == normalize_team_name(team_a))
    df["team_A_wp"] = df["home_wp_balance"].where(home_is_a, 100.0 - df["home_wp_balance"])
    df.loc[~(home_is_a | away_is_a), "team_A_wp"] = df.loc[~(home_is_a | away_is_a), "home_wp_balance"]
    df["team_B_wp"] = 100.0 - df["team_A_wp"]
    df["mikawa_wp"] = df["team_A_wp"]
    df["mikawa_wpa"] = df.groupby("game_id")["mikawa_wp"].diff().fillna(0.0)
    df["wp_balance"] = df["team_A_wp"]
    df["score_margin"] = df["score_margin_home"]
    df["quarter"] = df["period"]
    df["time"] = df["clock"]
    return df


def prediction_columns(version: str = "v3.5") -> list[str]:
    cols = [
        "game_id", "event_id", "time", "quarter", "period", "clock", "seconds_elapsed", "seconds_remaining",
        "home_team", "away_team", "home_score", "away_score", "score_margin", "score_margin_home",
        "event_type", "description", "initial_wp_balance", "wp_balance",
        "team_A", "team_A_wp", "team_B_wp", "mikawa_wp", "mikawa_wpa",
        *feature_columns_for(version),
    ]
    return list(dict.fromkeys(cols))
