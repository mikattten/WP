from __future__ import annotations

import pandas as pd

from .features import build_feature_table
from .model import load_model_bundle, predict_team_a_wp


def sanity_cases() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.DataFrame([{
        "game_id": "sanity",
        "season": "",
        "date": "",
        "home_team": "シーホース三河",
        "away_team": "琉球",
        "home_score": 0,
        "away_score": 0,
        "home_win": 0,
        "neutral_site": 0,
        "home_netrtg_pre": 0,
        "away_netrtg_pre": 0,
        "expected_pace": 70,
    }])
    rows = [
        (1, 1, "10:00", 0, 0, "start"),
        (2, 4, "00:07", 79, 82, "down_three"),
        (3, 4, "00:07", 82, 79, "up_three"),
        (4, 4, "00:00", 79, 82, "ended_loss"),
        (5, 4, "00:00", 82, 79, "ended_win"),
    ]
    pbp = pd.DataFrame([{
        "game_id": "sanity", "event_id": e, "period": p, "clock": c,
        "home_score": hs, "away_score": aw, "possession_team": "",
        "home_team": "シーホース三河", "away_team": "琉球", "event_team": "",
        "event_type": "state", "description": d,
    } for e, p, c, hs, aw, d in rows])
    return games, pbp


def run_sanity_checks(model_path, version: str = "v3.5") -> pd.DataFrame:
    games, pbp = sanity_cases()
    features = build_feature_table(games, pbp, version=version)
    bundle = load_model_bundle(model_path, version=version)
    features["team_A_wp"] = predict_team_a_wp(bundle, features)
    checks = [
        ("start_near_coin_flip", features.iloc[0]["team_A_wp"], 42, 62),
        ("seven_seconds_down_three", features.iloc[1]["team_A_wp"], 0.1, 15),
        ("seven_seconds_up_three", features.iloc[2]["team_A_wp"], 85, 99.9),
        ("ended_loss", features.iloc[3]["team_A_wp"], 0.1, 0.1),
        ("ended_win", features.iloc[4]["team_A_wp"], 99.9, 99.9),
    ]
    return pd.DataFrame([{"case": c, "team_A_wp": wp, "min_wp": mn, "max_wp": mx, "passed": mn <= wp <= mx} for c, wp, mn, mx in checks])
