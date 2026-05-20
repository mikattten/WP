from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yaml

from .config import (
    OVERTIME_SECONDS,
    PERIOD_SECONDS,
    TOTAL_SECONDS_REGULATION,
    WP_V35_CONFIG_YAML,
    feature_columns_for,
    normalize_version,
)


REQUIRED_GAMES_COLUMNS = {
    "game_id",
    "season",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_win",
    "neutral_site",
}

REQUIRED_PBP_COLUMNS = {
    "game_id",
    "event_id",
    "period",
    "clock",
    "home_score",
    "away_score",
    "home_team",
    "away_team",
    "event_type",
    "description",
}




def load_wp_v35_initial_config() -> dict[str, float]:
    defaults = {
        "base": 50.0,
        "net_rating_weight": 0.8,
        "oftg_weight": 0.25,
        "dftg_weight": 0.25,
        "home_bonus": 2.0,
        "cs_bonus": 1.0,
        "recent_form_weight": 0.5,
        "matchup_weight": 0.5,
        "min_wp": 5.0,
        "max_wp": 95.0,
    }
    if not WP_V35_CONFIG_YAML.exists():
        return defaults
    loaded = yaml.safe_load(WP_V35_CONFIG_YAML.read_text(encoding="utf-8")) or {}
    initial = loaded.get("initial_wp", {}) or {}
    return {key: float(initial.get(key, value)) for key, value in defaults.items()}


def validate_inputs(games: pd.DataFrame, pbp: pd.DataFrame) -> None:
    missing_games = REQUIRED_GAMES_COLUMNS - set(games.columns)
    missing_pbp = REQUIRED_PBP_COLUMNS - set(pbp.columns)
    if missing_games:
        raise ValueError(f"games data is missing columns: {sorted(missing_games)}")
    if missing_pbp:
        raise ValueError(f"pbp data is missing columns: {sorted(missing_pbp)}")


def parse_clock_to_seconds(clock: object) -> int:
    if pd.isna(clock):
        return 0
    text = str(clock).strip()
    match = re.fullmatch(r"(\d{1,2}):([0-5]\d)", text)
    if not match:
        raise ValueError(f"clock must be MM:SS, got {clock!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def seconds_elapsed(period: int, clock: object) -> int:
    period = int(period)
    clock_seconds = parse_clock_to_seconds(clock)
    if period <= 4:
        return (period - 1) * PERIOD_SECONDS + (PERIOD_SECONDS - clock_seconds)
    return TOTAL_SECONDS_REGULATION + (period - 5) * OVERTIME_SECONDS + (OVERTIME_SECONDS - clock_seconds)


def total_seconds_for_period(period: int) -> int:
    period = int(period)
    if period <= 4:
        return TOTAL_SECONDS_REGULATION
    return TOTAL_SECONDS_REGULATION + (period - 4) * OVERTIME_SECONDS


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["seconds_elapsed"] = [seconds_elapsed(p, c) for p, c in zip(out["period"], out["clock"])]
    out["total_seconds"] = [total_seconds_for_period(p) for p in out["period"]]
    out["seconds_remaining"] = (out["total_seconds"] - out["seconds_elapsed"]).clip(lower=0)
    out["game_progress"] = (out["seconds_elapsed"] / out["total_seconds"]).clip(0, 1)
    return out


def event_team_side(row: pd.Series) -> str | None:
    team = "" if pd.isna(row.get("event_team")) else str(row.get("event_team")).strip()
    if team == str(row.get("home_team")).strip():
        return "home"
    if team == str(row.get("away_team")).strip():
        return "away"
    return None


def infer_scoring_and_possession(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prev_home_score"] = out.groupby("game_id")["home_score"].shift(1).fillna(0)
    out["prev_away_score"] = out.groupby("game_id")["away_score"].shift(1).fillna(0)
    out["home_points_added"] = (out["home_score"] - out["prev_home_score"]).clip(lower=0)
    out["away_points_added"] = (out["away_score"] - out["prev_away_score"]).clip(lower=0)

    poss = []
    for _, row in out.iterrows():
        team = row.get("possession_team")
        team = "" if pd.isna(team) else str(team).strip()
        if row["home_points_added"] > 0:
            poss.append(0.0)
        elif row["away_points_added"] > 0:
            poss.append(1.0)
        elif team == str(row["home_team"]).strip():
            poss.append(1.0)
        elif team == str(row["away_team"]).strip():
            poss.append(0.0)
        else:
            poss.append(0.5)
    out["possession_indicator"] = poss
    return out


def add_live_context(df: pd.DataFrame) -> pd.DataFrame:
    out = infer_scoring_and_possession(df)
    side = out.apply(event_team_side, axis=1)
    event_type = out["event_type"].fillna("").astype(str).str.lower()
    description = out["description"].fillna("").astype(str).str.lower()

    for prefix in ["home", "away"]:
        is_side = side.eq(prefix)
        ft_event = event_type.str.contains("free_throw|ft", regex=True) | description.str.contains("フリースロー|free throw", regex=True)
        shot_event = event_type.str.contains("shot|score|2p|3p", regex=True) | description.str.contains("シュート|2p|3p", regex=True)
        three_event = description.str.contains("3p|3ポイント", regex=True)
        tov_event = event_type.str.contains("turnover|tov", regex=True)
        foul_event = event_type.str.contains("foul", regex=True) | description.str.contains("ファウル", regex=True)
        timeout_event = event_type.str.contains("timeout", regex=True) | description.str.contains("タイムアウト", regex=True)

        points = out[f"{prefix}_points_added"]
        out[f"{prefix}_fga_event"] = (is_side & shot_event & ~ft_event).astype(int)
        out[f"{prefix}_fgm_event"] = (is_side & shot_event & ~ft_event & (points > 0)).astype(int)
        out[f"{prefix}_3pm_event"] = (is_side & three_event & (points >= 3)).astype(int)
        out[f"{prefix}_fta_event"] = (is_side & ft_event).astype(int)
        out[f"{prefix}_tov_event"] = (is_side & tov_event).astype(int)
        out[f"{prefix}_foul_event"] = (is_side & foul_event).astype(int)
        out[f"{prefix}_timeout_event"] = (is_side & timeout_event).astype(int)

        out[f"{prefix}_fga_live"] = numeric(out, f"{prefix}_fga", np.nan).fillna(out.groupby("game_id")[f"{prefix}_fga_event"].cumsum())
        out[f"{prefix}_fta_live"] = numeric(out, f"{prefix}_fta", np.nan).fillna(out.groupby("game_id")[f"{prefix}_fta_event"].cumsum())
        out[f"{prefix}_tov_live"] = numeric(out, f"{prefix}_tov", np.nan).fillna(out.groupby("game_id")[f"{prefix}_tov_event"].cumsum())
        out[f"{prefix}_fouls_live"] = numeric(out, f"{prefix}_fouls", np.nan).fillna(out.groupby("game_id")[f"{prefix}_foul_event"].cumsum())
        used_timeouts = numeric(out, f"{prefix}_timeouts_used", np.nan).fillna(out.groupby("game_id")[f"{prefix}_timeout_event"].cumsum())
        out[f"{prefix}_timeouts_remaining"] = numeric(out, f"{prefix}_timeouts_remaining", np.nan).fillna((5 - used_timeouts).clip(lower=0))

    poss_elapsed = (
        out["home_fga_live"] + 0.44 * out["home_fta_live"] + out["home_tov_live"]
        + out["away_fga_live"] + 0.44 * out["away_fta_live"] + out["away_tov_live"]
    ) / 2
    expected_pace = numeric(out, "expected_pace", 70.0)
    out["estimated_possessions_elapsed"] = poss_elapsed.clip(lower=0)
    out["estimated_possessions_remaining"] = (expected_pace * (1 - out["game_progress"])).clip(lower=0)
    out["is_clutch"] = ((out["period"].astype(int) == 4) & (out["seconds_remaining"] <= 300) & ((out["home_score"] - out["away_score"]).abs() <= 5)).astype(int)
    return out


def add_initial_wp_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pregame_strength_diff"] = numeric(out, "home_netrtg_pre", 0.0) - numeric(out, "away_netrtg_pre", 0.0)
    out["pregame_oftg_diff"] = numeric(out, "home_oftg_pre", 0.0) - numeric(out, "away_oftg_pre", 0.0)
    out["pregame_dftg_diff"] = numeric(out, "away_dftg_pre", 0.0) - numeric(out, "home_dftg_pre", 0.0)
    out["cs_diff"] = numeric(out, "home_cs_exp", 0.0) - numeric(out, "away_cs_exp", 0.0)
    out["recent_form_diff"] = numeric(out, "home_recent_form", 0.0) - numeric(out, "away_recent_form", 0.0)
    out["matchup_diff"] = numeric(out, "home_matchup_edge", 0.0) - numeric(out, "away_matchup_edge", 0.0)
    out["home_indicator"] = np.where(numeric(out, "neutral_site", 0.0).astype(int) == 1, 0.0, 1.0)
    config = load_wp_v35_initial_config()
    adjustment = (
        config["net_rating_weight"] * out["pregame_strength_diff"]
        + config["oftg_weight"] * out["pregame_oftg_diff"]
        + config["dftg_weight"] * out["pregame_dftg_diff"]
        + config["home_bonus"] * out["home_indicator"]
        + config["cs_bonus"] * out["cs_diff"]
        + config["recent_form_weight"] * out["recent_form_diff"]
        + config["matchup_weight"] * out["matchup_diff"]
    )
    out["initial_wp_balance"] = (config["base"] + adjustment).clip(config["min_wp"], config["max_wp"])
    return out


def build_feature_table(games: pd.DataFrame, pbp: pd.DataFrame, version: str = "v3.5") -> pd.DataFrame:
    version = normalize_version(version)
    validate_inputs(games, pbp)

    optional_game_columns = [
        "home_netrtg_pre", "away_netrtg_pre", "home_oftg_pre", "away_oftg_pre",
        "home_dftg_pre", "away_dftg_pre", "home_cs_exp", "away_cs_exp",
        "home_recent_form", "away_recent_form", "home_matchup_edge", "away_matchup_edge",
        "expected_pace",
    ]
    game_columns = ["game_id", "home_win", "neutral_site", *[c for c in optional_game_columns if c in games.columns]]
    game_meta = games[game_columns].copy()
    pbp_clean = pbp.drop(columns=[c for c in game_meta.columns if c != "game_id" and c in pbp.columns], errors="ignore")
    merged = pbp_clean.merge(game_meta, on="game_id", how="left", validate="many_to_one")
    if merged["home_win"].isna().any():
        missing = sorted(merged.loc[merged["home_win"].isna(), "game_id"].dropna().unique())
        raise ValueError(f"pbp has game_ids missing from games: {missing}")

    merged = merged.sort_values(["game_id", "event_id"], kind="stable").copy()
    merged = add_time_features(merged)
    merged["score_margin_home"] = merged["home_score"] - merged["away_score"]
    merged = add_live_context(merged)
    merged = add_initial_wp_context(merged)
    return merged


def build_training_table(games: pd.DataFrame, pbp: pd.DataFrame, version: str = "v3.5") -> pd.DataFrame:
    features = build_feature_table(games, pbp, version=version)
    columns = ["game_id", "event_id", "period", "clock", *feature_columns_for(version), "home_win"]
    return features[columns].copy()
