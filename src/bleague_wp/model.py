from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_V35_PATH, feature_columns_for, normalize_version


class WPBalanceHeuristicV30:
    model_version = "v3.0"

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None):
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return wp_to_proba(wp_balance_v30(x))


class WPBalanceHeuristicV35:
    model_version = "v3.5"

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None):
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return wp_to_proba(wp_balance_v35(x))


def col(x: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name in x.columns:
        values = pd.to_numeric(x[name], errors="coerce")
        if isinstance(default, np.ndarray):
            default = pd.Series(default, index=x.index)
        return values.fillna(default).to_numpy(dtype=float)
    return np.full(len(x), default, dtype=float)


def wp_to_proba(wp_balance: np.ndarray) -> np.ndarray:
    wp = np.clip(wp_balance / 100.0, 0.001, 0.999)
    return np.column_stack([1 - wp, wp])


def apply_bounds(wp: np.ndarray, margin: np.ndarray, seconds: np.ndarray, possession: np.ndarray) -> np.ndarray:
    bounded = np.clip(wp, 0, 100)
    ended = seconds <= 0
    bounded = np.where(ended & (margin > 0), 99.9, bounded)
    bounded = np.where(ended & (margin < 0), 0.1, bounded)
    bounded = np.where(ended & (margin == 0), 50.0, bounded)

    losing = margin < 0
    opponent_ball = (losing & (possession <= 0.25)) | ((margin > 0) & (possession >= 0.75))
    late = (seconds <= 30) & losing
    very_late = (seconds <= 10) & losing
    late_caps = np.select(
        [margin <= -6, margin == -5, margin == -4, margin == -3, margin == -2, margin == -1],
        [3, 5, 8, 16, 26, 36],
        default=45,
    )
    opponent_caps = np.select(
        [margin <= -6, margin == -5, margin == -4, margin == -3, margin == -2, margin == -1],
        [2, 4, 7, 14, 22, 30],
        default=40,
    )
    very_late_caps = np.select(
        [margin <= -5, margin == -4, margin == -3, margin == -2, margin == -1],
        [1.5, 3.5, 11, 19, 32],
        default=40,
    )
    bounded = np.where(late, np.minimum(bounded, late_caps), bounded)
    bounded = np.where((seconds <= 30) & opponent_ball & losing, np.minimum(bounded, opponent_caps), bounded)
    bounded = np.where(very_late, np.minimum(bounded, very_late_caps), bounded)

    winning = margin > 0
    winning_ball = (winning & (possession >= 0.75))
    bounded = np.where((seconds <= 10) & winning & (margin >= 3), np.maximum(bounded, 89), bounded)
    bounded = np.where((seconds <= 30) & winning_ball, np.maximum(bounded, 100 - opponent_caps), bounded)
    return np.clip(bounded, 0.1, 99.9)


def temper_close_game(wp: np.ndarray, margin: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    close = np.abs(margin) <= 5
    early = close & (seconds >= 600)
    mid = close & (seconds < 600) & (seconds >= 180)
    wp = np.where(early, np.clip(wp, 26, 74), wp)
    wp = np.where(mid, np.clip(wp, 18, 82), wp)
    return wp


def wp_balance_v30(x: pd.DataFrame) -> np.ndarray:
    margin = col(x, "score_margin_home")
    seconds = col(x, "seconds_remaining", 2400).clip(min=0)
    progress = col(x, "game_progress").clip(min=0, max=1)
    poss = col(x, "possession_indicator", 0.5)
    strength = col(x, "pregame_strength_diff", 0.0)
    home = col(x, "home_indicator", 1.0)
    poss_remaining = col(x, "estimated_possessions_remaining", 70 * (1 - progress)).clip(min=0)

    pressure = margin / np.sqrt(poss_remaining + 1)
    logit = (0.50 + 0.50 * progress) * pressure + 0.055 * strength + 0.18 * home + 0.18 * (poss - 0.5)
    wp = 100 / (1 + np.exp(-logit))
    wp = temper_close_game(wp, margin, seconds)
    return apply_bounds(wp, margin, seconds, poss)


def wp_balance_v35(x: pd.DataFrame) -> np.ndarray:
    initial = col(x, "initial_wp_balance", 50.0)
    margin = col(x, "score_margin_home")
    seconds = col(x, "seconds_remaining", 2400).clip(min=0)
    progress = col(x, "game_progress").clip(min=0, max=1)
    poss = col(x, "possession_indicator", 0.5)
    home_fouls = col(x, "home_fouls_live")
    away_fouls = col(x, "away_fouls_live")
    home_to = col(x, "home_timeouts_remaining", 5)
    away_to = col(x, "away_timeouts_remaining", 5)
    clutch = col(x, "is_clutch", 0)
    poss_remaining = col(x, "estimated_possessions_remaining", 70 * (1 - progress)).clip(min=0)

    pressure_margin = margin / np.sqrt(poss_remaining + 1)
    point_swing = 8.0 * (0.55 + 0.65 * progress) * pressure_margin
    possession_swing = 2.8 * (poss - 0.5) * (0.6 + 1.2 * progress)
    foul_ft_swing = -0.55 * (home_fouls - away_fouls) * (0.2 + clutch)
    timeout_swing = 0.65 * (home_to - away_to) * (0.2 + clutch)
    clutch_swing = clutch * margin * 1.6
    wp = initial + point_swing + possession_swing + foul_ft_swing + timeout_swing + clutch_swing
    wp = temper_close_game(wp, margin, seconds)
    return apply_bounds(wp, margin, seconds, poss)


def build_model(version: str):
    return WPBalanceHeuristicV35() if normalize_version(version) == "v3.5" else WPBalanceHeuristicV30()


def train_model(training_df: pd.DataFrame, model_path: Path = MODEL_V35_PATH, version: str = "v3.5"):
    version = normalize_version(version)
    model = build_model(version)
    model.fit(training_df[feature_columns_for(version)], training_df.get("home_win"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns_for(version),
            "model_version": version,
            "row_count": len(training_df),
            "model_type": type(model).__name__,
            "official_win_probability": False,
        },
        model_path,
    )
    return model


def save_model(model_path: Path = MODEL_V35_PATH, version: str = "v3.5"):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    version = normalize_version(version)
    model = build_model(version)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns_for(version),
            "model_version": version,
            "row_count": 0,
            "model_type": type(model).__name__,
            "official_win_probability": False,
            "description": "Independent WP Balance indicator, not official win probability.",
        },
        model_path,
    )
    return model


def load_model_bundle(model_path: Path = MODEL_V35_PATH, version: str = "v3.5") -> dict:
    if model_path.exists():
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict) and "model" in bundle:
            return bundle
    model = save_model(model_path, version)
    return {"model": model, "feature_columns": feature_columns_for(version), "model_version": normalize_version(version)}


def predict_team_a_wp(model_or_bundle, feature_df: pd.DataFrame) -> pd.Series:
    if isinstance(model_or_bundle, dict):
        model = model_or_bundle["model"]
        columns = model_or_bundle["feature_columns"]
    else:
        model = model_or_bundle
        columns = feature_columns_for(getattr(model, "model_version", "v3.5"))
    return pd.Series(model.predict_proba(feature_df[columns])[:, 1] * 100.0, index=feature_df.index, name="team_A_wp")
