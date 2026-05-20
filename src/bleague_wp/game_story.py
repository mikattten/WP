from __future__ import annotations

import pandas as pd


def summarize_game(wp_df: pd.DataFrame, game_id: str) -> dict:
    game_df = wp_df[wp_df["game_id"] == game_id].sort_values("event_id").copy()
    if game_df.empty:
        raise ValueError(f"No rows for game_id={game_id}")
    clutch = game_df[(game_df["period"] == 4) & (game_df["seconds_remaining"] <= 300)]
    final_5_change = None
    if not clutch.empty:
        final_5_change = float(clutch["mikawa_wp"].iloc[-1] - clutch["mikawa_wp"].iloc[0])
    biggest = game_df.loc[game_df["mikawa_wpa"].abs().idxmax()]
    return {
        "game_id": game_id,
        "final_score": f"{game_df['home_team'].iloc[0]} {int(game_df['home_score'].iloc[-1])} - {game_df['away_team'].iloc[0]} {int(game_df['away_score'].iloc[-1])}",
        "max_mikawa_wp": float(game_df["mikawa_wp"].max()),
        "min_mikawa_wp": float(game_df["mikawa_wp"].min()),
        "biggest_mikawa_wpa": float(biggest["mikawa_wpa"]),
        "biggest_mikawa_wpa_event": str(biggest["description"]),
        "clutch_rows_count": int(len(clutch)),
        "final_5min_wp_change": final_5_change,
    }


def top_wpa_events(wp_df: pd.DataFrame, game_id: str, n: int = 10) -> pd.DataFrame:
    cols = ["game_id", "event_id", "period", "clock", "home_score", "away_score", "description", "mikawa_wp", "mikawa_wpa"]
    return wp_df[wp_df["game_id"] == game_id].assign(abs_wpa=lambda d: d["mikawa_wpa"].abs()).sort_values("abs_wpa", ascending=False).head(n)[cols]
