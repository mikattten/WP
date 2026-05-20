from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    japanize_matplotlib = None

from .config import CHART_DIR, PERIOD_SECONDS


def setup_japanese_font() -> None:
    candidates = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            prop = font_manager.FontProperties(fname=candidate)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def plot_wp_timeline(
    wp_df: pd.DataFrame,
    game_id: str,
    save_path: Path | None = None,
    team_a: str = "三河",
    balance_axis: bool = True,
) -> Path:
    setup_japanese_font()
    game_df = wp_df[wp_df["game_id"] == game_id].sort_values("event_id").copy()
    if game_df.empty:
        raise ValueError(f"No rows for game_id={game_id}")
    save_path = save_path or CHART_DIR / f"wp_balance_v3_5_{game_id}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_col = "team_A_wp" if "team_A_wp" in game_df.columns else "mikawa_wp"
    points = game_df[["seconds_elapsed", y_col]].drop_duplicates("seconds_elapsed", keep="last")
    if len(points) >= 3:
        timeline = pd.DataFrame({"seconds_elapsed": np.arange(0, max(2400, int(points["seconds_elapsed"].max())) + 1, 15)})
        timeline = pd.concat([timeline, points], ignore_index=True).sort_values("seconds_elapsed")
        timeline[y_col] = timeline[y_col].interpolate().ffill().bfill().ewm(alpha=0.42, adjust=False).mean()
        timeline = timeline.drop_duplicates("seconds_elapsed", keep="last")
    else:
        timeline = points

    fig, ax = plt.subplots(figsize=(12, 6.75))
    if balance_axis:
        ax.axhspan(50, 100, color="#1f8a70", alpha=0.055)
        ax.axhspan(0, 50, color="#c65f46", alpha=0.055)

    ax.plot(timeline["seconds_elapsed"], timeline[y_col], color="#0b4f8a", linewidth=3)
    ax.scatter(game_df["seconds_elapsed"], game_df[y_col], s=12, color="#0b4f8a", alpha=0.22)

    if balance_axis:
        ax.axhline(50, color="#53657d", linewidth=2.0, linestyle="--", alpha=0.9)
        ax.text(18, 51.4, "五分五分", color="#53657d", fontsize=10, weight="bold", va="bottom",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#53657d", "alpha": 0.75})
        ax.text(0.985, 0.94, f"{team_a}優勢", transform=ax.transAxes, ha="right", va="top",
                fontsize=18, color="#1f8a70", alpha=0.16, weight="bold")
        ax.text(0.985, 0.08, "相手優勢", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=18, color="#c65f46", alpha=0.16, weight="bold")
    else:
        ax.axhline(50, color="#7a869a", linewidth=1.2, linestyle="--", alpha=0.8)

    for quarter in range(1, 4):
        x = quarter * PERIOD_SECONDS
        ax.axvline(x, color="#d3dce6", linewidth=1)
        ax.text(x + 12, 4, f"{quarter + 1}Q", color="#697386", fontsize=9)

    if "mikawa_wpa" in game_df.columns:
        swing = game_df.loc[game_df["mikawa_wpa"].abs().idxmax()]
        ax.scatter([swing["seconds_elapsed"]], [swing[y_col]], s=80, color="#d94f30", zorder=5)
        desc = str(swing.get("description", ""))
        if len(desc) > 28:
            desc = desc[:25] + "..."
        ax.annotate(f"{int(swing['period'])}Q {swing['clock']}  {desc}",
                    xy=(swing["seconds_elapsed"], swing[y_col]), xytext=(18, 28),
                    textcoords="offset points",
                    arrowprops={"arrowstyle": "->", "color": "#d94f30"},
                    fontsize=9, color="#243447",
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d94f30", "alpha": 0.9})

    home = game_df["home_team"].iloc[0]
    away = game_df["away_team"].iloc[0]
    final_home = int(game_df["home_score"].iloc[-1])
    final_away = int(game_df["away_score"].iloc[-1])
    title = f"{team_a} WP Balance v3.5 Timeline: {away} @ {home}"
    subtitle = f"独自WP Balance / 公式勝率ではありません    Final: {home} {final_home} - {away} {final_away}"
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=26)
    ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, fontsize=10.5, color="#4f5b66", va="bottom")
    ax.set_xlim(0, max(2400, int(game_df["seconds_elapsed"].max())))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Game Time")
    ax.set_ylabel(f"{team_a}優勢 ← WP Balance → 相手優勢" if balance_axis else "WP Balance")
    ax.set_xticks([0, 600, 1200, 1800, 2400])
    ax.set_xticklabels(["1Q", "2Q", "3Q", "4Q", "End"])
    ax.set_yticks([0, 25, 50, 75, 100])
    if balance_axis:
        ax.set_yticklabels(["0%\n相手ほぼ勝利", "25%\n相手優勢", "50%\n五分五分", "75%\n三河優勢", "100%\n三河ほぼ勝利"])
    else:
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color="#eef2f6", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path
