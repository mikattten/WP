from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import CACHE_DIR
from .importers import PBP_COLUMNS, collapse_free_throw_trips

GAME_URL_TEMPLATE = "https://www.bleague.jp/game_detail/?ScheduleKey={schedule_key}"
USER_AGENT = "Mozilla/5.0 (compatible; Mikaten-WPBalance/3.5; +https://www.bleague.jp/)"
JST = timezone(timedelta(hours=9))


class BLeagueFetchError(RuntimeError):
    pass


def game_url(schedule_key: str | int, tab: str | None = None) -> str:
    url = GAME_URL_TEMPLATE.format(schedule_key=str(schedule_key).strip())
    if tab:
        url += f"&TAB={tab}"
    return url


def fetch_html_requests(schedule_key: str | int, tab: str | None = "P", timeout: int = 25) -> str:
    response = requests.get(
        game_url(schedule_key, tab=tab),
        headers={"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def fetch_html_playwright(schedule_key: str | int, timeout_ms: int = 30000) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - optional fallback
        raise BLeagueFetchError("Playwright fallback is not available in this environment.") from exc

    url = game_url(schedule_key, tab="P")
    with sync_playwright() as p:  # pragma: no cover - optional fallback
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        html = page.content()
        browser.close()
    return html


def extract_context_json(html: str) -> dict[str, Any]:
    match = re.search(r"_contexts_s3id\.data\s*=\s*(\{.*?\});\s*\n", html, re.S)
    if not match:
        title = BeautifulSoup(html, "html.parser").title
        title_text = title.get_text(strip=True) if title else "unknown page"
        raise BLeagueFetchError(f"B.LEAGUE context JSON was not found: {title_text}")
    return json.loads(match.group(1))


def fetch_game_context(schedule_key: str | int, use_cache: bool = True, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    schedule_key = str(schedule_key).strip()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"bleague_{schedule_key}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    html = fetch_html_requests(schedule_key, tab="P")
    try:
        data = extract_context_json(html)
    except BLeagueFetchError:
        html = fetch_html_playwright(schedule_key)
        data = extract_context_json(html)

    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def parse_score(score: object, previous: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    if score is None or pd.isna(score):
        return previous
    text = str(score).strip()
    match = re.search(r"(\d+)\s*-\s*(\d+)", text)
    if not match:
        return previous
    return int(match.group(1)), int(match.group(2))


def normalize_clock(clock: object) -> str:
    text = "10:00" if clock is None or pd.isna(clock) else str(clock).strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{1,2})", text)
    if not match:
        return "00:00"
    return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"


def classify_event(row: dict[str, Any], home_score_delta: int, away_score_delta: int) -> str:
    text = str(row.get("PlayText") or "")
    action = row.get("ActionCD1")
    if "タイムアウト" in text or action == 88:
        return "timeout"
    if "フリースロー" in text:
        return "free_throw"
    if "3Pシュート" in text:
        return "3pt_made" if home_score_delta or away_score_delta else "3pt_miss"
    if "2Pシュート" in text:
        return "2pt_made" if home_score_delta or away_score_delta else "2pt_miss"
    if "ターンオーバー" in text:
        return "turnover"
    if "ファウル" in text:
        return "foul"
    if "リバウンド" in text:
        return "rebound"
    if "ピリオドスタート" in text or action == 82:
        return "period_start"
    if "ピリオドエンド" in text or action == 83:
        return "period_end"
    if "試合開始" in text or action == 80:
        return "game_start"
    if "試合終了" in text or action == 81:
        return "game_end"
    if "プレイヤーイン" in text or action == 86:
        return "substitution_in"
    if "プレイヤーアウト" in text or action == 87:
        return "substitution_out"
    if home_score_delta or away_score_delta:
        return "score"
    return "event"


def unix_to_date(value: object) -> str:
    try:
        return datetime.fromtimestamp(int(value), JST).date().isoformat()
    except Exception:
        return ""


def context_to_standard_frames(data: dict[str, Any], context: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    context = context or {}
    game = data.get("Game") or {}
    play_by_plays = data.get("PlayByPlays") or []
    if not game:
        raise BLeagueFetchError("Game metadata is missing from B.LEAGUE data.")
    if not play_by_plays:
        raise BLeagueFetchError("Play-by-play data is not available yet for this ScheduleKey.")

    game_id = f"bleague_{game.get('ScheduleKey')}"
    home_team = game.get("HomeTeamNameJ") or game.get("HomeTeamShortNameJ") or "HOME"
    away_team = game.get("AwayTeamNameJ") or game.get("AwayTeamShortNameJ") or "AWAY"
    final_home = int(game.get("HomeTeamScore") or 0)
    final_away = int(game.get("AwayTeamScore") or 0)

    games = pd.DataFrame([{
        "game_id": game_id,
        "season": "2025-26",
        "date": unix_to_date(game.get("GameDateTime")),
        "home_team": home_team,
        "away_team": away_team,
        "home_score": final_home,
        "away_score": final_away,
        "home_win": int(final_home > final_away),
        "neutral_site": int(context.get("neutral_site", 0)),
        "home_netrtg_pre": context.get("home_netrtg_pre", 0.0),
        "away_netrtg_pre": context.get("away_netrtg_pre", 0.0),
        "home_oftg_pre": context.get("home_oftg_pre", 0.0),
        "away_oftg_pre": context.get("away_oftg_pre", 0.0),
        "home_dftg_pre": context.get("home_dftg_pre", 0.0),
        "away_dftg_pre": context.get("away_dftg_pre", 0.0),
        "home_cs_exp": context.get("home_cs_exp", 0.0),
        "away_cs_exp": context.get("away_cs_exp", 0.0),
        "home_recent_form": context.get("home_recent_form", 0.0),
        "away_recent_form": context.get("away_recent_form", 0.0),
        "home_matchup_edge": context.get("home_matchup_edge", 0.0),
        "away_matchup_edge": context.get("away_matchup_edge", 0.0),
        "expected_pace": context.get("expected_pace", 70.0),
        "schedule_key": game.get("ScheduleKey"),
        "game_url": game_url(game.get("ScheduleKey")),
    }])

    rows: list[dict[str, Any]] = []
    previous_score = (0, 0)
    for idx, item in enumerate(play_by_plays, start=1):
        home_score, away_score = parse_score(item.get("Score"), previous_score)
        home_delta = max(home_score - previous_score[0], 0)
        away_delta = max(away_score - previous_score[1], 0)
        previous_score = (home_score, away_score)
        team = item.get("TeamNameJ") or ""
        rows.append({
            "game_id": game_id,
            "event_id": idx,
            "period": int(item.get("Period") or 1),
            "clock": normalize_clock(item.get("RestTime")),
            "home_score": home_score,
            "away_score": away_score,
            "possession_team": team,
            "home_team": home_team,
            "away_team": away_team,
            "event_team": team,
            "event_type": classify_event(item, home_delta, away_delta),
            "description": item.get("PlayText") or "",
            "official_action_cd1": item.get("ActionCD1"),
            "official_no": item.get("No"),
            "player": item.get("PlayerNameJ1") or "",
        })

    pbp = pd.DataFrame(rows)
    pbp = collapse_free_throw_trips(pbp[[*PBP_COLUMNS, "official_action_cd1", "official_no", "player"]].copy())
    extras = {
        "summaries": pd.DataFrame(data.get("Summaries") or []),
        "home_boxscore": pd.DataFrame(data.get("HomeBoxscores") or []),
        "away_boxscore": pd.DataFrame(data.get("AwayBoxscores") or []),
    }
    return games, pbp, extras


def fetch_standard_frames(schedule_key: str | int, context: dict[str, float] | None = None, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    data = fetch_game_context(schedule_key, use_cache=use_cache)
    return context_to_standard_frames(data, context=context)
