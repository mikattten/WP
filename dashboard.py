from __future__ import annotations

import base64
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import streamlit as st

from bleague_wp.config import CHART_DIR, DATA_PROCESSED_DIR, DATA_RAW_DIR, MODEL_V30_PATH, MODEL_V35_PATH, PREDICTIONS_V35_CSV
from bleague_wp.bleague_fetcher import fetch_standard_frames
from bleague_wp.config import normalize_version
from bleague_wp.game_story import summarize_game, top_wpa_events
from bleague_wp.importers import import_v3_compatible_events
from bleague_wp.model import load_model_bundle, save_model
from bleague_wp.plotting import plot_wp_timeline
from bleague_wp.predict import add_wp_predictions, prediction_columns
from bleague_wp.schedule_finder import filter_master, load_cs_game_master, matchup_label, resolve_schedule_key


st.set_page_config(page_title="B.LEAGUE WP Balance v3.5", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent
ASSET_DIR = PROJECT_DIR / "assets"
MIKATEN_LOGO = ASSET_DIR / "mikaten_logo.png"
MIKATEN_BANNER = ASSET_DIR / "mikaten_banner.png"


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".") or "png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{encoded}"


def inject_mikaten_theme() -> None:
    banner_uri = image_data_uri(MIKATEN_BANNER)
    logo_uri = image_data_uri(MIKATEN_LOGO)
    hero_background = (
        f"linear-gradient(90deg, rgba(3, 17, 40, 0.82), rgba(5, 36, 82, 0.58), rgba(3, 17, 40, 0.86)), url('{banner_uri}')"
        if banner_uri
        else "linear-gradient(90deg, #031128, #063b7a)"
    )
    logo_html = f"<img class='mikaten-hero-logo' src='{logo_uri}' alt='ミカテン logo'>" if logo_uri else ""
    st.markdown(
        f"""
        <style>
        :root {{
            --mikaten-navy: #06152f;
            --mikaten-blue: #0750a4;
            --mikaten-ink: #071326;
            --mikaten-line: rgba(7, 80, 164, 0.22);
            --mikaten-paper: #f7fbff;
        }}
        .stApp {{
            background:
                linear-gradient(180deg, rgba(4, 18, 42, 0.035), rgba(255, 255, 255, 0.96) 300px),
                radial-gradient(circle at 18% 0%, rgba(7, 80, 164, 0.10), transparent 32%),
                #f5f8fc;
            color: var(--mikaten-ink);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #031128 0%, #062c62 54%, #031128 100%);
            border-right: 1px solid rgba(255,255,255,0.14);
        }}
        [data-testid="stSidebar"] * {{
            color: rgba(255, 255, 255, 0.92);
        }}
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [role="radiogroup"] label {{
            color: var(--mikaten-ink) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
        [data-testid="stSidebar"] [data-testid="stTextInput"] input {{
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(255,255,255,0.24);
            border-radius: 8px;
        }}
        .block-container {{
            padding-top: 1.2rem;
            max-width: 1420px;
        }}
        .mikaten-hero {{
            min-height: 270px;
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 24px;
            padding: 34px 38px;
            margin: 0 0 22px;
            border-radius: 10px;
            background-image: {hero_background};
            background-size: cover;
            background-position: center;
            box-shadow: 0 22px 46px rgba(3, 17, 40, 0.20);
            overflow: hidden;
            position: relative;
        }}
        .mikaten-hero::after {{
            content: "";
            position: absolute;
            inset: 14px;
            border: 1px solid rgba(255,255,255,0.34);
            border-radius: 8px;
            pointer-events: none;
        }}
        .mikaten-hero-content {{
            position: relative;
            z-index: 1;
            max-width: 780px;
            text-shadow: 0 2px 18px rgba(0,0,0,0.35);
        }}
        .mikaten-kicker {{
            color: rgba(255,255,255,0.86);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .mikaten-hero h1 {{
            margin: 0;
            color: #ffffff;
            font-size: clamp(2.1rem, 4.2vw, 4.8rem);
            line-height: 0.98;
            letter-spacing: 0;
            font-weight: 900;
        }}
        .mikaten-hero p {{
            max-width: 690px;
            margin: 16px 0 0;
            color: rgba(255,255,255,0.90);
            font-size: 1.02rem;
            line-height: 1.8;
        }}
        .mikaten-hero-logo {{
            position: relative;
            z-index: 1;
            width: 132px;
            height: 132px;
            object-fit: cover;
            border-radius: 999px;
            border: 4px solid rgba(255,255,255,0.86);
            box-shadow: 0 14px 34px rgba(0,0,0,0.32);
            background: #fff;
        }}
        .mikaten-section-title {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 22px 0 12px;
            color: var(--mikaten-navy);
            font-size: 1.02rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .mikaten-section-title::before {{
            content: "";
            width: 34px;
            height: 3px;
            border-radius: 999px;
            background: var(--mikaten-blue);
        }}
        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.88);
            border: 1px solid var(--mikaten-line);
            border-left: 5px solid var(--mikaten-blue);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 10px 26px rgba(6, 21, 47, 0.08);
        }}
        div[data-testid="stMetricLabel"] p {{
            color: rgba(7, 19, 38, 0.64);
            font-weight: 800;
            letter-spacing: 0.04em;
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--mikaten-navy);
        }}
        .stButton > button,
        .stDownloadButton > button {{
            border-radius: 8px;
            border: 1px solid rgba(7, 80, 164, 0.36);
            background: linear-gradient(180deg, #0a57ad, #063d83);
            color: #fff;
            font-weight: 800;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 1px solid var(--mikaten-line);
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
            font-weight: 750;
        }}
        .stDataFrame, [data-testid="stJson"] {{
            border: 1px solid var(--mikaten-line);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(6, 21, 47, 0.06);
        }}
        .mikaten-note {{
            color: rgba(7, 19, 38, 0.66);
            font-size: 0.92rem;
            line-height: 1.75;
            margin-top: -6px;
        }}
        @media (max-width: 760px) {{
            .mikaten-hero {{
                min-height: 360px;
                align-items: flex-end;
                padding: 24px;
            }}
            .mikaten-hero-logo {{
                display: none;
            }}
        }}
        </style>
        <section class="mikaten-hero">
            <div class="mikaten-hero-content">
                <div class="mikaten-kicker">MIKATEN / SEA HORSES MIKAWA ANALYSIS</div>
                <h1>B.LEAGUE WP Balance v3.5</h1>
                <p>三河定点観測スタイルで、試合の傾きを50%中心のWP Balanceとして読むための分析ダッシュボードです。公式Win Probabilityではなく、独自の試合支配バランス指標です。</p>
            </div>
            {logo_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


inject_mikaten_theme()


def ensure_model(version: str = "v3.5"):
    version = normalize_version(version)
    model_path = MODEL_V35_PATH if version == "v3.5" else MODEL_V30_PATH
    if not model_path.exists():
        save_model(model_path, version)
    return load_model_bundle(model_path, version=version)


def load_existing_predictions() -> pd.DataFrame | None:
    candidates = sorted(DATA_PROCESSED_DIR.glob("*wp_balance_v3_5.csv")) + sorted(DATA_PROCESSED_DIR.glob("*wp_predictions_v3.csv"))
    if PREDICTIONS_V35_CSV.exists():
        return pd.read_csv(PREDICTIONS_V35_CSV)
    if candidates:
        return pd.read_csv(candidates[-1])
    return None


def version_slug(version: str) -> str:
    return "v3_5" if normalize_version(version) == "v3.5" else "v3_0"


def context_from_inputs(prefix: str = "") -> dict:
    return {
        "home_netrtg_pre": st.session_state.get(prefix + "home_netrtg_pre", 0.0),
        "away_netrtg_pre": st.session_state.get(prefix + "away_netrtg_pre", 0.0),
        "home_oftg_pre": st.session_state.get(prefix + "home_oftg_pre", 0.0),
        "away_oftg_pre": st.session_state.get(prefix + "away_oftg_pre", 0.0),
        "home_dftg_pre": st.session_state.get(prefix + "home_dftg_pre", 0.0),
        "away_dftg_pre": st.session_state.get(prefix + "away_dftg_pre", 0.0),
        "home_cs_exp": st.session_state.get(prefix + "home_cs_exp", 0.0),
        "away_cs_exp": st.session_state.get(prefix + "away_cs_exp", 0.0),
        "home_recent_form": st.session_state.get(prefix + "home_recent_form", 0.0),
        "away_recent_form": st.session_state.get(prefix + "away_recent_form", 0.0),
        "home_matchup_edge": st.session_state.get(prefix + "home_matchup_edge", 0.0),
        "away_matchup_edge": st.session_state.get(prefix + "away_matchup_edge", 0.0),
        "expected_pace": st.session_state.get(prefix + "expected_pace", 70.0),
    }


def apply_master_context(context: dict, selected_game: pd.Series | None) -> dict:
    if selected_game is None:
        return context
    merged = context.copy()
    for key in list(context.keys()):
        if key in selected_game and pd.notna(selected_game[key]):
            merged[key] = float(selected_game[key])
    return merged


def build_predictions_from_upload(uploaded_file, game_id: str, context: dict, version: str, team_a: str) -> pd.DataFrame:
    model_bundle = ensure_model(version)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        source = tmp_dir / "events.csv"
        source.write_bytes(uploaded_file.getvalue())
        games_path = tmp_dir / f"{game_id}_games.csv"
        pbp_path = tmp_dir / f"{game_id}_pbp.csv"
        import_v3_compatible_events(
            source,
            games_path,
            pbp_path,
            home_netrtg_pre=context["home_netrtg_pre"],
            away_netrtg_pre=context["away_netrtg_pre"],
            home_oftg_pre=context["home_oftg_pre"],
            away_oftg_pre=context["away_oftg_pre"],
            home_dftg_pre=context["home_dftg_pre"],
            away_dftg_pre=context["away_dftg_pre"],
            home_cs_exp=context["home_cs_exp"],
            away_cs_exp=context["away_cs_exp"],
            home_recent_form=context["home_recent_form"],
            away_recent_form=context["away_recent_form"],
            home_matchup_edge=context["home_matchup_edge"],
            away_matchup_edge=context["away_matchup_edge"],
            expected_pace=context["expected_pace"],
        )
        games = pd.read_csv(games_path)
        pbp = pd.read_csv(pbp_path)
        predictions = add_wp_predictions(games, pbp, model_bundle, version=version, team_a=team_a)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    slug = version_slug(version)
    games.to_csv(DATA_RAW_DIR / f"{game_id}_games.csv", index=False)
    pbp.to_csv(DATA_RAW_DIR / f"{game_id}_pbp.csv", index=False)
    predictions[prediction_columns(version)].to_csv(DATA_PROCESSED_DIR / f"{game_id}_wp_balance_{slug}.csv", index=False)
    return predictions


def fetch_official_game(schedule_key: str, context: dict):
    games, pbp, extras = fetch_standard_frames(schedule_key, context=context, use_cache=True)
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    game_id = games["game_id"].iloc[0]
    games.to_csv(DATA_RAW_DIR / f"{game_id}_games.csv", index=False)
    pbp.to_csv(DATA_RAW_DIR / f"{game_id}_pbp.csv", index=False)
    for name, frame in extras.items():
        if not frame.empty:
            frame.to_csv(DATA_RAW_DIR / f"{game_id}_{name}.csv", index=False)
    return games, pbp, extras


def calculate_wp(games: pd.DataFrame, pbp: pd.DataFrame, version: str, team_a: str) -> pd.DataFrame:
    model_bundle = ensure_model(version)
    predictions = add_wp_predictions(games, pbp, model_bundle, version=version, team_a=team_a)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    game_id = str(games["game_id"].iloc[0])
    predictions[prediction_columns(version)].to_csv(
        DATA_PROCESSED_DIR / f"{game_id}_wp_balance_{version_slug(version)}.csv",
        index=False,
    )
    st.session_state["predictions"] = predictions
    return predictions


master = load_cs_game_master()

with st.sidebar:
    if MIKATEN_LOGO.exists():
        st.image(str(MIKATEN_LOGO), width="stretch")
    st.markdown("<div style='font-weight:900; letter-spacing:0.14em; text-align:center; margin:-6px 0 18px;'>MIKATEN</div>", unsafe_allow_html=True)
    st.header("Model")
    model_version = st.selectbox("モデルバージョン", ["v3.5", "v3.0"], index=0)
    st.caption("V3.5は独自WP Balance。公式Win Probabilityではありません。")
    st.divider()
    with st.expander("試合前コンテキスト", expanded=False):
        st.number_input("Home Net Rating", value=0.0, step=0.5, key="home_netrtg_pre")
        st.number_input("Away Net Rating", value=0.0, step=0.5, key="away_netrtg_pre")
        st.number_input("Home OFtg", value=0.0, step=0.5, key="home_oftg_pre")
        st.number_input("Away OFtg", value=0.0, step=0.5, key="away_oftg_pre")
        st.number_input("Home DFtg", value=0.0, step=0.5, key="home_dftg_pre")
        st.number_input("Away DFtg", value=0.0, step=0.5, key="away_dftg_pre")
        st.number_input("Home CS", value=0.0, step=0.5, key="home_cs_exp")
        st.number_input("Away CS", value=0.0, step=0.5, key="away_cs_exp")
        st.number_input("Home Recent Form", value=0.0, step=0.5, key="home_recent_form")
        st.number_input("Away Recent Form", value=0.0, step=0.5, key="away_recent_form")
        st.number_input("Home Matchup Edge", value=0.0, step=0.5, key="home_matchup_edge")
        st.number_input("Away Matchup Edge", value=0.0, step=0.5, key="away_matchup_edge")
        st.number_input("Expected Pace", value=70.0, step=1.0, key="expected_pace")

st.markdown('<div class="mikaten-section-title">Official CS Game Fetch</div>', unsafe_allow_html=True)
st.markdown('<p class="mikaten-note">まずは2025-26 B.LEAGUE CHAMPIONSHIPのみ対応。CS試合マスタからScheduleKeyを取り、公式game_detailのテキスト速報/Boxscore/Team Statsを取得します。</p>', unsafe_allow_html=True)

if master.empty:
    st.error("data/cs_2025_26_games.csv が見つかりません。")
    st.stop()

input_cols = st.columns([1.1, 1.4, 1.0, 2.2, 1.0])
season = input_cols[0].selectbox("season", sorted(master["season"].dropna().astype(str).unique()))
competition_options = sorted(master.loc[master["season"].astype(str).eq(season), "competition"].dropna().astype(str).unique())
competition = input_cols[1].selectbox("competition", competition_options)
round_options = [r for r in ["QF", "SF", "FINAL"] if r in set(master["round"].astype(str))]
round_name = input_cols[2].selectbox("round", round_options)
round_df = filter_master(master, season=season, competition=competition, round_name=round_name).copy()
round_df["matchup"] = round_df.apply(matchup_label, axis=1)
matchups = round_df["matchup"].drop_duplicates().tolist()
matchup = input_cols[3].selectbox("matchup", matchups)
game_options = round_df.loc[round_df["matchup"].eq(matchup), "game_no"].astype(str).tolist()
game_no = input_cols[4].selectbox("game_no", game_options)

selected_game = resolve_schedule_key(master, round_name, matchup, game_no)
default_key = "" if selected_game is None else str(int(selected_game["schedule_key"]))
advanced_cols = st.columns([1.2, 1.2, 2.2])
selected_date = advanced_cols[0].date_input("試合日", value=pd.to_datetime(selected_game["date"]).date() if selected_game is not None else None)
direct_schedule_key = advanced_cols[1].text_input("ScheduleKey直接入力", value="", placeholder=default_key)
schedule_key = direct_schedule_key.strip() or default_key
team_choices = []
if selected_game is not None:
    team_choices = [selected_game["home_team"], selected_game["away_team"]]
if "シーホース三河" not in team_choices:
    team_choices.append("シーホース三河")
team_a = advanced_cols[2].selectbox("WPの上方向チーム", team_choices, index=0)

meta_cols = st.columns(4)
meta_cols[0].metric("ScheduleKey", schedule_key or "-")
meta_cols[1].metric("Round", round_name)
meta_cols[2].metric("Game", game_no)
meta_cols[3].metric("Date", str(selected_date) if selected_date else "-")

base_context = context_from_inputs()
context = apply_master_context(base_context, selected_game)
button_cols = st.columns([1, 1, 3])
fetch_clicked = button_cols[0].button("試合データ取得", type="primary", disabled=not bool(schedule_key))
wp_clicked = button_cols[1].button("WP算出", disabled=not bool(schedule_key))

if fetch_clicked:
    try:
        with st.spinner("B.LEAGUE公式サイトから試合データを取得しています..."):
            games, pbp, extras = fetch_official_game(schedule_key, context)
        st.session_state["official_games"] = games
        st.session_state["official_pbp"] = pbp
        st.session_state["official_extras"] = extras
        st.session_state["official_schedule_key"] = schedule_key
        st.success(f"取得完了: {games['home_team'].iloc[0]} {games['home_score'].iloc[0]} - {games['away_score'].iloc[0]} {games['away_team'].iloc[0]} / {len(pbp):,} events")
    except Exception as exc:
        st.error(f"公式取得に失敗しました: {exc}")
        st.info("ScheduleKeyが未確定・試合前・公式ページ側にPBP未反映の場合は、下のCSVアップロードを使ってください。")

if wp_clicked:
    try:
        games = st.session_state.get("official_games")
        pbp = st.session_state.get("official_pbp")
        cached_key = st.session_state.get("official_schedule_key")
        if games is None or pbp is None or cached_key != schedule_key:
            with st.spinner("未取得のため、公式データ取得から実行しています..."):
                games, pbp, extras = fetch_official_game(schedule_key, context)
            st.session_state["official_games"] = games
            st.session_state["official_pbp"] = pbp
            st.session_state["official_extras"] = extras
            st.session_state["official_schedule_key"] = schedule_key
        predictions = calculate_wp(games, pbp, model_version, team_a)
        st.success("WP Balanceを算出しました。")
    except Exception as exc:
        st.error(f"WP算出に失敗しました: {exc}")

with st.expander("CSVアップロード / 既存CSVを見る", expanded=False):
    fallback_mode = st.radio("保険ルート", ["イベントCSVをアップロード", "既存の予測CSVを見る"], horizontal=True)
    if fallback_mode == "イベントCSVをアップロード":
        uploaded_events = st.file_uploader("wp_v3_compatible_events.csv", type="csv")
        if uploaded_events is not None:
            default_game_id = Path(uploaded_events.name).stem.replace("_wp_v3_compatible_events", "")
            game_id_input = st.text_input("game_id", value=default_game_id)
            csv_team_a = st.text_input("CSV Team A", value=team_a)
            if st.button("CSVからWP Balanceを計算"):
                predictions = build_predictions_from_upload(uploaded_events, game_id_input, base_context, model_version, csv_team_a)
                st.session_state["predictions"] = predictions
    else:
        uploaded_predictions = st.file_uploader("wp_balance_predictions_v3_5.csv", type="csv")
        if uploaded_predictions is not None:
            st.session_state["predictions"] = pd.read_csv(uploaded_predictions)
        elif st.button("最新の既存予測CSVを読み込む"):
            st.session_state["predictions"] = load_existing_predictions()

predictions = st.session_state.get("predictions")
if predictions is None or predictions.empty:
    st.info("CS試合を選んで『試合データ取得』→『WP算出』を押してください。取得できない場合はCSVアップロードを使えます。")
    st.stop()

st.markdown('<div class="mikaten-section-title">Game Control Snapshot</div>', unsafe_allow_html=True)
game_id = st.selectbox("観測する試合", predictions["game_id"].drop_duplicates().tolist())
game_df = predictions[predictions["game_id"] == game_id].sort_values("event_id")
summary = summarize_game(predictions, game_id)

metric_cols = st.columns(5)
metric_cols[0].metric("Final", summary["final_score"])
metric_cols[1].metric("Max WP", f"{summary['max_mikawa_wp']:.1f}%")
metric_cols[2].metric("Min WP", f"{summary['min_mikawa_wp']:.1f}%")
metric_cols[3].metric("Biggest WPA", f"{summary['biggest_mikawa_wpa']:+.1f}")
if summary["final_5min_wp_change"] is None:
    metric_cols[4].metric("Final 5m", "-")
else:
    metric_cols[4].metric("Final 5m", f"{summary['final_5min_wp_change']:+.1f}")

chart_path = plot_wp_timeline(
    predictions,
    game_id,
    save_path=CHART_DIR / f"app_wp_balance_{version_slug(model_version)}_{game_id}.png",
    team_a=team_a,
    balance_axis=True,
)

timeline_tab, wpa_tab, events_tab, raw_tab = st.tabs(["WP Timeline", "Top WPA", "Event Log", "Official Raw"])

with timeline_tab:
    st.markdown('<div class="mikaten-section-title">Balance Timeline</div>', unsafe_allow_html=True)
    st.markdown('<p class="mikaten-note">上に行くほどTeam A優勢、下に行くほど相手優勢。50%を中心線にした独自WP Balanceです。</p>', unsafe_allow_html=True)
    st.image(str(chart_path), width="stretch")

    download_cols = st.columns(3)
    download_cols[0].download_button(
        "予測CSVをダウンロード",
        predictions[prediction_columns(model_version)].to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{game_id}_wp_balance_{version_slug(model_version)}.csv",
        mime="text/csv",
    )
    download_cols[1].download_button(
        "詳細テーブルCSV",
        game_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{game_id}_event_detail.csv",
        mime="text/csv",
    )
    download_cols[2].download_button(
        "グラフPNGをダウンロード",
        chart_path.read_bytes(),
        file_name=f"wp_balance_{version_slug(model_version)}_{game_id}.png",
        mime="image/png",
    )

with wpa_tab:
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="mikaten-section-title">Top WPA Events</div>', unsafe_allow_html=True)
        st.dataframe(top_wpa_events(predictions, game_id), width="stretch")
    with right:
        st.markdown('<div class="mikaten-section-title">Summary</div>', unsafe_allow_html=True)
        st.json(summary)

with events_tab:
    st.markdown('<div class="mikaten-section-title">Event Log</div>', unsafe_allow_html=True)
    visible_cols = [c for c in ["event_id", "period", "clock", "home_score", "away_score", "event_type", "team_A_wp", "team_B_wp", "mikawa_wp", "mikawa_wpa", "description"] if c in game_df.columns]
    st.dataframe(game_df[visible_cols], width="stretch")

with raw_tab:
    extras = st.session_state.get("official_extras", {})
    if not extras:
        st.info("公式取得ルートを使うと、Boxscore / Team Statsもここに表示します。")
    else:
        for name, frame in extras.items():
            st.subheader(name)
            st.dataframe(frame, width="stretch")
