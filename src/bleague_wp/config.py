from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
CONFIG_DIR = PROJECT_ROOT / "config"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

GAMES_CSV = DATA_RAW_DIR / "games.csv"
PBP_CSV = DATA_RAW_DIR / "pbp.csv"
GAMES_ALL_CSV = DATA_PROCESSED_DIR / "games_all.csv"
PBP_ALL_CSV = DATA_PROCESSED_DIR / "pbp_all.csv"
PREDICTIONS_V30_CSV = DATA_PROCESSED_DIR / "wp_predictions_v3.csv"
PREDICTIONS_V35_CSV = DATA_PROCESSED_DIR / "wp_balance_predictions_v3_5.csv"
MODEL_V30_PATH = MODEL_DIR / "wp_model_v3.pkl"
MODEL_V35_PATH = MODEL_DIR / "wp_balance_model_v3_5.pkl"
CS_2025_26_GAMES_CSV = PROJECT_ROOT / "data" / "cs_2025_26_games.csv"
TEAM_ALIASES_JSON = CONFIG_DIR / "team_aliases.json"
WP_V35_CONFIG_YAML = CONFIG_DIR / "wp_v35_config.yaml"

TOTAL_SECONDS_REGULATION = 40 * 60
PERIOD_SECONDS = 10 * 60
OVERTIME_SECONDS = 5 * 60

MIKAWA_NAMES = {"三河", "シーホース三河", "SeaHorses Mikawa"}

FEATURE_COLUMNS_V30 = [
    "score_margin_home",
    "game_progress",
    "seconds_remaining",
    "possession_indicator",
    "pregame_strength_diff",
    "home_indicator",
    "estimated_possessions_remaining",
]

FEATURE_COLUMNS_V35 = [
    "initial_wp_balance",
    "score_margin_home",
    "game_progress",
    "seconds_remaining",
    "possession_indicator",
    "home_indicator",
    "pregame_strength_diff",
    "pregame_oftg_diff",
    "pregame_dftg_diff",
    "cs_diff",
    "recent_form_diff",
    "matchup_diff",
    "home_fouls_live",
    "away_fouls_live",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "is_clutch",
    "estimated_possessions_remaining",
]

MODEL_FEATURES = {
    "v3.0": FEATURE_COLUMNS_V30,
    "v3": FEATURE_COLUMNS_V30,
    "v3.5": FEATURE_COLUMNS_V35,
}


def normalize_version(version: str) -> str:
    aliases = {"v3": "v3.0", "3": "v3.0", "v3.0": "v3.0", "3.0": "v3.0", "v3.5": "v3.5", "3.5": "v3.5"}
    try:
        return aliases[version]
    except KeyError as exc:
        raise ValueError("version must be 'v3.0' or 'v3.5'") from exc


def feature_columns_for(version: str) -> list[str]:
    return MODEL_FEATURES[normalize_version(version)]


def default_prediction_path(version: str) -> Path:
    return PREDICTIONS_V35_CSV if normalize_version(version) == "v3.5" else PREDICTIONS_V30_CSV
