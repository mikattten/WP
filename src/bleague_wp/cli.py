from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import (
    CHART_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    GAMES_ALL_CSV,
    GAMES_CSV,
    MODEL_V30_PATH,
    MODEL_V35_PATH,
    PBP_ALL_CSV,
    PBP_CSV,
    PROJECT_ROOT,
    default_prediction_path,
    normalize_version,
)
from .features import build_training_table
from .game_story import summarize_game
from .importers import import_v3_compatible_events
from .model import load_model_bundle, save_model, train_model
from .plotting import plot_wp_timeline
from .predict import add_wp_predictions, prediction_columns
from .validation import run_sanity_checks


def resolve_path(base: Path, value) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"data", "models", "outputs"}:
        return PROJECT_ROOT / path
    if path.parts and path.parts[0] == base.name:
        return base.parent / path
    return base / path


def load_inputs(games_path: Path, pbp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(games_path), pd.read_csv(pbp_path)


def model_path_for(version: str) -> Path:
    return MODEL_V35_PATH if normalize_version(version) == "v3.5" else MODEL_V30_PATH


def cmd_import_events(args):
    game_id = args.game_id or Path(args.source).stem.replace("_wp_v3_compatible_events", "")
    games_out = args.games_output or DATA_RAW_DIR / f"{game_id}_games.csv"
    pbp_out = args.pbp_output or DATA_RAW_DIR / f"{game_id}_pbp.csv"
    paths = import_v3_compatible_events(
        args.source,
        games_out,
        pbp_out,
        home_netrtg_pre=args.home_netrtg_pre,
        away_netrtg_pre=args.away_netrtg_pre,
        home_oftg_pre=args.home_oftg_pre,
        away_oftg_pre=args.away_oftg_pre,
        home_dftg_pre=args.home_dftg_pre,
        away_dftg_pre=args.away_dftg_pre,
        home_cs_exp=args.home_cs_exp,
        away_cs_exp=args.away_cs_exp,
        home_recent_form=args.home_recent_form,
        away_recent_form=args.away_recent_form,
        home_matchup_edge=args.home_matchup_edge,
        away_matchup_edge=args.away_matchup_edge,
        expected_pace=args.expected_pace,
    )
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")


def cmd_build_model(args):
    if args.model is None:
        args.model = model_path_for(args.version)
    save_model(args.model, args.version)
    print(f"Wrote {args.model}")


def cmd_run_all(args):
    if args.model is None:
        args.model = model_path_for(args.version)
    if args.predictions is None:
        args.predictions = default_prediction_path(args.version)
    games, pbp = load_inputs(args.games, args.pbp)
    training = build_training_table(games, pbp, version=args.version)
    model = train_model(training, args.model, version=args.version)
    bundle = {"model": model, "feature_columns": training.columns.intersection(prediction_columns(args.version)).tolist(), "model_version": normalize_version(args.version)}
    # Reload from disk to keep metadata exact.
    bundle = load_model_bundle(args.model, version=args.version)
    pred = add_wp_predictions(games, pbp, bundle, version=args.version, team_a=args.team_a)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    pred[prediction_columns(args.version)].to_csv(args.predictions, index=False)
    for gid in pred["game_id"].drop_duplicates():
        path = CHART_DIR / f"wp_balance_{normalize_version(args.version).replace('.', '_')}_{gid}.png"
        plot_wp_timeline(pred, gid, save_path=path, team_a=args.team_a, balance_axis=args.balance_axis)
        print(f"Wrote {path}")
    print(f"Wrote {args.predictions}")
    print(json.dumps([summarize_game(pred, gid) for gid in pred["game_id"].drop_duplicates()], ensure_ascii=False, indent=2))


def cmd_predict(args):
    if args.model is None:
        args.model = model_path_for(args.version)
    if args.output is None:
        args.output = default_prediction_path(args.version)
    games, pbp = load_inputs(args.games, args.pbp)
    bundle = load_model_bundle(args.model, version=args.version)
    pred = add_wp_predictions(games, pbp, bundle, version=args.version, team_a=args.team_a)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred[prediction_columns(args.version)].to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


def cmd_plot(args):
    pred = pd.read_csv(args.predictions)
    path = plot_wp_timeline(pred, args.game_id, save_path=args.output, team_a=args.team_a, balance_axis=args.balance_axis)
    print(f"Wrote {path}")
    print(json.dumps(summarize_game(pred, args.game_id), ensure_ascii=False, indent=2))


def cmd_sanity(args):
    if args.model is None:
        args.model = model_path_for(args.version)
    result = run_sanity_checks(args.model, version=args.version)
    print(result.to_string(index=False))
    if not bool(result["passed"].all()):
        raise SystemExit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="B.LEAGUE WP Balance")
    sub = parser.add_subparsers(required=True)

    imp = sub.add_parser("import-events")
    imp.add_argument("source", type=Path)
    imp.add_argument("--game-id")
    imp.add_argument("--games-output", type=Path)
    imp.add_argument("--pbp-output", type=Path)
    for opt in ["home-netrtg-pre", "away-netrtg-pre", "home-oftg-pre", "away-oftg-pre", "home-dftg-pre", "away-dftg-pre", "home-cs-exp", "away-cs-exp", "home-recent-form", "away-recent-form", "home-matchup-edge", "away-matchup-edge"]:
        imp.add_argument(f"--{opt}", type=float, default=0.0)
    imp.add_argument("--expected-pace", type=float, default=70.0)
    imp.set_defaults(func=cmd_import_events)

    bm = sub.add_parser("build-model")
    bm.add_argument("--version", default="v3.5", choices=["v3.0", "v3.5"])
    bm.add_argument("--model", type=lambda p: resolve_path(model_path_for("v3.5").parent, p))
    bm.set_defaults(func=cmd_build_model)

    run = sub.add_parser("run-all")
    run.add_argument("--version", default="v3.5", choices=["v3.0", "v3.5"])
    run.add_argument("--games", type=lambda p: resolve_path(DATA_RAW_DIR, p), default=GAMES_CSV)
    run.add_argument("--pbp", type=lambda p: resolve_path(DATA_RAW_DIR, p), default=PBP_CSV)
    run.add_argument("--model", type=lambda p: resolve_path(model_path_for("v3.5").parent, p))
    run.add_argument("--predictions", type=lambda p: resolve_path(DATA_PROCESSED_DIR, p))
    run.add_argument("--team-a", default="三河")
    run.add_argument("--balance-axis", action="store_true", default=True)
    run.set_defaults(func=cmd_run_all)

    pred = sub.add_parser("predict")
    pred.add_argument("--version", default="v3.5", choices=["v3.0", "v3.5"])
    pred.add_argument("--games", type=lambda p: resolve_path(DATA_RAW_DIR, p), required=True)
    pred.add_argument("--pbp", type=lambda p: resolve_path(DATA_RAW_DIR, p), required=True)
    pred.add_argument("--model", type=lambda p: resolve_path(model_path_for("v3.5").parent, p))
    pred.add_argument("--output", type=lambda p: resolve_path(DATA_PROCESSED_DIR, p))
    pred.add_argument("--team-a", default="三河")
    pred.set_defaults(func=cmd_predict)

    plot = sub.add_parser("plot")
    plot.add_argument("game_id")
    plot.add_argument("--predictions", type=lambda p: resolve_path(DATA_PROCESSED_DIR, p), required=True)
    plot.add_argument("--output", type=lambda p: resolve_path(CHART_DIR, p))
    plot.add_argument("--team-a", default="三河")
    plot.add_argument("--balance-axis", action="store_true", default=True)
    plot.set_defaults(func=cmd_plot)

    sanity = sub.add_parser("sanity-check")
    sanity.add_argument("--version", default="v3.5", choices=["v3.0", "v3.5"])
    sanity.add_argument("--model", type=lambda p: resolve_path(model_path_for("v3.5").parent, p))
    sanity.set_defaults(func=cmd_sanity)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
