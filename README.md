# B.LEAGUE WP Balance

This project calculates an independent WP Balance indicator from B.LEAGUE event CSVs.

Important: this is **not official Win Probability**. It is a custom game-control / win-balance index.

## Versions

- `v3.0`: previous heuristic WP prototype.
- `v3.5`: current WP Balance prototype with pregame context in the initial WP.

Both versions are available through `--version v3.0` or `--version v3.5`.

## V3.5 Definition

Initial WP Balance:

```text
50%
+ Net Rating diff adjustment
+ OFtg diff adjustment
+ DFtg diff adjustment
+ home adjustment
+ CS adjustment
+ recent form adjustment
+ matchup adjustment
```

Realtime WP Balance:

```text
initial WP
+ score margin adjustment
+ time remaining adjustment
+ possession adjustment
+ foul / FT adjustment
+ timeout / clutch adjustment
```

All WP Balance values are clipped to 0-100.

## Commands

```bash
cd bleague_wp
```

Build the v3.5 model bundle:

```bash
PYTHONPATH=src python3 -m bleague_wp.cli build-model --version v3.5 --model models/wp_balance_model_v3_5.pkl
```

Import a v3-compatible event CSV:

```bash
PYTHONPATH=src python3 -m bleague_wp.cli import-events /path/to/events.csv --game-id your_game_id
```

Predict WP Balance:

```bash
PYTHONPATH=src python3 -m bleague_wp.cli predict \
  --version v3.5 \
  --games data/raw/your_game_id_games.csv \
  --pbp data/raw/your_game_id_pbp.csv \
  --model models/wp_balance_model_v3_5.pkl \
  --output data/processed/your_game_id_wp_balance_v3_5.csv
```

Plot:

```bash
PYTHONPATH=src python3 -m bleague_wp.cli plot your_game_id \
  --predictions data/processed/your_game_id_wp_balance_v3_5.csv \
  --output outputs/charts/wp_balance_v3_5_your_game_id.png
```

Sanity check:

```bash
PYTHONPATH=src python3 -m bleague_wp.cli sanity-check --version v3.5 --model models/wp_balance_model_v3_5.pkl
```

Web app / Dashboard:

```bash
PYTHONPATH=src streamlit run app.py
```

Open the local URL shown by Streamlit, usually http://localhost:8501

The app can:

- upload a `wp_v3_compatible_events.csv` file
- set pregame context inputs for v3.5 initial WP Balance
- calculate `team_A_wp`, `team_B_wp`, `mikawa_wp`, and `mikawa_wpa`
- show the WP Balance chart
- show summary and Top WPA events
- download the prediction CSV and chart PNG

## Output Columns

The v3.5 prediction CSV includes:

- `team_A_wp`
- `team_B_wp`
- `wp_balance`
- `mikawa_wp`
- `mikawa_wpa`
- `time`
- `quarter`
- `score_margin`
- `event_type`

## Current Limitations

The model is still a heuristic prototype. Full v3.5 learning requires complete PBP:

- possessions
- missed shots
- rebounds
- turnovers
- free throws
- fouls
- timeouts
- robust pregame team ratings

## Streamlit App: CS Official Fetch MVP

The dashboard now supports a 2025-26 B.LEAGUE CHAMPIONSHIP MVP flow:

1. Select season, competition, round, matchup, and GAME1/GAME2/GAME3 from `data/cs_2025_26_games.csv`.
2. Use the resolved ScheduleKey to fetch `https://www.bleague.jp/game_detail/?ScheduleKey=...`.
3. Parse the embedded B.LEAGUE game context JSON for game metadata, PlayByPlays, box score, and team summaries.
4. Convert the official play-by-play into the internal standard event schema.
5. Calculate independent WP Balance using `v3.5` by default, with `v3.0` kept for comparison.

Fallbacks are intentionally preserved:

- Direct ScheduleKey input when the CS master is missing a game.
- CSV upload when the official page has no play-by-play yet or network access fails.
- Existing prediction CSV loading for review-only use.

V3.5 initial-WP weights are configured in `config/wp_v35_config.yaml`; team aliases are configured in `config/team_aliases.json`.

## Public Deployment

This project is ready to deploy as a public Streamlit app. Use `app.py` as the entrypoint.

Required files for deployment:

- `app.py`
- `dashboard.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `src/bleague_wp/`
- `config/`
- `data/cs_2025_26_games.csv`
- `assets/`

### Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Create a new Streamlit app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

### Render or Similar Hosts

Use the included `Procfile`:

```bash
web: streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

### Docker

The included `Dockerfile` runs the app on port `8501`:

```bash
docker build -t bleague-wp-balance .
docker run -p 8501:8501 bleague-wp-balance
```

Public deployments fetch B.LEAGUE pages with `requests` and parse the embedded official game context JSON. If a future B.LEAGUE page requires browser execution, the existing Playwright fallback in `src/bleague_wp/bleague_fetcher.py` can be enabled by adding Playwright and browser runtime setup to the deployment environment.

