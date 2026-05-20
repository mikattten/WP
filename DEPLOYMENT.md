# Deployment Guide

This app is ready for public deployment as a Streamlit app.

## Recommended: Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload or push this project folder.
3. Open Streamlit Community Cloud.
4. Create a new app from the repository.
5. Set the app entrypoint to `app.py`.
6. Deploy.

The public URL will look like `https://your-app-name.streamlit.app`.

## Files That Must Be Included

- `app.py`
- `dashboard.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `src/bleague_wp/`
- `config/team_aliases.json`
- `config/wp_v35_config.yaml`
- `data/cs_2025_26_games.csv`
- `assets/mikaten_logo.png`
- `assets/mikaten_banner.png`

## Files Intentionally Excluded

The `.gitignore` excludes local/generated artifacts:

- `data/cache/*.json`
- `data/raw/*.csv`
- `data/processed/*.csv`
- `outputs/charts/*.png`
- `models/*.pkl`
- `__pycache__/`

The app can recreate model bundles and generated outputs at runtime.

## Render / Docker

Render can use the included `Procfile`:

```bash
web: streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

Docker can use the included `Dockerfile`:

```bash
docker build -t bleague-wp-balance .
docker run -p 8501:8501 bleague-wp-balance
```

## Public App Note

This is an independent WP Balance indicator. It is not official B.LEAGUE Win Probability.
