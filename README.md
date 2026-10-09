# BucketWeights

NBA shooting luck, every night: each player's true shooting level and what he'd score at it, and
whether each game's winner earned it or got lucky. Runs itself: GitHub Actions fetches ESPN's
play-by-play three times a night, updates the stats database, rebuilds the site and publishes it to
Cloudflare Pages. Separate from HockeyWeights (nothing here touches it). Method: `docs/methodology.md`.

## Commands

| Command | What it does |
| --- | --- |
| `.venv/bin/python -m bucketweights.update` | Fetch new games from ESPN, store them, write `build/data/*.json` |
| `.venv/bin/python -m bucketweights.update --preseason` | Same, including preseason games (rehearsal) |
| `.venv/bin/python -m bucketweights.site` | Build the site into `build/site` |
| `python3 scripts/serve.py` | Look at the built site at http://localhost:4200 |
| `.venv/bin/python -m bucketweights.backfill 2024 2025 2026` | Load past seasons (the nightly run does this itself if the database is missing) |
| `.venv/bin/python -m unittest discover tests` | Tests |

The stats database lives between runs on the repository's `data` release (`bucketweights.sqlite.gz`).
Don't delete it; if it's lost, the next run rebuilds it from history in a few minutes, but this
season's games would have to be fetched again.

## Setup (once, on a Mac)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## What's here

| Path | What |
| --- | --- |
| `research/00_probe_sources.py` | Which data sources answer, how fast, with which fields (`results/probe-*.json`) |
| `.github/workflows/probe-sources.yml` | The same probe from GitHub's servers ("Run workflow" in the Actions tab) |
| `research/pilot/` | Go/no-go pilot: `parse.py` (ESPN play-by-play → shots, games), `model.py` (make chances, exact score distributions), `run_games.py`, `analyze.py` |
| `results/pilot-report.md` | Pilot findings in plain English |
| `data/` | Downloaded and derived data (git-ignored; re-download below) |

Historical play-by-play comes from sportsdataverse's GitHub releases:

```bash
B=https://github.com/sportsdataverse/sportsdataverse-data/releases/download
mkdir -p data/raw/espn_pbp data/raw/espn_team_box
for y in 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
  curl -sL -o data/raw/espn_pbp/play_by_play_$y.parquet $B/espn_nba_pbp/play_by_play_$y.parquet
  curl -sL -o data/raw/espn_team_box/team_box_$y.parquet $B/espn_nba_team_boxscores/team_box_$y.parquet
done
```
