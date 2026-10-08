# NBA Luck Site (research stage)

Was each NBA result deserved, or decided by shooting luck? Methodology research first; the site
comes only after the method is approved. Separate from HockeyWeights (nothing here touches it).

## Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy pyarrow scipy requests
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
