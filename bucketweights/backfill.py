"""
One-time (and self-healing) history load: past seasons from sportsdataverse's copies of ESPN's
play-by-play, run through the same shot rules as the nightly reader. The model needs the three
seasons before the current one for every player's level.

    python -m bucketweights.backfill 2024 2025 2026
"""
from __future__ import annotations

import io
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import db
from . import shots as S

URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_pbp/play_by_play_{season}.parquet"
# NBA Cup finals don't count in regular-season stats; the history files don't mark them.
CUP_FINALS = {401607495, 401734908, 401809839}


def load_parquet(season: int) -> pd.DataFrame:
    local = db.ROOT / "data" / "raw" / "espn_pbp" / f"play_by_play_{season}.parquet"
    if local.exists():
        return pd.read_parquet(local)
    r = requests.get(URL.format(season=season), timeout=120)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def backfill(season: int, con=None) -> int:
    con = con or db.connect()
    df = load_parquet(season)
    df = df[df.season_type == 2].sort_values(["game_id", "game_play_number"])
    pa = df["points_attempted"] if "points_attempted" in df.columns else pd.Series(np.nan, index=df.index)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    for gid, g in df.groupby("game_id", sort=False):
        first = g.iloc[0]
        reg = g[g.period_number <= 4]
        game = {
            "game_id": int(gid), "season": season, "season_type": 2,
            "competition_type": "CC" if int(gid) in CUP_FINALS else "",
            "date": str(first.game_date)[:10],
            "home_id": int(first.home_team_id), "away_id": int(first.away_team_id),
            "home_pts": int(g.home_score.max()), "away_pts": int(g.away_score.max()),
            "home_reg": int(reg.home_score.max()) if len(reg) else 0, "away_reg": int(reg.away_score.max()) if len(reg) else 0,
            "periods": int(g.period_number.max()),
            "spread_home": float(first.home_team_spread) if bool(first.get("game_spread_available", False)) else None,
        }
        rows = []
        sh = g[g.shooting_play & g.team_id.notna() & g.athlete_id_1.notna()]
        for p in sh.itertuples():
            x = None if pd.isna(p.coordinate_x_raw) else float(p.coordinate_x_raw)
            y = None if pd.isna(p.coordinate_y_raw) else float(p.coordinate_y_raw)
            pts = pa.get(p.Index)
            pts = None if pts is None or pd.isna(pts) else int(pts)
            kind = S.kind_of(p.type_text or "", p.text or "", pts if pts in (1, 2, 3) else None, bool(p.scoring_play), int(p.score_value or 0), x, y)
            if kind == "FT" and pts not in (None, 1):
                continue  # the odd mis-coded row
            dist = None if kind == "FT" else S.distance(p.text or "", x, y)
            rows.append({
                "game_id": int(gid), "seq": int(p.game_play_number), "team_id": int(p.team_id), "shooter": int(p.athlete_id_1),
                "kind": kind, "zone": S.zone_of(kind, dist, p.type_text or "", y), "made": int(bool(p.scoring_play)),
                "period": int(p.period_number), "dist": dist, "type_text": p.type_text or "",
            })
        names = sh.drop_duplicates("athlete_id_1")
        players = [{"player_id": int(r.athlete_id_1), "name": r.athlete_name_1 or "", "team_id": int(r.team_id)} for r in names.itertuples()]
        db.write_game(con, game, rows, players, "history", "ok", now)
        n += 1
    return n


if __name__ == "__main__":
    con = db.connect()
    for s in [int(a) for a in sys.argv[1:]] or [2024, 2025, 2026]:
        print(s, backfill(s, con), "games", flush=True)
