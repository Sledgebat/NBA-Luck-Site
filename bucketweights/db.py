"""
The stats database (SQLite). Kept between nightly runs as a file on the repository's `data`
release, like HockeyWeights. Every write replaces a whole game at once, so running the update
twice, or re-reading a game after a stat correction, never duplicates anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("BW_DB", ROOT / "db" / "bucketweights.sqlite"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
  game_id INTEGER PRIMARY KEY,
  season INTEGER NOT NULL,            -- year the season ends (2027 = 2026-27)
  season_type INTEGER NOT NULL,       -- 1 preseason, 2 regular season, 3 playoffs, 5 play-in
  competition_type TEXT DEFAULT '',   -- 'CC' = NBA Cup final (not part of regular-season stats)
  date TEXT NOT NULL,                 -- ESPN's (US Eastern) game date, YYYY-MM-DD
  home_id INTEGER NOT NULL, away_id INTEGER NOT NULL,
  home_pts INTEGER, away_pts INTEGER, home_reg INTEGER, away_reg INTEGER, periods INTEGER,
  spread_home REAL,
  source TEXT NOT NULL,               -- 'espn' or 'history'
  content_hash TEXT,
  check_status TEXT DEFAULT 'ok',     -- 'ok' or what didn't match the box score
  fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS games_season ON games(season, season_type, date);
CREATE TABLE IF NOT EXISTS shots (
  game_id INTEGER NOT NULL, seq INTEGER NOT NULL,
  team_id INTEGER NOT NULL, shooter INTEGER NOT NULL,
  kind TEXT NOT NULL, zone TEXT NOT NULL, made INTEGER NOT NULL,
  period INTEGER NOT NULL, dist REAL, type_text TEXT,
  PRIMARY KEY (game_id, seq)
);
CREATE INDEX IF NOT EXISTS shots_shooter ON shots(shooter);
CREATE TABLE IF NOT EXISTS players (
  player_id INTEGER PRIMARY KEY, name TEXT, short_name TEXT, position TEXT, jersey TEXT,
  team_id INTEGER, last_game_date TEXT
);
CREATE TABLE IF NOT EXISTS appearances (
  game_id INTEGER NOT NULL, player_id INTEGER NOT NULL, team_id INTEGER NOT NULL,
  minutes REAL, did_not_play INTEGER,
  PRIMARY KEY (game_id, player_id)
);
CREATE TABLE IF NOT EXISTS runs (
  run_at TEXT PRIMARY KEY, new_games INTEGER, changed_games INTEGER, status TEXT, notes TEXT
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def content_hash(game: dict, shots: list[dict]) -> str:
    keep = {k: game[k] for k in ("home_pts", "away_pts", "home_reg", "away_reg", "periods")}
    blob = json.dumps([keep, [(s["seq"], s["shooter"], s["kind"], s["made"], s["zone"]) for s in shots]], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def stored_hash(con: sqlite3.Connection, game_id: int) -> str | None:
    r = con.execute("SELECT content_hash FROM games WHERE game_id = ?", (game_id,)).fetchone()
    return r["content_hash"] if r else None


def write_game(con: sqlite3.Connection, game: dict, shots: list[dict], players: list[dict], source: str,
               check_status: str, fetched_at: str) -> None:
    """Replace one game (row, shots, appearances) in a single transaction; upsert its players."""
    h = content_hash(game, shots)
    with con:
        con.execute("DELETE FROM shots WHERE game_id = ?", (game["game_id"],))
        con.execute("DELETE FROM appearances WHERE game_id = ?", (game["game_id"],))
        con.execute(
            """INSERT OR REPLACE INTO games (game_id, season, season_type, competition_type, date, home_id, away_id,
                 home_pts, away_pts, home_reg, away_reg, periods, spread_home, source, content_hash, check_status, fetched_at)
               VALUES (:game_id, :season, :season_type, :competition_type, :date, :home_id, :away_id, :home_pts, :away_pts,
                 :home_reg, :away_reg, :periods, :spread_home, :source, :hash, :check_status, :fetched_at)""",
            {**game, "source": source, "hash": h, "check_status": check_status, "fetched_at": fetched_at},
        )
        con.executemany(
            """INSERT OR REPLACE INTO shots (game_id, seq, team_id, shooter, kind, zone, made, period, dist, type_text)
               VALUES (:game_id, :seq, :team_id, :shooter, :kind, :zone, :made, :period, :dist, :type_text)""",
            shots,
        )
        for p in players:
            con.execute(
                """INSERT INTO players (player_id, name, short_name, position, jersey, team_id, last_game_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(player_id) DO UPDATE SET name = excluded.name, short_name = excluded.short_name,
                     position = CASE WHEN excluded.position != '' THEN excluded.position ELSE players.position END,
                     jersey = CASE WHEN excluded.jersey != '' THEN excluded.jersey ELSE players.jersey END,
                     team_id = CASE WHEN excluded.last_game_date >= COALESCE(players.last_game_date, '') THEN excluded.team_id ELSE players.team_id END,
                     last_game_date = MAX(COALESCE(players.last_game_date, ''), excluded.last_game_date)""",
                (p["player_id"], p["name"], p.get("short_name", ""), p.get("position", ""), p.get("jersey", ""), p["team_id"], game["date"]),
            )
            con.execute(
                "INSERT OR REPLACE INTO appearances (game_id, player_id, team_id, minutes, did_not_play) VALUES (?, ?, ?, ?, ?)",
                (game["game_id"], p["player_id"], p["team_id"], p.get("minutes"), int(bool(p.get("did_not_play")))),
            )
