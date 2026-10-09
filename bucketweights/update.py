"""
The nightly update: fetch finished games from ESPN, store them, recompute everything, write the
site's data files.

    python -m bucketweights.update                 # normal run
    python -m bucketweights.update --preseason     # also take preseason games (rehearsal)
    python -m bucketweights.update --from 2026-10-20 --to 2026-10-25

Safe to run any number of times: each game is replaced whole, and only when ESPN's copy changed.
Games from the last three days are always re-read, which picks up ESPN's stat corrections.
It fails (and GitHub emails Josh) only when something is actually broken: ESPN unreachable,
a game that can't be read, or data that has stopped arriving in the middle of the season.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import db, espn, model

ET = ZoneInfo("America/New_York")
OUT = Path(db.ROOT / "build" / "data")
RECHECK_DAYS = 3
TEAMS = {t["espn_id"]: t for t in json.loads((Path(__file__).with_name("teams.json")).read_text())}


def log(*a):
    print(*a, flush=True)


def today_et() -> date:
    return datetime.now(ET).date()


def fetch(con, start: date, end: date, preseason: bool) -> dict:
    """Store every finished game from start to end (ET dates). Returns counts and problems."""
    types = {2, 3, 5} | ({1} if preseason else set())
    stats = {"new": 0, "changed": 0, "same": 0, "problems": [], "finals_seen": 0}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recheck_from = today_et() - timedelta(days=RECHECK_DAYS)
    d = start
    while d <= end:
        for ev in espn.scoreboard(d.strftime("%Y%m%d")):
            if not ev["final"] or ev["season_type"] not in types:
                continue
            if not all(t in TEAMS for t in ev["team_ids"]):
                continue  # All-Star games and other exhibitions
            stats["finals_seen"] += 1
            have = db.stored_hash(con, ev["game_id"])
            if have and d < recheck_from:
                stats["same"] += 1
                continue
            try:
                j = espn.summary(ev["game_id"])
                game, shots, players = espn.parse_summary(j, d.isoformat(), ev["season"], ev["season_type"], ev["competition_type"])
                status = check_game(game, shots, espn.box_totals(j))
            except Exception as e:  # noqa: BLE001 - recorded, and fails the run below
                stats["problems"].append(f"{ev['game_id']} ({d}): {type(e).__name__}: {e}")
                continue
            h = db.content_hash(game, shots)
            if h == have:
                stats["same"] += 1
                continue
            db.write_game(con, game, shots, players, "espn", status, now)
            stats["changed" if have else "new"] += 1
            if status != "ok":
                log(f"  check: {game['game_id']} {d}: {status}")
        d += timedelta(days=1)
    return stats


def check_game(game: dict, shots: list[dict], box: dict) -> str:
    """Play-by-play vs ESPN's box score. Points must match; attempts may differ by heaves."""
    notes = []
    for side in ("home", "away"):
        tid = game[f"{side}_id"]
        mine = [s for s in shots if s["team_id"] == tid]
        pts = sum(model.VALUE[s["kind"]] * s["made"] for s in mine)
        if pts != game[f"{side}_pts"]:
            notes.append(f"{side} points {pts} vs {game[side + '_pts']}")
        b = box.get(tid)
        if b:
            ftm = sum(s["made"] for s in mine if s["kind"] == "FT")
            fta = sum(1 for s in mine if s["kind"] == "FT")
            tpm = sum(s["made"] for s in mine if s["kind"] == "3")
            if b["ft"][0] is not None and (ftm, fta) != b["ft"]:
                notes.append(f"{side} FT {ftm}-{fta} vs {b['ft'][0]}-{b['ft'][1]}")
            if b["three"][0] is not None and tpm != b["three"][0]:
                notes.append(f"{side} 3PM {tpm} vs {b['three'][0]}")
    return "ok" if not notes else "; ".join(notes)


def window(con, args) -> tuple[date, date]:
    end = date.fromisoformat(args.to) if args.to else today_et()
    if args.from_:
        return date.fromisoformat(args.from_), end
    season = model.season_of(end.isoformat())
    r = con.execute("SELECT MAX(date) d FROM games WHERE season = ? AND source = 'espn'", (season,)).fetchone()
    if r["d"]:
        last = date.fromisoformat(r["d"])
        return min(last, end - timedelta(days=RECHECK_DAYS)), end
    # nothing stored for this season yet: start from the first of October
    return date(season - 1, 10, 1), end


def stale_problem(con, season: int, days: int = 8) -> str | None:
    """In the middle of the regular season, no new game for `days` days means something broke
    (the All-Star break is the longest normal gap, about six days)."""
    today = today_et()
    first = con.execute("SELECT MIN(date) d, MAX(date) m FROM games WHERE season = ? AND season_type = 2 AND source = 'espn'", (season,)).fetchone()
    if not first["d"]:
        return None
    in_season = date.fromisoformat(first["d"]) + timedelta(days=7) <= today <= date(season, 4, 12)
    last = date.fromisoformat(first["m"])
    if in_season and (today - last).days > days:
        return f"No new games stored since {last} ({(today - last).days} days) in the middle of the season"
    return None


# ---------------------------------------------------------------- outputs


def names(con) -> dict[int, dict]:
    rows = con.execute("SELECT player_id, name, short_name, position, jersey, team_id FROM players").fetchall()
    return {r["player_id"]: dict(r) for r in rows}


def team_abbrev(tid) -> str:
    t = TEAMS.get(int(tid)) if tid is not None else None
    return t["abbrev"] if t else ""


def player_json(r, who: dict) -> dict:
    p = who.get(int(r.player_id), {})
    out = {
        "id": int(r.player_id), "name": p.get("name", ""), "short": p.get("short_name", ""), "pos": p.get("position", ""),
        "team": team_abbrev(p.get("team_id")), "gp": int(r.gp), "ppg": round(float(r.ppg), 1),
        "proj_ppg": round(float(r.proj_ppg), 1), "luck_ppg": round(float(r.luck_ppg), 2), "fga_pg": round(float(r.fga_pg), 1),
    }
    for k in model.KINDS:
        a = int(getattr(r, f"a{k}"))
        out[k] = {
            "m": int(getattr(r, f"m{k}")), "a": a,
            "pct": round(getattr(r, f"m{k}") / a, 4) if a else None,
            "level": round(float(getattr(r, f"level{k}")), 4),
            "lo": round(float(getattr(r, f"lo{k}")), 4), "hi": round(float(getattr(r, f"hi{k}")), 4),
        }
    return out


def game_json(g: dict, who: dict) -> dict:
    g = dict(g)
    g["home"], g["away"] = team_abbrev(g["home_id"]), team_abbrev(g["away_id"])
    for s in g["swing"]:
        p = who.get(s["player_id"], {})
        s["name"], s["short"], s["team"] = p.get("name", ""), p.get("short_name", ""), team_abbrev(s["team_id"])
    return g


def outputs(con, season: int, preseason: bool = False) -> dict:
    """Every data file for a season, as {file name: content}. Also used by the site build for last
    season's pages before opening night."""
    shots = model.load_shots(con, list(range(season - model.PARAMS["prior_seasons"], season + 1)))
    rates = model.league_rates(shots[shots.season == season - 1])
    games = pd.read_sql("SELECT * FROM games WHERE season = ? ORDER BY date, game_id", con, params=(season,))
    appearances = pd.read_sql(
        "SELECT a.* FROM appearances a JOIN games g USING (game_id) WHERE g.season = ?", con, params=(season,))
    who = names(con)
    shown = games if preseason else games[games.season_type != 1]
    verdicts = [game_json(v, who) for v in model.game_verdicts(shots, shown, season, rates)]
    table = model.player_table(shots, appearances, season, rates) if len(games) else pd.DataFrame()
    players = [player_json(r, who) for r in table.itertuples()] if len(table) else []
    hc = model.hot_cold(table) if len(table) else {"heat_check": pd.DataFrame(), "defrost": pd.DataFrame(), "eligible": 0, "min_games": 5, "min_fga": 8}
    hot = {
        "heat_check": [player_json(r, who) for r in hc["heat_check"].itertuples()],
        "defrost": [player_json(r, who) for r in hc["defrost"].itertuples()],
        "eligible": hc["eligible"], "min_games": hc["min_games"], "min_fga": hc["min_fga"],
    }
    dates = sorted({v["date"] for v in verdicts})
    last_night = {"date": dates[-1] if dates else None, "games": [v for v in verdicts if dates and v["date"] == dates[-1]]}
    teams = {}
    for v in verdicts:
        if v["season_type"] != 2 or v["competition_type"] == "CC":
            continue
        for side, other in (("home", "away"), ("away", "home")):
            ab = v[side]
            t = teams.setdefault(ab, {"team": ab, "gp": 0, "w": 0, "l": 0, "luck_pts": 0.0, "robbed": 0, "robbery_wins": 0, "deserved_w": 0.0})
            won = v[f"{side}_pts"] > v[f"{other}_pts"]
            chance = v["home_chance"] if side == "home" else 1 - v["home_chance"]
            t["gp"] += 1
            t["w" if won else "l"] += 1
            t["luck_pts"] += v["teams"][side]["luck_pts"]
            t["deserved_w"] += chance
            if v["label"] == "Robbery":
                t["robbery_wins" if won else "robbed"] += 1
    for t in teams.values():
        t["luck_pts"] = round(t["luck_pts"], 1)
        t["luck_ppg"] = round(t["luck_pts"] / t["gp"], 2) if t["gp"] else 0
        t["deserved_w"] = round(t["deserved_w"], 1)
    regular_dates = sorted({v["date"] for v in verdicts if v["season_type"] == 2})
    meta = {
        "season": season, "season_label": f"{season - 1}-{str(season)[2:]}",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_through": dates[-1] if dates else None,
        "issue": len(regular_dates), "games": len(verdicts), "players": len(players),
        "model": model.PARAMS["version"],
    }
    return {
        "meta.json": meta, "players.json": players, "hotcold.json": hot, "last_night.json": last_night,
        "games.json": verdicts, "teams.json": sorted(teams.values(), key=lambda t: t["team"]),
    }


def write_outputs(con, season: int, preseason: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    files = outputs(con, season, preseason)
    for name, obj in files.items():
        (OUT / name).write_text(json.dumps(obj, separators=(",", ":"), default=model.safe))
    return files["meta.json"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preseason", action="store_true")
    ap.add_argument("--from", dest="from_")
    ap.add_argument("--to")
    ap.add_argument("--season", type=int, help="season to build outputs for (default: today's)")
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args(argv)
    con = db.connect()
    db.drop_non_nba_games(con, set(TEAMS))
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats = {"new": 0, "changed": 0, "problems": [], "finals_seen": 0}
    failed = False
    try:
        if not args.no_fetch:
            start, end = window(con, args)
            log(f"Fetching {start} to {end}")
            stats = fetch(con, start, end, args.preseason)
            log(f"  {stats['new']} new, {stats['changed']} changed, {stats.get('same', 0)} unchanged, {len(stats['problems'])} problems")
        season = args.season or model.season_of((date.fromisoformat(args.to) if args.to else today_et()).isoformat())
        meta = write_outputs(con, season, args.preseason)
        log(f"Wrote {OUT}: issue {meta['issue']}, {meta['games']} games, {meta['players']} players, data through {meta['data_through']}")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:  # shown on the run's page on GitHub
            with open(summary, "a") as f:
                f.write(f"### Update\n\n- {stats['new']} new games, {stats['changed']} changed, {stats.get('same', 0)} unchanged\n"
                        f"- Season {meta['season_label']}: {meta['games']} games, {meta['players']} players, data through {meta['data_through']}\n")
                for p in stats["problems"]:
                    f.write(f"- **Problem:** {p}\n")
        stale = stale_problem(con, season)
        if stale and not args.to:
            stats["problems"].append(stale)
        for p in stats["problems"]:
            log("PROBLEM:", p)
        failed = bool(stats["problems"])
    except espn.FeedError as e:
        log("ESPN could not be reached:", e)
        failed = True
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed = True
    with con:
        con.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
                    (run_at, stats["new"], stats["changed"], "failed" if failed else "ok", "; ".join(stats["problems"])[:2000]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
