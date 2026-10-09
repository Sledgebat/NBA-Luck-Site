"""
Which NBA data sources answer from this machine, how fast, and do they carry the fields the model
needs? Run locally and from GitHub Actions (the hosting environment) and compare.

    .venv/bin/python research/00_probe_sources.py [YYYYMMDD]

Writes results/probe-<hostname>.json and prints a table. Exit code 0 even when sources fail:
the point is the report.
"""
import json
import os
import socket
import sys
import time
from pathlib import Path

import requests

DATE = sys.argv[1] if len(sys.argv) > 1 else "20260410"  # a regular-season night in 2025-26
UA = {"User-Agent": "Mozilla/5.0 (probe; NBA luck research)"}
STATS_HEADERS = {
    **UA,
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json",
}
results = []


def probe(name, url, headers=UA, timeout=20, check=None):
    t0 = time.time()
    row = {"source": name, "url": url}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        row.update(status=r.status_code, seconds=round(time.time() - t0, 2), bytes=len(r.content))
        if r.ok and check:
            row["fields"] = check(r.json())
    except Exception as e:  # noqa: BLE001 — a probe reports every failure
        row.update(status="error", seconds=round(time.time() - t0, 2), error=f"{type(e).__name__}: {e}"[:200])
    results.append(row)
    return row


# --- ESPN (planned primary) ---
espn_ids = []


def espn_board(j):
    espn_ids.extend(e["id"] for e in j.get("events", []))
    return {"games": len(j.get("events", []))}


probe("espn scoreboard", f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={DATE}", check=espn_board)


def espn_summary(j):
    plays = j.get("plays", [])
    shots = [p for p in plays if p.get("shootingPlay")]
    with_xy = [p for p in shots if "coordinate" in p and p["coordinate"].get("x", -1) not in (None, -214748340)]
    return {
        "plays": len(plays),
        "shooting_plays": len(shots),
        "shots_with_coordinates": len(with_xy),
        "has_participants": any(p.get("participants") for p in shots),
        "sample_types": sorted({p.get("type", {}).get("text", "") for p in plays})[:40],
        "has_boxscore_players": bool(j.get("boxscore", {}).get("players")),
    }


if espn_ids:
    probe("espn summary (pbp+box)", f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={espn_ids[0]}", check=espn_summary)

# --- cdn.nba.com (planned fallback) ---
nba_ids = []


def schedule(j):
    dates = j["leagueSchedule"]["gameDates"]
    want = f"{DATE[4:6]}/{DATE[6:]}/{DATE[:4]}"
    for d in dates:
        if d["gameDate"].startswith(want):
            nba_ids.extend(g["gameId"] for g in d["games"])
    labels = sorted({g.get("gameLabel", "") for d in dates for g in d["games"]})
    return {"season": j["leagueSchedule"].get("seasonYear"), "game_dates": len(dates), "game_labels": labels[:20]}


probe("cdn.nba.com schedule", "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json", check=schedule)
probe("cdn.nba.com today's scoreboard", "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
      check=lambda j: {"games": len(j["scoreboard"]["games"])})


def cdn_pbp(j):
    acts = j["game"]["actions"]
    return {
        "actions": len(acts),
        "has_possession_field": all("possession" in a for a in acts),
        "shots_with_xy": sum(1 for a in acts if a.get("isFieldGoal") and a.get("x") is not None),
        "action_types": sorted({a["actionType"] for a in acts}),
    }


if nba_ids:
    gid = nba_ids[0]
    probe("cdn.nba.com play-by-play", f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json", check=cdn_pbp)
    probe("cdn.nba.com box score", f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json",
          check=lambda j: {"players": sum(len(j["game"][s]["players"]) for s in ("homeTeam", "awayTeam"))})

# --- stats.nba.com (expected to be unreliable) ---
d = f"{DATE[:4]}-{DATE[4:6]}-{DATE[6:]}"
probe("stats.nba.com scoreboard", f"https://stats.nba.com/stats/scoreboardv3?GameDate={d}&LeagueID=00", headers=STATS_HEADERS, timeout=15)

# --- sportsdataverse (historical backfill) ---


def release(j):
    names = sorted(a["name"] for a in j.get("assets", []) if a["name"].endswith(".parquet"))
    return {"parquet_files": len(names), "latest": names[-3:]}


probe("sportsdataverse ESPN pbp release", "https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases/tags/espn_nba_pbp", check=release)
probe("sportsdataverse pbp file download", "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_team_boxscores/team_box_2026.parquet")

host = socket.gethostname()
out = Path(__file__).resolve().parent.parent / "results" / f"probe-{host}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"host": host, "date_probed": DATE, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": results}, indent=2))
for r in results:
    print(f"{r['source']:42} {str(r.get('status')):>6} {r.get('seconds', ''):>6}s  {r.get('error', '')}")

# On GitHub Actions, also write a table to the run's summary page (readable without signing in).
summary = os.environ.get("GITHUB_STEP_SUMMARY")
if summary:
    with open(summary, "a") as f:
        f.write(f"### Data sources from {host}\n\n| Source | Status | Seconds | Note |\n| --- | --- | --- | --- |\n")
        for r in results:
            note = r.get("error", "") or ", ".join(f"{k}: {v}" for k, v in (r.get("fields") or {}).items() if not isinstance(v, (list, dict)))
            f.write(f"| {r['source']} | {r.get('status')} | {r.get('seconds', '')} | {note[:120]} |\n")
print(f"\nSaved {out}")
