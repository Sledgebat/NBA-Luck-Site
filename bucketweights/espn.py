"""
ESPN's public NBA feeds: the scoreboard for a date and the summary (box score + play-by-play) for
a game. Retries with back-off, and two ESPN hosts that serve the same data, so one bad host or a
blip doesn't fail the night.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from . import shots as S

HOSTS = ["https://site.api.espn.com", "https://site.web.api.espn.com"]
PATH = "/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "BucketWeights/1.0 (independent fan site; nightly update)"}


class FeedError(RuntimeError):
    pass


def get_json(path: str, params: dict | None = None, tries: int = 4) -> Any:
    last: Exception | None = None
    for attempt in range(tries):
        for host in HOSTS:
            try:
                r = requests.get(host + PATH + path, params=params, headers=HEADERS, timeout=20)
                if r.status_code == 200:
                    return r.json()
                last = FeedError(f"{host}{PATH}{path} → HTTP {r.status_code}")
            except (requests.RequestException, ValueError) as e:
                last = e
        time.sleep(2 * 2**attempt)
    raise FeedError(f"ESPN unreachable for {path} {params}: {last}")


def scoreboard(date_yyyymmdd: str) -> list[dict]:
    """Every game on an ESPN date (US Eastern), with its status and type."""
    j = get_json("/scoreboard", {"dates": date_yyyymmdd, "limit": 50})
    out = []
    for e in j.get("events", []):
        c = e["competitions"][0]
        out.append({
            "game_id": int(e["id"]),
            "season": int(e["season"]["year"]),
            "season_type": int(e["season"]["type"]),
            "competition_type": (c.get("type") or {}).get("abbreviation", ""),
            "notes": "; ".join(n.get("headline", "") for n in c.get("notes", [])),
            "final": bool(e["status"]["type"].get("completed")),
            "start": e.get("date"),
        })
    return out


def summary(game_id: int) -> dict:
    return get_json("/summary", {"event": game_id})


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_summary(j: dict, date: str, season: int, season_type: int, competition_type: str) -> tuple[dict, list[dict], list[dict]]:
    """One game's row, its shots and its players, in the same shape as the history backfill."""
    comp = j["header"]["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    home, away = sides["home"], sides["away"]

    def reg(c):
        ls = c.get("linescores") or []
        return int(sum(_num(p.get("displayValue", p.get("value"))) or 0 for p in ls[:4]))

    periods = max(len(home.get("linescores") or []), len(away.get("linescores") or []))
    game = {
        "game_id": int(j["header"]["id"]),
        "season": season,
        "season_type": season_type,
        "competition_type": competition_type,
        "date": date,
        "home_id": int(home["team"]["id"]),
        "away_id": int(away["team"]["id"]),
        "home_pts": int(home.get("score") or 0),
        "away_pts": int(away.get("score") or 0),
        "home_reg": reg(home),
        "away_reg": reg(away),
        "periods": periods,
        "spread_home": None,
    }
    for pc in j.get("pickcenter", []) or []:
        if pc.get("spread") is not None:
            # ESPN's spread is from the home side's view, negative when home is favoured
            game["spread_home"] = -float(pc["spread"])
            break

    rows = []
    for p in j.get("plays", []):
        if not p.get("shootingPlay"):
            continue
        team = (p.get("team") or {}).get("id")
        parts = p.get("participants") or []
        if team is None or not parts:
            continue
        coord = p.get("coordinate") or {}
        x, y = _num(coord.get("x")), _num(coord.get("y"))
        type_text = (p.get("type") or {}).get("text", "")
        text = p.get("text", "")
        kind = S.kind_of(type_text, text, p.get("pointsAttempted"), bool(p.get("scoringPlay")), int(p.get("scoreValue") or 0), x, y)
        dist = None if kind == "FT" else S.distance(text, x, y)
        rows.append({
            "game_id": game["game_id"],
            "seq": int(p.get("sequenceNumber") or len(rows)),
            "team_id": int(team),
            "shooter": int(parts[0]["athlete"]["id"]),
            "kind": kind,
            "zone": S.zone_of(kind, dist, type_text, y),
            "made": int(bool(p.get("scoringPlay"))),
            "period": int((p.get("period") or {}).get("number") or 0),
            "dist": dist,
            "type_text": type_text,
        })

    players = []
    for team in (j.get("boxscore") or {}).get("players", []):
        tid = int(team["team"]["id"])
        for block in team.get("statistics", []):
            for a in block.get("athletes", []):
                ath = a.get("athlete") or {}
                if not ath.get("id"):
                    continue
                players.append({
                    "player_id": int(ath["id"]),
                    "name": ath.get("displayName", ""),
                    "short_name": ath.get("shortName", ""),
                    "position": (ath.get("position") or {}).get("abbreviation", ""),
                    "jersey": ath.get("jersey", ""),
                    "team_id": tid,
                    "minutes": _num((a.get("stats") or [None])[0]) if a.get("stats") else None,
                    "did_not_play": bool(a.get("didNotPlay")),
                })
    return game, rows, players


def box_totals(j: dict) -> dict[int, dict]:
    """ESPN's own team box score (FGM-FGA, 3PM-3PA, FTM-FTA, points), to check the play-by-play."""
    out = {}
    for t in (j.get("boxscore") or {}).get("teams", []):
        stats = {s.get("name"): s.get("displayValue") for s in t.get("statistics", [])}
        def pair(name):
            v = stats.get(name) or ""
            a, _, b = v.partition("-")
            return (int(a), int(b)) if a.isdigit() and b.isdigit() else (None, None)
        out[int(t["team"]["id"])] = {
            "fg": pair("fieldGoalsMade-fieldGoalsAttempted"),
            "three": pair("threePointFieldGoalsMade-threePointFieldGoalsAttempted"),
            "ft": pair("freeThrowsMade-freeThrowsAttempted"),
        }
    return out
