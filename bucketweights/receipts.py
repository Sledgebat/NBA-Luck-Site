"""
The receipts: every Heat check and Defrost call, checked against what happened next. Misses included.

  record    each night's lists (top 10 each) go into the `calls` table, exactly as published
  replay    last season's lists rebuilt every Monday from only the games before that date, so the
            page has real calls from day one (done once, by itself, the first time it's needed)
  evaluate  a call is judged on scoring per shot: (points) ÷ (field-goal tries + 0.44 × free-throw
            tries), from the day after the call vs the season up to it. Defrost is right when it
            went up, Heat check when it went down. That's the claim we make ("at the same shots and
            minutes"); points a game also move with minutes and role, which we don't call.
  chances   the game chances: when we made a team the favourite at 70-80%, did it win 70-80%?

Only one call per player per list per week counts in the totals (the week's first night), so a
player who sits on a list for a month isn't counted thirty times.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from . import model, update

LISTS = ("defrost", "heat_check")
TOP = 10
MIN_GAMES_AFTER, MIN_CHANCES_AFTER = 5, 40
FT_WEIGHT = 0.44
BUCKETS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]


def record(con, day: str, season: int, hot: dict, source: str = "nightly") -> int:
    """Save the lists as published for `day` (replacing an earlier run's for the same day)."""
    rows = [(day, season, which, rank, p["id"], p["name"], p["team"], p["gp"], p["ppg"], p["proj_ppg"], source)
            for which in LISTS for rank, p in enumerate(hot.get(which, [])[:TOP], 1)]
    with con:
        con.execute("DELETE FROM calls WHERE day = ? AND source = ?", (day, source))
        con.executemany("INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def replay(con, season: int) -> int:
    """Last season's lists as they would have stood every Monday, from only the games before it."""
    shots = model.load_shots(con, list(range(season - model.PARAMS["prior_seasons"], season + 1)))
    rates = model.league_rates(shots[shots.season == season - 1])
    apps = pd.read_sql(
        "SELECT a.*, g.date FROM appearances a JOIN games g USING (game_id) WHERE g.season = ? AND g.season_type = 2",
        con, params=(season,))
    who = update.names(con)
    reg = shots[(shots.season == season) & shots.regular]
    if reg.empty:
        return 0
    first, last = date.fromisoformat(reg.date.min()), date.fromisoformat(reg.date.max())
    cut = first + timedelta(days=28)
    cut += timedelta(days=(7 - cut.weekday()) % 7)  # a Monday, four weeks in
    n = 0
    while cut <= last - timedelta(days=14):
        day = (cut - timedelta(days=1)).isoformat()  # the lists as of the night before
        before = shots[(shots.season < season) | (shots.date <= day)]
        table = model.player_table(before, apps[apps.date <= day], season, rates)
        hc = model.hot_cold(table, n=TOP)
        hot = {k: [update.player_json(r, who) for r in hc[k].itertuples()] for k in LISTS}
        n += record(con, day, season, hot, "replay")
        cut += timedelta(days=7)
    return n


def ensure_replay(con, season: int) -> None:
    have = con.execute("SELECT COUNT(*) n FROM calls WHERE season = ? AND source = 'replay'", (season,)).fetchone()["n"]
    if not have:
        n = replay(con, season)
        print(f"Receipts: replayed {n} calls for {season - 1}-{str(season)[2:]}", flush=True)


def _per_shot(shots: pd.DataFrame) -> pd.DataFrame:
    """Per shooter: points, chances (FGA + 0.44 FTA), games."""
    s = shots.assign(p=shots.made * shots.kind.map(model.VALUE),
                     c=shots.kind.map({"2": 1.0, "3": 1.0, "FT": FT_WEIGHT}))
    return s.groupby("shooter").agg(pts=("p", "sum"), ch=("c", "sum"), g=("game_id", "nunique"))


def evaluate(con, season: int, source: str) -> dict:
    """Every call of a season checked so far, by week, with totals."""
    calls = pd.read_sql("SELECT * FROM calls WHERE season = ? AND source = ? ORDER BY day, list, rank", con,
                        params=(season, source))
    out = {"season": season, "season_label": f"{season - 1}-{str(season)[2:]}", "weeks": [], "totals": {}}
    if calls.empty:
        return out
    # the week's first night stands for the week
    calls["week"] = pd.to_datetime(calls.day).dt.to_period("W-SUN").dt.start_time.dt.date.astype(str)
    firsts = calls.groupby("week").day.min()
    calls = calls[calls.day.isin(firsts.values)]
    shots = model.load_shots(con, [season])
    shots = shots[shots.regular & (shots.zone != "heave")]
    last_game = shots.date.max()
    totals = {w: {"right": 0, "checked": 0, "early": 0} for w in LISTS}
    for day, group in calls.groupby("day", sort=True):
        then = _per_shot(shots[shots.date <= day])
        after = _per_shot(shots[shots.date > day])
        week = {"day": day, "lists": {}}
        for which in LISTS:
            rows = []
            for c in group[group.list == which].itertuples():
                b = then.loc[c.player_id] if c.player_id in then.index else None
                a = after.loc[c.player_id] if c.player_id in after.index else None
                pps_then = b.pts / b.ch if b is not None and b.ch else None
                enough = a is not None and a.g >= MIN_GAMES_AFTER and a.ch >= MIN_CHANCES_AFTER
                pps_after = a.pts / a.ch if enough else None
                if not enough or pps_then is None:
                    status = "early"
                else:
                    up = pps_after > pps_then
                    status = "right" if (up if which == "defrost" else not up) else "miss"
                rows.append({"rank": int(c.rank), "id": int(c.player_id), "name": c.name, "team": c.team,
                             "ppg": c.ppg, "proj_ppg": c.proj_ppg, "pps_then": pps_then, "pps_after": pps_after,
                             "games_after": int(a.g) if a is not None else 0, "status": status})
                t = totals[which]
                if status == "early":
                    t["early"] += 1
                else:
                    t["checked"] += 1
                    t["right"] += status == "right"
            r = sum(x["status"] == "right" for x in rows)
            n = sum(x["status"] != "early" for x in rows)
            week["lists"][which] = {"rows": rows, "right": r, "checked": n}
        out["weeks"].append(week)
    out["weeks"].reverse()  # newest first
    out["totals"] = totals
    out["through"] = last_game
    return out


def chances(con, games: list[dict]) -> list[dict]:
    """The favourite's chance (at true shooting) vs how often it won, in regulation (a tie counts half)."""
    if not games:
        return []
    ids = [g["game_id"] for g in games]
    reg = pd.read_sql(f"SELECT game_id, home_reg, away_reg FROM games WHERE game_id IN ({','.join('?' * len(ids))})",
                      con, params=ids).set_index("game_id")
    rows = []
    for g in games:
        if g["season_type"] != 2 or g["game_id"] not in reg.index:
            continue
        h, a = reg.loc[g["game_id"], "home_reg"], reg.loc[g["game_id"], "away_reg"]
        if pd.isna(h) or pd.isna(a):
            continue
        home_won = 1.0 if h > a else 0.5 if h == a else 0.0
        p = g["home_chance"]
        fav_p, fav_won = (p, home_won) if p >= 0.5 else (1 - p, 1 - home_won)
        rows.append((fav_p, fav_won))
    out = []
    for lo, hi in BUCKETS:
        xs = [r for r in rows if lo <= r[0] < hi]
        if xs:
            out.append({"lo": lo, "hi": min(hi, 1.0), "games": len(xs),
                        "said": sum(r[0] for r in xs) / len(xs), "won": sum(r[1] for r in xs) / len(xs)})
    return out
