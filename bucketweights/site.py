"""
Builds the static site into build/site from the stats database and the data files written by
update.py. No server: every page is a plain HTML file, hosted free on Cloudflare Pages.

    python -m bucketweights.site                  # this season
    python -m bucketweights.site --season 2026    # a past season (for checking)
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import db, model, update

SITE = db.ROOT / "site"
OUT = db.ROOT / "build" / "site"
DATA = update.OUT
ISSUE_COLOURS = ["yellow", "white", "cyan", "black", "orange", "stone"]
TEAM_LIST = json.loads((Path(__file__).with_name("teams.json")).read_text())
BY_ABBREV = {t["abbrev"]: t for t in TEAM_LIST}
BY_ESPN = {t["espn_id"]: t for t in TEAM_LIST}
REPO = "https://github.com/Sledgebat/NBA-Luck-Site"


# ---------------------------------------------------------------- small helpers for templates


def pct(x, digits=1):
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def pct_bare(x, digits=1):
    return "—" if x is None else f"{100 * x:.{digits}f}"


def signed(x, digits=1):
    if x is None:
        return "—"
    v = round(x, digits)
    return f"+{v:.{digits}f}" if v > 0 else (f"−{abs(v):.{digits}f}" if v < 0 else f"{0:.{digits}f}")


def long_date(iso: str | None) -> str:
    if not iso:
        return ""
    d = date.fromisoformat(iso)
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}"


def short_date(iso: str | None) -> str:
    if not iso:
        return ""
    d = date.fromisoformat(iso)
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b')}"


def verdict_class(label: str) -> str:
    return {"Robbery": "v-robbery", "Coin flip": "v-coin", "Earned it": "v-earned"}.get(label, "")


def team_name(abbrev: str, short: bool = True) -> str:
    t = BY_ABBREV.get(abbrev)
    return (t["short"] if short else t["name"]) if t else abbrev


def meter(level, lo, hi, now, lo_axis, hi_axis) -> dict:
    """Positions (0–100) for the little level meter: range, level mark, this season's dot."""
    span = max(hi_axis - lo_axis, 1e-9)
    pos = lambda v: None if v is None else max(0.0, min(100.0, 100 * (v - lo_axis) / span))
    return {"lo": pos(lo), "hi": pos(hi), "level": pos(level), "now": pos(now), "axis": (lo_axis, hi_axis)}


def env() -> Environment:
    e = Environment(loader=FileSystemLoader(SITE / "templates"), autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
    e.filters.update(pct=pct, pct_bare=pct_bare, signed=signed, long_date=long_date, short_date=short_date,
                     verdict_class=verdict_class, team_name=team_name)
    e.globals.update(TEAMS=TEAM_LIST, BY_ABBREV=BY_ABBREV, REPO=REPO, meter=meter)
    return e


# ---------------------------------------------------------------- data


def read(name):
    return json.loads((DATA / name).read_text())


def season_lines(con, season: int) -> pd.DataFrame:
    """Regular-season shooting by player and season, for the card backs."""
    seasons = list(range(season - model.PARAMS["prior_seasons"], season + 1))
    q = f"""SELECT s.shooter player_id, g.season, s.kind, SUM(s.made) m, COUNT(*) a,
                   COUNT(DISTINCT s.game_id) sg, MAX(g.date) last_date
            FROM shots s JOIN games g USING (game_id)
            WHERE g.season IN ({",".join("?" * len(seasons))}) AND g.season_type = 2 AND g.competition_type != 'CC' AND s.zone != 'heave'
            GROUP BY s.shooter, g.season, s.kind"""
    x = pd.read_sql(q, con, params=seasons)
    pts = pd.read_sql(
        f"""SELECT s.shooter player_id, g.season, COUNT(DISTINCT s.game_id) gp,
                   SUM(s.made * CASE s.kind WHEN '3' THEN 3 WHEN '2' THEN 2 ELSE 1 END) pts
            FROM shots s JOIN games g USING (game_id)
            WHERE g.season IN ({",".join("?" * len(seasons))}) AND g.season_type = 2 AND g.competition_type != 'CC'
            GROUP BY s.shooter, g.season""", con, params=seasons)
    team = pd.read_sql(
        f"""SELECT s.shooter player_id, g.season, s.team_id, COUNT(*) n FROM shots s JOIN games g USING (game_id)
            WHERE g.season IN ({",".join("?" * len(seasons))}) AND g.season_type = 2 GROUP BY 1, 2, 3""", con, params=seasons)
    team = team.sort_values("n").groupby(["player_id", "season"]).tail(1)[["player_id", "season", "team_id"]]
    wide = x.pivot_table(index=["player_id", "season"], columns="kind", values=["m", "a"], fill_value=0)
    wide.columns = [f"{a}{k}" for a, k in wide.columns]
    out = pts.set_index(["player_id", "season"]).join(wide).join(team.set_index(["player_id", "season"])).reset_index().fillna(0)
    return out


def game_logs(con, season: int, rates: dict, shots: pd.DataFrame) -> pd.DataFrame:
    """Per player per game this season: points, 3s, FTs and shooting luck (3s + FTs)."""
    s = shots[(shots.season == season) & shots.regular]
    pts = s.assign(v=s.kind.map(model.VALUE)).assign(p=lambda d: d.made * d.v).groupby(["shooter", "game_id"]).agg(
        pts=("p", "sum"), date=("date", "first"), team_id=("team_id", "first"), home_id=("home_id", "first"), away_id=("away_id", "first"))
    luck = model.shot_luck(shots, season, rates)
    luck = luck[luck.regular]
    l3 = luck[luck.kind == "3"].groupby(["shooter", "game_id"]).agg(m3=("made", "sum"), a3=("made", "size"), x3=("p", "sum"))
    lf = luck[luck.kind == "FT"].groupby(["shooter", "game_id"]).agg(mf=("made", "sum"), af=("made", "size"))
    lk = luck.groupby(["shooter", "game_id"]).luck.sum().rename("luck")
    return pts.join(l3).join(lf).join(lk).fillna(0).reset_index()


def last_shot(last_night: dict) -> dict | None:
    """The night's single most extreme three-point line, made vs expected."""
    best = None
    for g in last_night.get("games", []):
        for side, other in (("home", "away"), ("away", "home")):
            m, a, x = g["teams"][side]["three"]
            gap = m - x
            if a and (best is None or abs(gap) > abs(best["gap"])):
                won = g[f"{side}_pts"] > g[f"{other}_pts"]
                best = {"gap": gap, "m": m, "a": a, "x": x, "team": g[side], "opp": g[other], "won": won,
                        "score": f"{g[side + '_pts']}–{g[other + '_pts']}", "game_id": g["game_id"]}
    return best


# ---------------------------------------------------------------- build


def build(season: int | None = None) -> Path:
    con = db.connect()
    meta = read("meta.json")
    season = season or meta["season"]
    players, hot, last_night, games, teams = read("players.json"), read("hotcold.json"), read("last_night.json"), read("games.json"), read("teams.json")
    shots = model.load_shots(con, list(range(season - model.PARAMS["prior_seasons"], season + 1)))
    rates = model.league_rates(shots[shots.season == season - 1])
    lines = season_lines(con, season)
    logs = game_logs(con, season, rates, shots)
    check = dict(con.execute("SELECT game_id, check_status FROM games WHERE season = ?", (season,)).fetchall())

    # Before the first regular-season game: last season's lists, so the cover isn't empty.
    carried = None
    if not hot["heat_check"] and not hot["defrost"]:
        prev_shots = model.load_shots(con, list(range(season - 1 - model.PARAMS["prior_seasons"], season)))
        prev_rates = model.league_rates(prev_shots[prev_shots.season == season - 2])
        apps = pd.read_sql("SELECT a.* FROM appearances a JOIN games g USING (game_id) WHERE g.season = ?", con, params=(season - 1,))
        table = model.player_table(prev_shots, apps, season - 1, prev_rates)
        hc = model.hot_cold(table, n=5)
        who = update.names(con)
        carried = {
            "season_label": f"{season - 2}-{str(season - 1)[2:]}",
            "heat_check": [update.player_json(r, who) for r in hc["heat_check"].itertuples()],
            "defrost": [update.player_json(r, who) for r in hc["defrost"].itertuples()],
        }

    issue = meta.get("issue") or 0
    colour = ISSUE_COLOURS[issue % len(ISSUE_COLOURS)] if issue else "yellow"
    now_et = datetime.now(ZoneInfo("America/New_York"))
    common = {"meta": meta, "issue_colour": colour, "built": now_et.strftime("%-d %b %Y, %-I:%M %p ET")}

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SITE / "static", OUT / "static")
    e = env()

    def page(path: str, template: str, depth: int, **ctx):
        f = OUT / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(e.get_template(template).render(root="../" * depth, **common, **ctx))

    by_id = {p["id"]: p for p in players}
    games_sorted = sorted(games, key=lambda g: (g["date"], g["game_id"]), reverse=True)
    ln_games = sorted(last_night.get("games", []), key=lambda g: g["winner_chance"])
    for g in ln_games + games_sorted:
        g["check"] = check.get(g["game_id"], "ok")

    page("index.html", "home.html", 0, nav="home", hot=hot, carried=carried, last_night=dict(last_night, games=ln_games),
         shot=last_shot(last_night))

    # players
    page("players/index.html", "players.html", 1, nav="players", players=sorted(players, key=lambda p: -p["ppg"]))
    for p in players:
        mine = lines[lines.player_id == p["id"]].sort_values("season")
        log = logs[logs.shooter == p["id"]].sort_values("date", ascending=False)
        page(f"players/{p['id']}/index.html", "player.html", 2, nav="players", p=p, lines=mine.to_dict("records"),
             log=log.to_dict("records"), BY_ESPN=BY_ESPN)

    # games
    by_date: dict[str, list] = {}
    for g in games_sorted:
        by_date.setdefault(g["date"], []).append(g)
    page("games/index.html", "games.html", 1, nav="games", by_date=by_date)
    for g in games_sorted:
        page(f"games/{g['game_id']}/index.html", "game.html", 2, nav="games", g=g, by_id=by_id)

    # teams
    page("teams/index.html", "teams.html", 1, nav="teams", teams=sorted(teams, key=lambda t: -t["luck_pts"]))
    for t in TEAM_LIST:
        ab = t["abbrev"]
        roster = sorted([p for p in players if p["team"] == ab], key=lambda p: -p["ppg"])
        tg = [g for g in games_sorted if ab in (g["home"], g["away"])]
        row = next((x for x in teams if x["team"] == ab), None)
        page(f"teams/{ab.lower()}/index.html", "team.html", 2, nav="teams", t=t, row=row, roster=roster, games=tg)

    page("guide/index.html", "guide.html", 1, nav="guide")
    page("404.html", "404.html", 0, nav="")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    a = ap.parse_args()
    out = build(a.season)
    n = sum(1 for _ in out.rglob("*.html"))
    print(f"Built {n} pages in {out}")
