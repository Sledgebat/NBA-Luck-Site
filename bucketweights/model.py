"""
The BucketWeights model, version 1 (docs/methodology.md; parameters in params.json).

  levels      each player's true level for 3s, 2s and free throws: his last three seasons
              (fading by w a season) and this one, pulled toward the league rate by k attempts.
  verdicts    each game's exact chances with 3s and free throws at the shooters' levels known
              before the game, everything else as it happened, then calibrated.
  players     points a game now vs at his true shooting (same shots and minutes).

Regular-season games only feed the levels (no Cup final, preseason, play-in or playoffs).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

PARAMS = json.loads((Path(__file__).with_name("params.json")).read_text())
VALUE = {"2": 2, "3": 3, "FT": 1}
KINDS = ("3", "FT", "2")


def logit(p):
    return np.log(p / (1 - p))


def inv_logit(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- loading


def load_shots(con, seasons: list[int]) -> pd.DataFrame:
    q = f"""SELECT s.game_id, s.seq, s.team_id, s.shooter, s.kind, s.zone, s.made, s.period,
                   g.season, g.date, g.home_id, g.away_id, g.season_type, g.competition_type
            FROM shots s JOIN games g USING (game_id)
            WHERE g.season IN ({",".join("?" * len(seasons))})"""
    s = pd.read_sql(q, con, params=seasons)
    s["regular"] = (s.season_type == 2) & (s.competition_type != "CC")
    return s


# ---------------------------------------------------------------- levels


def league_rates(prev: pd.DataFrame) -> dict:
    """League make rates from the previous season: per kind (no heaves), per zone, corner/above shift."""
    x = prev[prev.regular]
    out = {k: float(x[(x.kind == k) & (x.zone != "heave")].made.mean()) for k in KINDS}
    zone = x.groupby("zone").made.mean()
    out["heave"] = float(zone.get("heave", 0.03))
    ca = x[x.zone.isin(["corner", "above"])]
    allr = ca.made.mean()
    out["zone_shift"] = {z: float(logit(ca[ca.zone == z].made.mean()) - logit(allr)) for z in ("corner", "above")}
    return out


def prior_counts(shots: pd.DataFrame, season: int, kind: str) -> pd.DataFrame:
    """Faded makes and attempts from the three seasons before `season`, per player."""
    w = PARAMS["levels"][kind]["w"]
    x = shots[shots.regular & (shots.kind == kind) & (shots.zone != "heave") & (shots.season < season) & (shots.season >= season - PARAMS["prior_seasons"])]
    wt = w ** (season - x.season)
    return pd.DataFrame({"pm": (x.made * wt).groupby(x.shooter).sum(), "pa": wt.groupby(x.shooter).sum()})


def levels_before_games(shots: pd.DataFrame, season: int, kind: str, mu: float) -> pd.Series:
    """Each (game_id, shooter)'s level using only regular-season games on earlier dates."""
    k = PARAMS["levels"][kind]["k"]
    prior = prior_counts(shots, season, kind)
    cur = shots[(shots.season == season) & (shots.kind == kind) & (shots.zone != "heave")]
    pg = cur.groupby(["shooter", "date", "game_id"]).agg(m=("made", "sum"), a=("made", "size"), reg=("regular", "first")).reset_index()
    pg = pg.sort_values(["shooter", "date", "game_id"])
    # only regular-season games add to the level, but every game gets one
    pg["m_reg"] = np.where(pg.reg, pg.m, 0)
    pg["a_reg"] = np.where(pg.reg, pg.a, 0)
    g = pg.groupby("shooter")
    pg["mb"] = g.m_reg.cumsum() - pg.m_reg
    pg["ab"] = g.a_reg.cumsum() - pg.a_reg
    pm = pg.shooter.map(prior.pm).fillna(0)
    pa = pg.shooter.map(prior.pa).fillna(0)
    pg["p"] = (pm + pg.mb + k * mu) / (pa + pg.ab + k)
    return pg.set_index(["game_id", "shooter"]).p


def current_levels(shots: pd.DataFrame, season: int, kind: str, mu: float) -> pd.DataFrame:
    """Every player's level now (all regular-season games so far), with its 80% range."""
    k = PARAMS["levels"][kind]["k"]
    prior = prior_counts(shots, season, kind)
    cur = shots[(shots.season == season) & shots.regular & (shots.kind == kind) & (shots.zone != "heave")]
    now = pd.DataFrame({"m": cur.groupby("shooter").made.sum(), "a": cur.groupby("shooter").made.size()})
    d = prior.join(now, how="outer").fillna(0)
    d["alpha"] = d.pm + d.m + k * mu
    d["beta"] = (d.pa - d.pm) + (d.a - d.m) + k * (1 - mu)
    d["level"] = d.alpha / (d.alpha + d.beta)
    lo = (1 - PARAMS["range"]) / 2
    d["lo"] = beta_dist.ppf(lo, d.alpha, d.beta)
    d["hi"] = beta_dist.ppf(1 - lo, d.alpha, d.beta)
    return d


# ---------------------------------------------------------------- exact game chances


def points_pmf(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exact distribution of sum(v_i × Bernoulli(p_i)), indexed by points."""
    pmf = np.zeros(int(v.sum()) + 1)
    pmf[0] = 1.0
    top = 0
    for pi, vi in zip(p, v):
        vi = int(vi)
        nxt = pmf * (1 - pi)
        nxt[vi: top + vi + 1] += pmf[: top + 1] * pi
        pmf = nxt
        top += vi
    return pmf


def raw_home_chance(home_fixed: int, hp, hv, away_fixed: int, ap, av) -> tuple[float, float]:
    a, b = points_pmf(np.asarray(hp, float), np.asarray(hv)), points_pmf(np.asarray(ap, float), np.asarray(av))
    d = np.convolve(a, b[::-1])
    margins = np.arange(len(d)) - (len(b) - 1) + home_fixed - away_fixed
    win, tie = float(d[margins > 0].sum()), float(d[margins == 0].sum())
    return win + tie / 2, tie


def calibrate(p_home_raw: float) -> float:
    a, b = PARAMS["recalibration"]["a"], PARAMS["recalibration"]["b"]
    p = min(max(p_home_raw, 1e-4), 1 - 1e-4)
    return float(inv_logit(a + b * logit(p)))


def label(winner_chance: float) -> str:
    if winner_chance < PARAMS["labels"]["robbery_below"]:
        return "Robbery"
    if winner_chance < PARAMS["labels"]["earned_from"]:
        return "Coin flip"
    return "Earned it"


def shot_luck(shots: pd.DataFrame, season: int, rates: dict) -> pd.DataFrame:
    """Every 3 and free throw of `season` with its make chance (shooter's level before the game)
    and its luck in points (made minus expected, times its value)."""
    s = shots[shots.season == season].copy()
    lv3 = levels_before_games(shots, season, "3", rates["3"])
    lvft = levels_before_games(shots, season, "FT", rates["FT"])
    key = pd.MultiIndex.from_frame(s[["game_id", "shooter"]])
    p3 = lv3.reindex(key).to_numpy()
    pft = lvft.reindex(key).to_numpy()
    shift = s.zone.map(rates["zone_shift"]).fillna(0).to_numpy()
    p3 = inv_logit(logit(np.clip(np.nan_to_num(p3, nan=rates["3"]), 0.01, 0.99)) + shift)
    s["p"] = np.where(s.kind == "3", np.where(s.zone == "heave", rates["heave"], p3), np.where(s.kind == "FT", np.nan_to_num(pft, nan=rates["FT"]), np.nan))
    s = s[s.p.notna()]
    s["v"] = s.kind.map(VALUE)
    s["luck"] = (s.made - s.p) * s.v
    return s


def game_verdicts(shots: pd.DataFrame, games: pd.DataFrame, season: int, rates: dict) -> list[dict]:
    """Verdicts for every finished game of `season` in `games` (one row per game)."""
    s = shot_luck(shots, season, rates)
    by_game = dict(tuple(s.groupby("game_id")))
    out = []
    for g in games.itertuples():
        x = by_game.get(g.game_id)
        if x is None:
            continue
        reg = x[x.period <= 4]
        h, a = reg[reg.team_id == g.home_id], reg[reg.team_id == g.away_id]
        raw, tie = raw_home_chance(
            g.home_reg - int((h.made * h.v).sum()), h.p, h.v,
            g.away_reg - int((a.made * a.v).sum()), a.p, a.v,
        )
        home = calibrate(raw)
        home_won = g.home_pts > g.away_pts
        winner_chance = home if home_won else 1 - home
        teams = {}
        for side, tid in (("home", g.home_id), ("away", g.away_id)):
            t = x[x.team_id == tid]
            t3, tft = t[t.kind == "3"], t[t.kind == "FT"]
            teams[side] = {
                "team_id": int(tid),
                "three": [int(t3.made.sum()), int(len(t3)), round(float(t3.p.sum()), 1)],
                "ft": [int(tft.made.sum()), int(len(tft)), round(float(tft.p.sum()), 1)],
                "luck_pts": round(float(t.luck.sum()), 1),
            }
        sw = x.groupby(["shooter", "team_id"]).agg(luck=("luck", "sum")).reset_index()
        sw = sw.reindex(sw.luck.abs().sort_values(ascending=False).index).head(2)
        swing = []
        for r in sw.itertuples():
            t3 = x[(x.shooter == r.shooter) & (x.kind == "3")]
            swing.append({"player_id": int(r.shooter), "team_id": int(r.team_id), "luck_pts": round(float(r.luck), 1),
                          "three": [int(t3.made.sum()), int(len(t3))]})
        out.append({
            "game_id": int(g.game_id), "date": g.date, "season_type": int(g.season_type), "competition_type": g.competition_type,
            "home_id": int(g.home_id), "away_id": int(g.away_id), "home_pts": int(g.home_pts), "away_pts": int(g.away_pts),
            "periods": int(g.periods), "home_chance": round(home, 4), "home_chance_raw": round(raw, 4),
            "winner_chance": round(winner_chance, 4), "label": label(winner_chance),
            "teams": teams, "swing": swing,
        })
    return out


# ---------------------------------------------------------------- players


def player_table(shots: pd.DataFrame, appearances: pd.DataFrame, season: int, rates: dict) -> pd.DataFrame:
    """This season so far, per player: games, points, makes/attempts by kind, levels, projection."""
    cur = shots[(shots.season == season) & shots.regular & (shots.zone != "heave")]
    reg_games = set(shots[(shots.season == season) & shots.regular].game_id)
    ap = appearances[appearances.game_id.isin(reg_games) & (appearances.minutes.fillna(0) > 0)]
    gp = ap.groupby("player_id").game_id.nunique()
    shot_games = cur.groupby("shooter").game_id.nunique()
    d = pd.DataFrame({"gp": gp}).join(pd.DataFrame({"sg": shot_games}), how="outer").fillna(0)
    d["gp"] = d[["gp", "sg"]].max(axis=1).astype(int)
    # heaves count toward his points, never his level
    allpts = shots[(shots.season == season) & shots.regular]
    d["pts"] = (allpts.made * allpts.kind.map(VALUE)).groupby(allpts.shooter).sum().reindex(d.index).fillna(0)
    luck = pd.Series(0.0, index=d.index)
    for k in KINDS:
        lv = current_levels(shots, season, k, rates[k])
        lv = lv.reindex(d.index)
        d[f"m{k}"] = lv.m.fillna(0).astype(int)
        d[f"a{k}"] = lv.a.fillna(0).astype(int)
        mu = rates[k]
        d[f"level{k}"] = lv.level.fillna(mu)
        d[f"lo{k}"] = lv.lo.fillna(mu)
        d[f"hi{k}"] = lv.hi.fillna(mu)
        luck += VALUE[k] * (d[f"m{k}"] - d[f"a{k}"] * d[f"level{k}"])
    # his latest game (played minutes or took a shot), so the lists can leave out players who are out
    dates = shots[(shots.season == season)].drop_duplicates("game_id").set_index("game_id").date
    last_ap = ap.assign(date=ap.game_id.map(dates)).groupby("player_id").date.max()
    last_shot = cur.groupby("shooter").date.max()
    both = pd.concat([last_ap, last_shot], axis=1).fillna("")  # ISO dates: text order is date order
    d["last_date"] = both.max(axis=1).reindex(d.index).replace("", None)
    d = d[d.gp > 0].copy()
    d["ppg"] = d.pts / d.gp
    d["luck_ppg"] = luck.reindex(d.index) / d.gp
    d["proj_ppg"] = d.ppg - d.luck_ppg
    d["fga_pg"] = (d.a2 + d.a3) / d.gp
    d.index.name = "player_id"
    return d.drop(columns=["sg"]).reset_index()


def hot_cold(table: pd.DataFrame, n: int = 10, min_fga: float = 8.0, active_days: int = 14) -> dict:
    """
    Heat check (shooting above his level: expect a cool-off) and Defrost (below: bounce-back).
    Regulars only: 8+ shots a game, and at least 5 games or a quarter of the most games anyone
    has played, whichever is more (so 20 games by the end of the season). And only players who
    have played in the last `active_days` days: an injured player's numbers freeze, and he'd sit
    on the list for weeks with a bounce-back he can't have (found by the receipts, 9 Oct 2026).
    """
    min_games = max(5, int(round(0.25 * table.gp.max()))) if len(table) else 5
    x = table[(table.gp >= min_games) & (table.fga_pg >= min_fga)]
    if len(x) and "last_date" in x:
        latest = pd.to_datetime(table.last_date).max()
        x = x[pd.to_datetime(x.last_date) >= latest - pd.Timedelta(days=active_days)]
    gap = x.proj_ppg - x.ppg
    return {
        "heat_check": x.loc[gap.sort_values().index].head(n)[gap.sort_values().head(n) < 0],
        "defrost": x.loc[gap.sort_values(ascending=False).index].head(n)[gap.sort_values(ascending=False).head(n) > 0],
        "min_games": min_games, "min_fga": min_fga, "eligible": int(len(x)),
    }


def season_of(date_str: str) -> int:
    """ESPN season year (the year it ends) for a date: October onwards belongs to next year."""
    y, m = int(date_str[:4]), int(date_str[5:7])
    return y + 1 if m >= 8 else y


def safe(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v
