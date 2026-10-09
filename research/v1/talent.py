"""
Check 1: how to estimate a player's true shooting level before each game.

  level = (w·makes last season + w²·two seasons ago + w³·three ago + this season's makes before
           the game + k·league rate) / (same with attempts + k)

k (prior weight, in attempts) and w (how much each older season counts) are fitted by held-out
log loss: every shot in 2017-18 to 2024-25 is predicted with only what was known before its game,
and the (k, w) that predicts makes best wins. Compared with simpler rules (league average, the
player's raw rate this season, raw rate over four seasons). Then two adjustments are tested the
same way: where a 3 was taken (corner / above the break), and home vs away.

2025-26 is held out and not used here. Writes results/v1-talent.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pilot"))
import parse  # noqa: E402

EVAL = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
K_GRID = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 350, 400, 500, 650, 800, 1000]
W_GRID = [0.0, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load() -> pd.DataFrame:
    shots, games = [], []
    for y in range(2015, 2026):  # 2026 (2025-26) is the holdout
        s, g = parse.parse(y)
        shots.append(s)
        games.append(g)
    s = pd.concat(shots, ignore_index=True)
    g = pd.concat(games, ignore_index=True)[["game_id", "home_id"]]
    s = s.merge(g, on="game_id")
    s["home"] = (s.team_id == s.home_id).astype(int)
    s["shooter"] = s.shooter.astype("Int64")
    return s[s.shooter.notna()]


def player_games(s: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Per player-game makes/attempts, plus this season's makes/attempts before the game."""
    x = s[(s.kind == kind) & (s.zone != "heave")]
    pg = x.groupby(["season", "shooter", "date", "game_id"]).made.agg(m="sum", a="count").reset_index()
    pg = pg.sort_values(["season", "shooter", "date", "game_id"])
    g = pg.groupby(["season", "shooter"])
    pg["m_bef"] = g.m.cumsum() - pg.m
    pg["a_bef"] = g.a.cumsum() - pg.a
    tot = x.groupby(["season", "shooter"]).made.agg(M="sum", A="count").reset_index()
    for d in (1, 2, 3):
        prev = tot.assign(season=tot.season + d).rename(columns={"M": f"M{d}", "A": f"A{d}"})
        pg = pg.merge(prev, on=["season", "shooter"], how="left")
    pg[[f"{c}{d}" for c in "MA" for d in (1, 2, 3)]] = pg[[f"{c}{d}" for c in "MA" for d in (1, 2, 3)]].fillna(0)
    league = x.groupby("season").made.mean()
    pg["mu"] = pg.season.map(lambda y: league.get(y - 1, league.get(y)))
    return pg[pg.season.isin(EVAL)].reset_index(drop=True)


def logloss(pg: pd.DataFrame, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(pg.m * np.log(p) + (pg.a - pg.m) * np.log(1 - p))
    return float(ll.sum() / pg.a.sum())


def level(pg: pd.DataFrame, k: float, w: float) -> np.ndarray:
    pm = sum(w**d * pg[f"M{d}"] for d in (1, 2, 3))
    pa = sum(w**d * pg[f"A{d}"] for d in (1, 2, 3))
    return ((pm + pg.m_bef + k * pg.mu) / (pa + pg.a_bef + k)).to_numpy()


def fit(pg: pd.DataFrame) -> dict:
    grid = {(k, w): logloss(pg, level(pg, k, w)) for k in K_GRID for w in W_GRID}
    (k, w), best = min(grid.items(), key=lambda kv: kv[1])
    # baselines
    raw_season = np.where(pg.a_bef > 0, pg.m_bef / pg.a_bef.clip(lower=1), pg.mu)
    four = (pg.M1 + pg.M2 + pg.M3 + pg.m_bef)
    four_a = (pg.A1 + pg.A2 + pg.A3 + pg.a_bef)
    raw_four = np.where(four_a > 0, four / four_a.clip(lower=1), pg.mu)
    base = {
        "league average": logloss(pg, pg.mu.to_numpy()),
        "player's raw rate this season": logloss(pg, raw_season),
        "player's raw rate, 4 seasons": logloss(pg, raw_four),
        "this season only, shrunk (best k, w=0)": min(v for (kk, ww), v in grid.items() if ww == 0.0),
        "fitted (k, w)": best,
    }
    # how flat is the optimum? log loss within 0.0001 of the best
    near = sorted({kk for (kk, ww), v in grid.items() if v - best < 1e-4})
    return {"k": k, "w": w, "logloss": best, "baselines": base, "k_within_0.0001": [near[0], near[-1]], "shots": int(pg.a.sum())}


def zone_home_test(s: pd.DataFrame, pg3: pd.DataFrame, k: float, w: float) -> dict:
    """Does shifting a player's 3P level by zone (corner / above the break) or by home/away help?"""
    lv = pg3.assign(lv=level(pg3, k, w))[["game_id", "shooter", "lv", "season"]]
    x = s[(s.kind == "3") & (s.zone != "heave") & s.season.isin(EVAL)].merge(lv, on=["game_id", "shooter", "season"])
    logit = lambda p: np.log(p / (1 - p))
    inv = lambda z: 1 / (1 + np.exp(-z))
    y = x.made.to_numpy()

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    out = {"base": ll(x.lv.to_numpy())}
    # zone shift: fitted on earlier seasons only (league corner and above-break rates vs overall)
    prev = s[(s.kind == "3") & (s.zone.isin(["corner", "above"]))]
    rate = prev.groupby(["season", "zone"]).made.mean().unstack()
    allr = prev.groupby("season").made.mean()
    shift = {(yy, z): logit(rate.loc[yy - 1, z]) - logit(allr.loc[yy - 1]) for yy in EVAL for z in ("corner", "above")}
    zs = np.array([shift[(a, b)] for a, b in zip(x.season, x.zone)])
    out["zone shift"] = ll(inv(logit(x.lv.to_numpy()) + zs))
    # home shift: league home-minus-away make rate (logit) from earlier seasons
    hs = {}
    for yy in EVAL:
        p = s[(s.kind == "3") & (s.zone != "heave") & (s.season == yy - 1)]
        hr = p.groupby("home").made.mean()
        hs[yy] = (logit(hr[1]) - logit(hr[0])) / 2
    h = np.array([hs[a] * (1 if b else -1) for a, b in zip(x.season, x.home)])
    out["home shift"] = ll(inv(logit(x.lv.to_numpy()) + h))
    out["zone + home"] = ll(inv(logit(x.lv.to_numpy()) + zs + h))
    out["home_logit_half_gap_by_season"] = {int(k): round(float(v), 4) for k, v in hs.items()}
    out["shots"] = int(len(x))
    return out


def home_gap(s: pd.DataFrame) -> dict:
    """Raw home vs away make rates, all seasons (context for the home shift)."""
    out = {}
    for kind in ("3", "FT", "2"):
        x = s[(s.kind == kind) & (s.zone != "heave")]
        t = x.groupby("home").made.agg(["mean", "count"])
        out[kind] = {"home": round(float(t.loc[1, "mean"]), 4), "away": round(float(t.loc[0, "mean"]), 4), "n": int(t["count"].sum())}
    return out


if __name__ == "__main__":
    s = load()
    R = {"home_vs_away_raw": home_gap(s)}
    print("Home vs away make rates:", R["home_vs_away_raw"])
    pgs = {}
    for kind in ("3", "FT", "2"):
        pg = player_games(s, kind)
        pgs[kind] = pg
        r = fit(pg)
        R[kind] = r
        print(f"\n{kind}: k={r['k']} w={r['w']}  (k within 0.0001 of best: {r['k_within_0.0001']}), {r['shots']:,} shots")
        for name, v in r["baselines"].items():
            print(f"   {name:42} {v:.5f}")
    zt = zone_home_test(s, pgs["3"], R["3"]["k"], R["3"]["w"])
    R["3"]["adjustments"] = zt
    print("\n3PT adjustments (log loss per shot):", {k: v for k, v in zt.items() if isinstance(v, float)})
    print("home half-gap by season (logit):", zt["home_logit_half_gap_by_season"])
    out = parse.ROOT / "results" / "v1-talent.json"
    out.write_text(json.dumps(R, indent=2, default=float))
    print("Saved", out)
