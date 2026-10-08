"""
Pilot luck model: make probabilities for every shot (four variants), the exact chance of every
score, and each game's "deserved" win chance.

Variants (what is treated as luck, and at what probability):
  A  3s + FTs at the shooter's shrunk rate; 2s as they happened          (Josh's proposal)
  B  A, plus every 2 at its zone's league rate                           (all shot-making)
  C  3s at the zone's league rate, FTs at the league rate; 2s as they happened (shooter-blind)
  D  every shot at its zone's league rate                                (shooter-blind, all shots)

Everything known before a game only: shooter rates come from earlier seasons (decayed) plus this
season's games on earlier dates; zone rates from the previous season.
Pilot simplifications (fixed properly in the full study): no offensive-rebound value for misses,
no special handling of shooting fouls, ties after regulation count as half a win.
"""
import numpy as np
import pandas as pd

DECAY = 0.7  # weight on each earlier season (pilot guess; the full study fits it)


def shrinkage_k(shots: pd.DataFrame, kind: str, min_att: int = 1) -> dict:
    """
    Beta-binomial prior weight k (in attempts) from player-season totals, by method of moments:
    the between-player variance of true rates is what's left of the observed variance after
    subtracting the binomial noise each player's attempts imply. k is the number of attempts at
    which a player's own rate and the league mean get equal weight (the "stabilisation point").
    """
    x = shots[(shots.kind == kind) & (shots.zone != "heave")]
    t = x.groupby("shooter").made.agg(["sum", "count"])
    t = t[t["count"] >= min_att]
    n, r = t["count"].to_numpy(float), (t["sum"] / t["count"]).to_numpy()
    mu = t["sum"].sum() / n.sum()
    w = n / n.sum()
    tau2 = np.sum(w * ((r - mu) ** 2 - mu * (1 - mu) / n) / (1 - 1 / n.clip(min=2)))
    return {"mu": mu, "tau2": tau2, "k": mu * (1 - mu) / tau2 - 1 if tau2 > 0 else float("inf"), "players": len(t)}


def shooter_rates(all_shots: pd.DataFrame, season: int, kind: str, k: float, mu: float) -> pd.Series:
    """
    Each (game, shooter) in `season`: the shrunk make rate known before that game, as
    (prior makes + this season's earlier makes + k·mu) / (prior attempts + earlier attempts + k).
    """
    x = all_shots[(all_shots.kind == kind) & (all_shots.zone != "heave")]
    prior = x[(x.season < season) & (x.season >= season - 3)].assign(
        w=lambda d: DECAY ** (season - d.season)
    )
    pm = (prior.made * prior.w).groupby(prior.shooter).sum()
    pa = prior.w.groupby(prior.shooter).sum()

    cur = x[x.season == season]
    per_game = cur.groupby(["shooter", "date", "game_id"]).made.agg(["sum", "count"]).reset_index()
    per_game = per_game.sort_values(["shooter", "date"])
    # this season before the game's date (doubleheaders don't exist; same-date rows are one game)
    per_game["m_before"] = per_game.groupby("shooter")["sum"].cumsum() - per_game["sum"]
    per_game["a_before"] = per_game.groupby("shooter")["count"].cumsum() - per_game["count"]
    m = per_game.m_before + per_game.shooter.map(pm).fillna(0)
    a = per_game.a_before + per_game.shooter.map(pa).fillna(0)
    per_game["p"] = (m + k * mu) / (a + k)
    return per_game.set_index(["game_id", "shooter"]).p


def make_probabilities(all_shots: pd.DataFrame, season: int, k3: float, kft: float) -> pd.DataFrame:
    """The season's shots with p_A..p_D columns (NaN = not random in that variant)."""
    prev = all_shots[all_shots.season == season - 1]
    zone_rate = prev.groupby("zone").made.mean()
    ft_league = zone_rate["FT"]
    mu3 = prev[(prev.kind == "3") & (prev.zone != "heave")].made.mean()

    s = all_shots[all_shots.season == season].copy()
    key = pd.MultiIndex.from_frame(s[["game_id", "shooter"]])
    p3 = shooter_rates(all_shots, season, "3", k3, mu3).reindex(key).to_numpy()
    pft = shooter_rates(all_shots, season, "FT", kft, ft_league).reindex(key).to_numpy()
    pz = s.zone.map(zone_rate).to_numpy()

    is3, isft, is2 = (s.kind == "3").to_numpy(), (s.kind == "FT").to_numpy(), (s.kind == "2").to_numpy()
    heave = (s.zone == "heave").to_numpy()
    # a shooter's 3 adjusted for zone: shift his rate by the zone's gap to the overall 3P rate (logit scale)
    logit = lambda p: np.log(p / (1 - p))
    inv = lambda z: 1 / (1 + np.exp(-z))
    p3z = inv(logit(np.clip(p3, 0.01, 0.99)) + logit(pz.clip(0.01, 0.99)) - logit(mu3))
    shooter3 = np.where(heave, pz, p3z)

    nan = np.full(len(s), np.nan)
    s["p_A"] = np.where(is3, shooter3, np.where(isft, pft, nan))
    s["p_B"] = np.where(is2, pz, s.p_A)
    s["p_C"] = np.where(is3 | isft, pz, nan)
    s["p_D"] = pz
    return s


def points_pmf(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exact distribution of sum(v_i * Bernoulli(p_i)) as an array indexed by points."""
    pmf = np.zeros(int(v.sum()) + 1)
    pmf[0] = 1.0
    top = 0
    for pi, vi in zip(p, v):
        vi = int(vi)
        nxt = pmf.copy() * (1 - pi)
        nxt[vi : top + vi + 1] += pmf[: top + 1] * pi
        pmf = nxt
        top += vi
    return pmf


def game_chances(home_fixed: int, home_p, home_v, away_fixed: int, away_p, away_v) -> dict:
    """Home side's chances of winning / tying in regulation, plus margin mean and variance."""
    a = points_pmf(home_p, home_v)
    b = points_pmf(away_p, away_v)
    d = np.convolve(a, b[::-1])  # index i ↔ (home random pts − away random pts) = i − (len(b) − 1)
    margins = np.arange(len(d)) - (len(b) - 1) + home_fixed - away_fixed
    win, tie = d[margins > 0].sum(), d[margins == 0].sum()
    mean = float((margins * d).sum())
    var = float(((margins - mean) ** 2 * d).sum())
    return {"win": float(win), "tie": float(tie), "mean": mean, "var": var}


VARIANTS = ["A", "B", "C", "D"]
VALUE = {"2": 2, "3": 3, "FT": 1}


def season_games(shots_p: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    One row per game: for each variant, the home side's deserved win chance (regulation; a tie
    counts half), expected regulation margin and its variance, and full-game expected points for
    both sides (for luck-adjusted ratings).
    """
    s = shots_p.assign(v=shots_p.kind.map(VALUE))
    reg = s[s.period <= 4]
    rows = []
    by_game_reg = dict(tuple(reg.groupby("game_id")))
    by_game_all = dict(tuple(s.groupby("game_id")))
    for g in games.itertuples():
        r = by_game_reg.get(g.game_id)
        al = by_game_all.get(g.game_id)
        if r is None:
            continue
        out = {"game_id": g.game_id}
        for var in VARIANTS:
            col = f"p_{var}"
            rnd = r[r[col].notna()]
            h, a_ = rnd[rnd.team_id == g.home_id], rnd[rnd.team_id == g.away_id]
            h_fixed = g.home_reg - int((h.made * h.v).sum())
            a_fixed = g.away_reg - int((a_.made * a_.v).sum())
            c = game_chances(h_fixed, h[col].to_numpy(), h.v.to_numpy(), a_fixed, a_[col].to_numpy(), a_.v.to_numpy())
            out[f"win_{var}"] = c["win"]
            out[f"tie_{var}"] = c["tie"]
            out[f"deserved_{var}"] = c["win"] + c["tie"] / 2
            out[f"mean_{var}"] = c["mean"]
            out[f"var_{var}"] = c["var"]
            # full game (incl. OT) expected points: actual minus luck on the random shots
            ra = al[al[col].notna()]
            luck = ((ra.made - ra[col]) * ra.v).groupby(ra.team_id).sum()
            out[f"xpts_home_{var}"] = g.home_pts - luck.get(g.home_id, 0.0)
            out[f"xpts_away_{var}"] = g.away_pts - luck.get(g.away_id, 0.0)
        rows.append(out)
    return games.merge(pd.DataFrame(rows), on="game_id")
