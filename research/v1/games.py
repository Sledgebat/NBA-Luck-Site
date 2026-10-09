"""
Check 2: game verdicts with the fitted shooting levels, and how to make them honest.

Each game: 3s and free throws are luck, each at the shooter's level known before the game (fitted
k and w from talent.py; corner/above-break shift for 3s; heaves at the league heave rate); 2s and
everything else as it happened. Exact score distributions, regulation only, a tie after
regulation counts half (overtime is close to a coin flip).

Versions compared on 2017-18 to 2024-25 (2025-26 stays held out):
  pilot   the pilot's settings (k=300, w=0.7, no zone shift)
  v1      fitted k, w + zone shift
  v1+home v1 plus the league's home shooting edge on 3s and FTs
and then a recalibration fitted on other seasons: logit(p') = a + b·logit(p) for the home side,
which can absorb the home edge (a) and any overconfidence (b < 1). Scored against the actual
regulation result (win 1, tie ½, loss 0) with leave-one-season-out fitting.

Writes results/v1-games.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pilot"))
import model  # noqa: E402
import parse  # noqa: E402

SEASONS = [2017, 2018, 2019, 2022, 2023, 2024, 2025]
FIT = json.loads((parse.ROOT / "results" / "v1-talent.json").read_text())
logit = lambda p: np.log(p / (1 - p))
inv = lambda z: 1 / (1 + np.exp(-z))


def levels(all_shots, season, kind, k, w):
    """Shooter level before each game (same formula as talent.py), keyed by (game_id, shooter)."""
    x = all_shots[(all_shots.kind == kind) & (all_shots.zone != "heave")]
    league = x.groupby("season").made.mean()
    mu = league.get(season - 1, league.get(season))
    prior = x[(x.season < season) & (x.season >= season - 3)]
    wt = w ** (season - prior.season)
    pm = (prior.made * wt).groupby(prior.shooter).sum()
    pa = wt.groupby(prior.shooter).sum()
    cur = x[x.season == season]
    pg = cur.groupby(["shooter", "date", "game_id"]).made.agg(["sum", "count"]).reset_index().sort_values(["shooter", "date", "game_id"])
    pg["mb"] = pg.groupby("shooter")["sum"].cumsum() - pg["sum"]
    pg["ab"] = pg.groupby("shooter")["count"].cumsum() - pg["count"]
    m = pg.mb + pg.shooter.map(pm).fillna(0)
    a = pg.ab + pg.shooter.map(pa).fillna(0)
    pg["p"] = (m + k * mu) / (a + k)
    return pg.set_index(["game_id", "shooter"]).p, mu


def shot_probs(all_shots, games_all, season, version):
    s = all_shots[all_shots.season == season].merge(games_all[["game_id", "home_id"]], on="game_id")
    s["home"] = (s.team_id == s.home_id).astype(int)
    prev = all_shots[all_shots.season == season - 1]
    zone_rate = prev.groupby("zone").made.mean()
    key = pd.MultiIndex.from_frame(s[["game_id", "shooter"]])
    if version == "pilot":
        k3, w3, kft, wft = 300, 0.7, 25, 0.7
    else:
        k3, w3 = FIT["3"]["k"], FIT["3"]["w"]
        kft, wft = FIT["FT"]["k"], FIT["FT"]["w"]
    l3, mu3 = levels(all_shots, season, "3", k3, w3)
    lft, _ = levels(all_shots, season, "FT", kft, wft)
    p3 = l3.reindex(key).to_numpy()
    pft = lft.reindex(key).to_numpy()
    z3 = np.zeros(len(s))
    if version != "pilot":
        corner_above = prev[prev.zone.isin(["corner", "above"])]
        allr = corner_above.made.mean()
        zr = corner_above.groupby("zone").made.mean()
        z3 = s.zone.map({z: logit(zr[z]) - logit(allr) for z in ("corner", "above")}).fillna(0).to_numpy()
    h3 = hft = np.zeros(len(s))
    if version == "v1+home":
        def half_gap(kind):
            x = prev[(prev.kind == kind) & (prev.zone != "heave")].merge(games_all[["game_id", "home_id"]], on="game_id")
            r = x.groupby(x.team_id == x.home_id).made.mean()
            return (logit(r[True]) - logit(r[False])) / 2
        sign = np.where(s.home == 1, 1, -1)
        h3, hft = half_gap("3") * sign, half_gap("FT") * sign
    is3 = (s.kind == "3").to_numpy()
    isft = (s.kind == "FT").to_numpy()
    heave = (s.zone == "heave").to_numpy()
    p3x = inv(logit(np.clip(p3, 0.01, 0.99)) + z3 + h3)
    pftx = inv(logit(np.clip(pft, 0.01, 0.99)) + hft)
    s["p"] = np.where(is3, np.where(heave, zone_rate.get("heave", 0.03), p3x), np.where(isft, pftx, np.nan))
    return s


def game_rows(s, games):
    s = s[s.p.notna()].assign(v=lambda d: d.kind.map({"3": 3, "FT": 1}))
    reg = dict(tuple(s[s.period <= 4].groupby("game_id")))
    out = []
    for g in games.itertuples():
        r = reg.get(g.game_id)
        if r is None:
            continue
        h, a = r[r.team_id == g.home_id], r[r.team_id == g.away_id]
        c = model.game_chances(
            g.home_reg - int((h.made * h.v).sum()), h.p.to_numpy(), h.v.to_numpy(),
            g.away_reg - int((a.made * a.v).sum()), a.p.to_numpy(), a.v.to_numpy(),
        )
        out.append({"game_id": g.game_id, "p": c["win"] + c["tie"] / 2, "tie": c["tie"], "mean": c["mean"], "var": c["var"]})
    return games.merge(pd.DataFrame(out), on="game_id")


def score(df, p):
    y = df.reg_result.to_numpy()
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    bins = pd.cut(p, [0, 0.05, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 0.95, 1], include_lowest=True)
    t = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).agg(n=("y", "size"), pred=("p", "mean"), act=("y", "mean"))
    win = np.where(df.home_pts > df.away_pts, p, 1 - p)
    return {
        "brier": float(((p - y) ** 2).mean()),
        "logloss": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
        "mean_pred": float(p.mean()), "mean_actual": float(y.mean()),
        "table": [[str(i), int(r.n), round(float(r.pred), 3), round(float(r.act), 3)] for i, r in t.iterrows()],
        "winner_under_35": float((win < 0.35).mean()), "winner_under_25": float((win < 0.25).mean()), "winner_35_65": float(((win >= 0.35) & (win < 0.65)).mean()),
    }


def recalibrate(df, col):
    """Leave-one-season-out fit of logit(p') = a + b·logit(p); returns recalibrated p and the all-season (a, b)."""
    x = logit(np.clip(df[col].to_numpy(), 1e-4, 1 - 1e-4))
    y = df.reg_result.to_numpy()

    def fit(xx, yy):
        f = lambda ab: -np.sum(yy * np.log(inv(ab[0] + ab[1] * xx)) + (1 - yy) * np.log(1 - inv(ab[0] + ab[1] * xx)))
        return minimize(f, [0.0, 1.0], method="Nelder-Mead").x

    out = np.zeros(len(df))
    for s in df.season.unique():
        tr, te = df.season.to_numpy() != s, df.season.to_numpy() == s
        a, b = fit(x[tr], y[tr])
        out[te] = inv(a + b * x[te])
    return out, [float(v) for v in fit(x, y)]


if __name__ == "__main__":
    all_shots = pd.concat([parse.parse(y)[0] for y in range(2015, 2026)], ignore_index=True)
    games_all = pd.concat([parse.parse(y)[1] for y in range(2015, 2026)], ignore_index=True)
    rows = []
    for season in SEASONS:
        games = games_all[games_all.season == season]
        per = {}
        for version in ("pilot", "v1", "v1+home"):
            per[version] = game_rows(shot_probs(all_shots, games_all, season, version), games).set_index("game_id")
        base = per["v1"].copy()
        for version, d in per.items():
            base[f"p_{version}"] = d.p
            base[f"tie_{version}"] = d.tie
            base[f"z_{version}"] = ((d.home_reg - d.away_reg) - d["mean"]) / np.sqrt(d["var"])
        rows.append(base.reset_index())
        print(season, len(base), flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["reg_result"] = np.where(df.home_reg > df.away_reg, 1.0, np.where(df.home_reg < df.away_reg, 0.0, 0.5))
    R = {"games": len(df), "seasons": SEASONS, "versions": {}}
    actual_ties = float((df.home_reg == df.away_reg).mean())
    for version in ("pilot", "v1", "v1+home"):
        r = score(df, df[f"p_{version}"].to_numpy())
        r["z_variance"] = float(df[f"z_{version}"].var())
        r["ties_actual_per_predicted"] = actual_ties / float(df[f"tie_{version}"].mean())
        rc, ab = recalibrate(df, f"p_{version}")
        r["recalibrated"] = score(df, rc)
        r["recalibration_ab_all_seasons"] = ab
        R["versions"][version] = r
        print(f"\n{version}: Brier {r['brier']:.4f} → {r['recalibrated']['brier']:.4f} recalibrated (a, b = {ab[0]:.3f}, {ab[1]:.3f}); "
              f"mean pred {r['mean_pred']:.3f} vs actual {r['mean_actual']:.3f}; var(z) {r['z_variance']:.2f}; ties ×{r['ties_actual_per_predicted']:.2f}")
        print("   raw:          " + "  ".join(f"[{p}→{a}, n={n}]" for _, n, p, a in r["table"]))
        print("   recalibrated: " + "  ".join(f"[{p}→{a}, n={n}]" for _, n, p, a in r["recalibrated"]["table"]))
        print(f"   winner <35%: raw {r['winner_under_35']:.3f}, recalibrated {r['recalibrated']['winner_under_35']:.3f};  <25%: {r['winner_under_25']:.3f} / {r['recalibrated']['winner_under_25']:.3f}")
    df.to_parquet(parse.DERIVED / "v1_games.parquet")
    (parse.ROOT / "results" / "v1-games.json").write_text(json.dumps(R, indent=2))
