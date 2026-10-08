"""
Pilot go/no-go analysis. Reads data/derived/pilot_games.parquet (run_games.py), writes
results/pilot.json and prints the tables used in results/pilot-report.md.

  1. How often does shooting luck flip a game? (label rates)
  2. Are the probabilities honest? (calibration, spread of outcomes)
  3. How big is luck over a season, in wins?
  4. Does luck-adjusted margin predict future results better than raw margin?
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import parse

V = ["A", "B", "C", "D"]
NAMES = {
    "A": "3s+FTs, shooter rates (proposal)",
    "B": "all shots, shooter 3s/FTs + zone 2s",
    "C": "3s+FTs, league rates",
    "D": "all shots, league/zone rates",
}
rng = np.random.default_rng(7)
g = pd.read_parquet(parse.DERIVED / "pilot_games.parquet")
g["home_win"] = (g.home_pts > g.away_pts).astype(int)
g["reg_result"] = np.where(g.home_reg > g.away_reg, 1.0, np.where(g.home_reg < g.away_reg, 0.0, 0.5))
R = {"games": len(g), "seasons": sorted(g.season.unique().tolist())}

# ---------- 1. label rates ----------
print("\n1. WINNER'S DESERVED CHANCE (share of games)")
labels = {}
for v in V:
    w = np.where(g.home_win == 1, g[f"deserved_{v}"], 1 - g[f"deserved_{v}"])
    labels[v] = {
        "winner_under_50": float((w < 0.5).mean()),
        "winner_under_35": float((w < 0.35).mean()),
        "winner_under_25": float((w < 0.25).mean()),
        "winner_35_65": float(((w >= 0.35) & (w < 0.65)).mean()),
        "winner_65_plus": float((w >= 0.65).mean()),
        "winner_90_plus": float((w >= 0.9).mean()),
        "median_winner_chance": float(np.median(w)),
    }
    print(f"  {v} {NAMES[v]:40} " + "  ".join(f"{k}={x:.3f}" for k, x in labels[v].items()))
R["labels"] = labels

# ---------- 2. calibration ----------
print("\n2. CALIBRATION: deserved (home) vs actual regulation result; z = (actual − expected margin)/sd")
cal = {}
for v in V:
    p = g[f"deserved_{v}"]
    bins = pd.cut(p, [0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0], include_lowest=True)
    t = g.groupby(bins, observed=True).agg(n=("reg_result", "size"), predicted=(f"deserved_{v}", "mean"), actual=("reg_result", "mean"))
    z = ((g.home_reg - g.away_reg) - g[f"mean_{v}"]) / np.sqrt(g[f"var_{v}"])
    tie_ratio = float((g.home_reg == g.away_reg).mean() / g[f"tie_{v}"].mean())
    cal[v] = {
        "table": [{"bin": str(i), **{k: float(x) for k, x in r.items()}} for i, r in t.iterrows()],
        "mean_predicted": float(p.mean()),
        "mean_actual": float(g.reg_result.mean()),
        "z_variance": float(z.var()),
        "actual_ties_per_predicted": tie_ratio,
        "brier_vs_reg_result": float(((p - g.reg_result) ** 2).mean()),
    }
    print(f"  {v}: mean pred {p.mean():.3f} vs actual {g.reg_result.mean():.3f};  var(z) = {z.var():.2f} (1 = right spread);  ties actual/predicted = {tie_ratio:.2f}")
    print("     " + "  ".join(f"[{r.predicted:.2f}→{r.actual:.2f}, n={int(r.n)}]" for _, r in t.iterrows()))
R["calibration"] = cal

# ---------- team-game long table ----------
def long_table(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = pd.DataFrame(
            {
                "game_id": g.game_id,
                "season": g.season,
                "date": g.date,
                "team": g[f"{side}_abbrev"],
                "team_id": g[f"{side}_id"],
                "home": int(side == "home"),
                "margin": g[f"{side}_pts"] - g[f"{opp}_pts"],
                "win": (g[f"{side}_pts"] > g[f"{opp}_pts"]).astype(int),
            }
        )
        for v in V:
            d[f"xm_{v}"] = g[f"xpts_{side}_{v}"] - g[f"xpts_{opp}_{v}"]
            d[f"dw_{v}"] = g[f"deserved_{v}"] if side == "home" else 1 - g[f"deserved_{v}"]
            # one side only: the opponents' shooting luck removed (defence), or the team's own (offence)
            d[f"xdef_{v}"] = g[f"{side}_pts"] - g[f"xpts_{opp}_{v}"]
            d[f"xoff_{v}"] = g[f"xpts_{side}_{v}"] - g[f"{opp}_pts"]
        rows.append(d)
    return pd.concat(rows).sort_values(["season", "team_id", "date"]).reset_index(drop=True)


t = long_table(g)
t["n_before"] = t.groupby(["season", "team_id"]).cumcount()

# ---------- 3. luck over a season ----------
print("\n3. LUCK IN WINS PER TEAM-SEASON (actual wins − deserved wins)")
season_luck = {}
for v in V:
    s = t.groupby(["season", "team"]).agg(w=("win", "sum"), d=(f"dw_{v}", "sum"), gp=("win", "size"))
    s["luck"] = s.w - s.d
    # each game's deserved chance is a coin with that weight: the model's own SD of luck wins
    s["model_sd"] = t.assign(q=t[f"dw_{v}"] * (1 - t[f"dw_{v}"])).groupby(["season", "team"]).q.sum() ** 0.5
    top = s.sort_values("luck")
    season_luck[v] = {
        "sd": float(s.luck.std()),
        "model_sd_mean": float(s.model_sd.mean()),
        "share_abs_ge_3": float((s.luck.abs() >= 3).mean()),
        "share_abs_ge_5": float((s.luck.abs() >= 5).mean()),
        "unluckiest": [f"{a} {b}: {r.luck:+.1f} ({int(r.w)} W vs {r.d:.1f})" for (a, b), r in top.head(5).iterrows()],
        "luckiest": [f"{a} {b}: {r.luck:+.1f} ({int(r.w)} W vs {r.d:.1f})" for (a, b), r in top.tail(5)[::-1].iterrows()],
    }
    L = season_luck[v]
    print(f"  {v}: SD {L['sd']:.2f} wins (model's own SD {L['model_sd_mean']:.2f}); |luck| ≥ 3: {L['share_abs_ge_3']:.0%}, ≥ 5: {L['share_abs_ge_5']:.0%}")
    print(f"     luckiest: {L['luckiest'][:3]}\n     unluckiest: {L['unluckiest'][:3]}")
R["season_luck"] = season_luck

# ---------- 4. prediction ----------
# cumulative means of each predictor over the team's earlier games
preds = {"raw": "margin", "wpct": "win", **{f"adj_{v}": f"xm_{v}" for v in V}, "dwpct_A": "dw_A", "dwpct_B": "dw_B",
         "def_A": "xdef_A", "off_A": "xoff_A", "def_C": "xdef_C"}
grp = t.groupby(["season", "team_id"])
for name, col in preds.items():
    t[f"r_{name}"] = grp[col].transform(lambda x: x.shift().expanding().mean())

# split-half reliability of each per-game measure (odd vs even games, full season)
print("\n4a. SPLIT-HALF RELIABILITY (odd vs even games, correlation across team-seasons)")
rel = {}
for name, col in preds.items():
    odd = t[t.n_before % 2 == 1].groupby(["season", "team_id"])[col].mean()
    even = t[t.n_before % 2 == 0].groupby(["season", "team_id"])[col].mean()
    rel[name] = float(np.corrcoef(odd, even.reindex(odd.index))[0, 1])
print("  " + "  ".join(f"{k}={x:.3f}" for k, x in rel.items()))
R["split_half"] = rel


def fit_logit(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    f = lambda b: np.sum(np.logaddexp(0, X @ b) - y * (X @ b))
    return minimize(f, np.zeros(X.shape[1]), method="BFGS").x


def predict_logit(b, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-(X @ b)))


home = t[t.home == 1].set_index("game_id")
away = t[t.home == 0].set_index("game_id")
G = home.join(away, lsuffix="_h", rsuffix="_a")
G = G.join(g.set_index("game_id")[["spread_home"]])
G = G[(G.n_before_h >= 10) & (G.n_before_a >= 10)].copy()
y = G.win_h.to_numpy()
seasons = G.season_h.to_numpy()


def loso(features):
    X = np.column_stack([G[f"r_{f}_h"] - G[f"r_{f}_a"] for f in features])
    p = np.zeros(len(G))
    for s in np.unique(seasons):
        tr, te = seasons != s, seasons == s
        p[te] = predict_logit(fit_logit(X[tr], y[tr]), X[te])
    return p


def ll(p):
    p = p.clip(1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


sets = {
    "raw margin": ["raw"],
    "win %": ["wpct"],
    **{f"adjusted margin {v}": [f"adj_{v}"] for v in V},
    "deserved win % A": ["dwpct_A"],
    "deserved win % B": ["dwpct_B"],
    "raw + adjusted B": ["raw", "adj_B"],
    "raw + adjusted A": ["raw", "adj_A"],
    "defence-only A": ["def_A"],
    "offence-only A": ["off_A"],
    "defence-only C": ["def_C"],
    "raw + defence-only A": ["raw", "def_A"],
}
P = {k: loso(f) for k, f in sets.items()}
spread_ok = G.spread_home.notna().to_numpy()
Xs = G.spread_home.to_numpy()[:, None]
ps = np.full(len(G), np.nan)
for s in np.unique(seasons):
    tr, te = (seasons != s) & spread_ok, (seasons == s) & spread_ok
    ps[te] = predict_logit(fit_logit(Xs[tr], y[tr]), Xs[te])

buckets = {"games 10-20": (10, 21), "games 21-41": (21, 42), "games 42+": (42, 99)}
nb = np.minimum(G.n_before_h, G.n_before_a).to_numpy()
print(f"\n4b. NEXT-GAME PREDICTION (leave-one-season-out logistic on rating gap; {len(G)} games; lower is better)")
print(f"  {'predictor':24}  log loss  Brier   " + "  ".join(f"{b:>12}" for b in buckets) + "   Δ log loss vs raw [95% CI]")
base = ll(P["raw margin"])
game_pred = {}
for k, p in P.items():
    l = ll(p)
    by = {b: float(l[(nb >= lo) & (nb < hi)].mean()) for b, (lo, hi) in buckets.items()}
    d = l - base
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    game_pred[k] = {"log_loss": float(l.mean()), "brier": float(((p - y) ** 2).mean()), "by_bucket": by, "delta_vs_raw": float(d.mean()), "delta_ci": ci}
    print(f"  {k:24}  {l.mean():.4f}  {((p-y)**2).mean():.4f}  " + "  ".join(f"{x:12.4f}" for x in by.values()) + f"   {d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
ls = ll(ps[spread_ok])
print(f"  {'closing spread (market)':24}  {ls.mean():.4f}  {((ps[spread_ok]-y[spread_ok])**2).mean():.4f}   (same games: raw {base[spread_ok].mean():.4f})")
game_pred["closing spread"] = {"log_loss": float(ls.mean()), "raw_same_games": float(base[spread_ok].mean())}
R["next_game"] = game_pred

# rest-of-season, team level
print("\n4c. REST-OF-SEASON WIN % FROM THE FIRST N GAMES (leave-one-season-out linear fit; RMSE in win %, lower is better)")
ros = {}
for N in (10, 20, 30, 41):
    rows = []
    for (season, team_id), d in t.groupby(["season", "team_id"]):
        if len(d) < N + 20:
            continue
        first, rest = d.iloc[:N], d.iloc[N:]
        rows.append({"season": season, "y": rest.win.mean(), **{k: first[c].mean() for k, c in preds.items()}})
    df = pd.DataFrame(rows)
    res = {}
    for k in preds:
        err = []
        for s in df.season.unique():
            tr, te = df[df.season != s], df[df.season == s]
            b = np.polyfit(tr[k], tr.y, 1)
            err.append(te.y - np.polyval(b, te[k]))
        e = np.concatenate(err)
        res[k] = float(np.sqrt(np.mean(e**2)))
    ros[N] = res
    print(f"  N={N:2}: " + "  ".join(f"{k}={x:.4f}" for k, x in res.items()))
R["rest_of_season_rmse"] = ros

# ---------- 5. eye test: 2024-25's biggest robberies ----------
print("\n5. 2024-25: BIGGEST ROBBERIES (lowest winner chance), variants A and B")
eye = {}
for v in ("A", "B"):
    x = g[g.season == 2025].copy()
    x["w"] = np.where(x.home_win == 1, x[f"deserved_{v}"], 1 - x[f"deserved_{v}"])
    x = x.sort_values("w").head(10)
    eye[v] = [
        f"{r.date.date()} {r.away_abbrev} {r.away_pts} @ {r.home_abbrev} {r.home_pts}: winner had {r.w:.0%}" + (" (OT)" if r.periods > 4 else "")
        for r in x.itertuples()
    ]
    print(f"  {v}:")
    for line in eye[v]:
        print("    " + line)
R["eye_test_2025"] = eye

# shooting-luck bias check: league-wide actual minus expected points per team-game should be ~0
R["bias_pts_per_team_game"] = {v: float(((g.home_pts + g.away_pts) - (g[f"xpts_home_{v}"] + g[f"xpts_away_{v}"])).mean() / 2) for v in V}
print("\nBias check, actual − expected points per team-game:", {k: round(x, 2) for k, x in R["bias_pts_per_team_game"].items()})

out = parse.ROOT / "results" / "pilot.json"
out.write_text(json.dumps(R, indent=2, default=str))
print(f"\nSaved {out}")
