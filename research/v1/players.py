"""
Check 3: player bounce-backs with the fitted shooting levels, at three points in the season
(1 Dec, 1 Jan, 1 Feb), and whether the stated ranges are honest.

  projection   his points a game so far, with every make reset to his level (2s, 3s and FTs,
               each at his own shrunk level with talent.py's k and w), same shots and minutes.
  range        the level's 80% range, from the beta posterior; checked by asking whether his
               actual rest-of-season 3P% lands inside the 80% predictive range (beta-binomial
               for the attempts he actually took) about 80% of the time.

Writes results/v1-players.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import betabinom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pilot"))
import parse  # noqa: E402

SEASONS = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2017, 2018, 2019, 2022, 2023, 2024, 2025]
OUT = sys.argv[2] if len(sys.argv) > 2 else "v1-players.json"
LAST = max(SEASONS) + 1  # seasons of data to load
CUTS = ["12-01", "01-01", "02-01"]
FIT = json.loads((parse.ROOT / "results" / "v1-talent.json").read_text())
VALUE = {"2": 2, "3": 3, "FT": 1}
MIN_GAMES, MIN_FGA = 10, 8

all_shots = pd.concat([parse.parse(y)[0] for y in range(2015, LAST)], ignore_index=True)
all_shots = all_shots[(all_shots.zone != "heave") & all_shots.shooter.notna()]

rows = []
for season in SEASONS:
    cur = all_shots[all_shots.season == season]
    prev = all_shots[all_shots.season == season - 1]
    for cut_md in CUTS:
        cut = pd.Timestamp(f"{season - 1 if cut_md.startswith('12') else season}-{cut_md}")
        h1, h2 = cur[cur.date < cut], cur[cur.date >= cut]
        d = pd.DataFrame({"g1": h1.groupby("shooter").game_id.nunique(), "g2": h2.groupby("shooter").game_id.nunique()}).dropna()
        for k in VALUE:
            kk, w = FIT[k]["k"], FIT[k]["w"]
            mu = prev[prev.kind == k].made.mean()
            pr = all_shots[(all_shots.kind == k) & (all_shots.season < season) & (all_shots.season >= season - 3)]
            wt = w ** (season - pr.season)
            pm = (pr.made * wt).groupby(pr.shooter).sum().reindex(d.index).fillna(0)
            pa = wt.groupby(pr.shooter).sum().reindex(d.index).fillna(0)
            for half, part in (("1", h1), ("2", h2)):
                t = part[part.kind == k].groupby("shooter").made.agg(["sum", "count"]).reindex(d.index).fillna(0)
                d[f"m{k}_{half}"], d[f"a{k}_{half}"] = t["sum"], t["count"]
            d[f"alpha{k}"] = pm + d[f"m{k}_1"] + kk * mu
            d[f"beta{k}"] = (pa - pm) + (d[f"a{k}_1"] - d[f"m{k}_1"]) + kk * (1 - mu)
            d[f"lvl{k}"] = d[f"alpha{k}"] / (d[f"alpha{k}"] + d[f"beta{k}"])
        fga1 = d.a2_1 + d.a3_1
        d = d[(d.g1 >= MIN_GAMES) & (d.g2 >= MIN_GAMES) & (fga1 / d.g1 >= MIN_FGA)].copy()
        d["season"], d["cut"] = season, cut_md
        rows.append(d)
D = pd.concat(rows)

ppg1 = sum(VALUE[k] * D[f"m{k}_1"] for k in VALUE) / D.g1
ppg2 = sum(VALUE[k] * D[f"m{k}_2"] for k in VALUE) / D.g2
ch1 = D.a2_1 + D.a3_1 + 0.44 * D.aFT_1
ch2 = D.a2_2 + D.a3_2 + 0.44 * D.aFT_2
R = {"k_w": {k: [FIT[k]["k"], FIT[k]["w"]] for k in VALUE}, "player_cuts": len(D), "by_method": {}}
print(f"{len(D)} player-cutoffs ({', '.join(CUTS)}; ≥{MIN_GAMES} games each side, ≥{MIN_FGA} FGA a game)")
for name, kinds in (("naive", ()), ("3s+FTs", ("3", "FT")), ("all shots", ("2", "3", "FT"))):
    luck = sum(VALUE[k] * (D[f"m{k}_1"] - D[f"a{k}_1"] * D[f"lvl{k}"]) for k in kinds) if kinds else 0 * D.g1
    proj = ppg1 - luck / D.g1
    pps1, pps_proj, pps2 = ppg1 * D.g1 / ch1, proj * D.g1 / ch1, ppg2 * D.g2 / ch2
    pc, ac = pps_proj - pps1, pps2 - pps1
    big = (proj - ppg1).abs() >= 1.5
    r = {
        "ppg_rmse": float(np.sqrt(((ppg2 - proj) ** 2).mean())),
        "pts_per_chance_rmse": float(np.sqrt(((pps2 - pps_proj) ** 2).mean())),
        "slope": float(np.polyfit(pc, ac, 1)[0]) if kinds else None,
        "big_calls": int(big.sum()),
        "big_right_direction": float((np.sign((proj - ppg1)[big]) == np.sign((ppg2 - ppg1)[big])).mean()) if big.any() else None,
    }
    R["by_method"][name] = r
    slope = "—" if r["slope"] is None else f"{r['slope']:.2f}"
    right = "" if r["big_right_direction"] is None else f", {r['big_right_direction']:.0%} right way"
    print(f"  {name:10} PPG RMSE {r['ppg_rmse']:.3f}  pts/chance RMSE {r['pts_per_chance_rmse']:.4f}  slope {slope}  big calls {r['big_calls']}{right}")

# coverage of the 80% predictive range for rest-of-season makes
cov = {}
for k in ("3", "FT", "2"):
    x = D[D[f"a{k}_2"] >= 20]
    n = x[f"a{k}_2"].astype(int).to_numpy()
    lo = betabinom.ppf(0.10, n, x[f"alpha{k}"], x[f"beta{k}"])
    hi = betabinom.ppf(0.90, n, x[f"alpha{k}"], x[f"beta{k}"])
    m = x[f"m{k}_2"].to_numpy()
    inside = (m >= lo) & (m <= hi)
    cov[k] = {"players": int(len(x)), "inside_80pct_range": float(inside.mean()), "below": float((m < lo).mean()), "above": float((m > hi).mean())}
    print(f"  80% range, {k}: {inside.mean():.1%} inside ({(m < lo).mean():.1%} below, {(m > hi).mean():.1%} above), {len(x)} player-cutoffs")
R["coverage"] = cov
(parse.ROOT / "results" / OUT).write_text(json.dumps(R, indent=2))
