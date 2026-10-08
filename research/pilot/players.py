"""
Pilot 2: player bounce-backs. "X is averaging 18, but at his true shooting he'd be at 21."

At 1 January of each season, for every regular player, estimate his scoring with the shooting
luck taken out, then check against what he actually scored from January on.

  naive          keep scoring what he's scoring
  usual %        his makes reset to his own recent seasons' raw rates (what fans mean by "his average")
  true talent    his makes reset to a shrunk estimate: his recent seasons + this season, pulled
                 toward the league by k attempts (k measured from the data)

Each is run with 3s + FTs as luck, and with 2s as well (at the player's own shrunk 2P%). Shot
volume is held at his first-half rate in all of them, so the comparison is about shooting only.
Only information available on 1 January is used. Writes results/players-pilot.json.
"""
import json

import numpy as np
import pandas as pd

import model
import parse

SEASONS = [2017, 2018, 2019, 2022, 2023, 2024, 2025]
VALUE = {"2": 2, "3": 3, "FT": 1}
MIN_GAMES, MIN_FGA = 15, 8

all_shots = pd.concat([parse.parse(y)[0] for y in range(2015, 2027)], ignore_index=True)
all_shots = all_shots[all_shots.zone != "heave"]  # heaves: tiny, and not part of anyone's shooting talent

K = {kind: np.median([model.shrinkage_k(all_shots[all_shots.season == y], kind)["k"] for y in SEASONS]) for kind in ("2", "3", "FT")}
print("Prior weight k (attempts):", {k: round(v) for k, v in K.items()})

names = {}
for y in (2025, 2024):
    raw = pd.read_parquet(parse.RAW / "espn_pbp" / f"play_by_play_{y}.parquet", columns=["athlete_id_1", "athlete_name_1"]).dropna()
    names.update(dict(zip(raw.athlete_id_1.astype("int64"), raw.athlete_name_1)))

rows = []
for season in SEASONS:
    cut = pd.Timestamp(f"{season}-01-01")
    cur = all_shots[all_shots.season == season]
    prior = all_shots[(all_shots.season < season) & (all_shots.season >= season - 3)].assign(w=lambda d: model.DECAY ** (season - d.season))
    prev = all_shots[all_shots.season == season - 1]
    mu = {k: prev[prev.kind == k].made.mean() for k in VALUE}

    for half, part in (("1", cur[cur.date < cut]), ("2", cur[cur.date >= cut])):
        g = part.groupby("shooter").game_id.nunique().rename(f"g{half}")
        t = part.groupby(["shooter", "kind"]).made.agg(["sum", "count"]).unstack(fill_value=0)
        t.columns = [f"{a}{b}_{half}" for a, b in t.columns]  # sum2_1, count3_1, ...
        if half == "1":
            h1 = t.join(g)
        else:
            h2 = t.join(g)
    pm = (prior.made * prior.w).groupby([prior.shooter, prior.kind]).sum().unstack(fill_value=0)
    pa = prior.w.groupby([prior.shooter, prior.kind]).sum().unstack(fill_value=0)

    d = h1.join(h2, how="inner").fillna(0)
    for k in VALUE:
        for c in (f"sum{k}_1", f"count{k}_1", f"sum{k}_2", f"count{k}_2"):
            if c not in d:
                d[c] = 0
        d[f"pm{k}"] = pm.get(k, pd.Series(dtype=float)).reindex(d.index).fillna(0)
        d[f"pa{k}"] = pa.get(k, pd.Series(dtype=float)).reindex(d.index).fillna(0)
    fga1 = d["count2_1"] + d["count3_1"]
    d = d[(d.g1 >= MIN_GAMES) & (d.g2 >= MIN_GAMES) & (fga1 / d.g1 >= MIN_FGA)].copy()
    d["season"] = season
    d["ppg1"] = sum(VALUE[k] * d[f"sum{k}_1"] for k in VALUE) / d.g1
    d["ppg2"] = sum(VALUE[k] * d[f"sum{k}_2"] for k in VALUE) / d.g2
    # points per shooting chance, which takes playing time and volume out of the target
    d["pps1"] = d.ppg1 * d.g1 / (d.count2_1 + d.count3_1 + 0.44 * d.countFT_1)
    d["pps2"] = d.ppg2 * d.g2 / (d.count2_2 + d.count3_2 + 0.44 * d.countFT_2)
    for k in VALUE:
        d[f"p_true{k}"] = (d[f"pm{k}"] + d[f"sum{k}_1"] + K[k] * mu[k]) / (d[f"pa{k}"] + d[f"count{k}_1"] + K[k])
        # "his usual %": recent seasons' raw rate; this season's if he has no history
        d[f"p_usual{k}"] = np.where(d[f"pa{k}"] >= 20, d[f"pm{k}"] / d[f"pa{k}"].clip(lower=1e-9), d[f"sum{k}_1"] / d[f"count{k}_1"].clip(lower=1))
    rows.append(d)

D = pd.concat(rows)
D["name"] = D.index.map(lambda i: names.get(int(i), str(i)))


def projected(d, how: str, kinds) -> tuple[pd.Series, pd.Series]:
    """Projected PPG and points per chance with `kinds` reset to `how` rates (volume as in the first half)."""
    luck = sum(VALUE[k] * (d[f"sum{k}_1"] - d[f"count{k}_1"] * d[f"p_{how}{k}"]) for k in kinds)
    ppg = d.ppg1 - luck / d.g1
    chances = d.count2_1 + d.count3_1 + 0.44 * d.countFT_1
    return ppg, ppg * d.g1 / chances


R = {"k": K, "player_seasons": len(D), "seasons": SEASONS, "methods": {}}
print(f"\n{len(D)} player-seasons (≥{MIN_GAMES} games each side of 1 Jan, ≥{MIN_FGA} FGA a game before it)")
print(f"{'method':32} {'PPG RMSE':>9} {'PPG MAE':>8} {'pts/chance RMSE':>16} {'slope':>6}   big swings: direction right, predicted vs actual change")
methods = {"naive": None}
for how in ("usual", "true"):
    methods[f"{how} %, 3s+FTs"] = (how, ("3", "FT"))
    methods[f"{how} %, all shots"] = (how, ("2", "3", "FT"))
for name, spec in methods.items():
    if spec is None:
        ppg, pps = D.ppg1, D.pps1
    else:
        ppg, pps = projected(D, *spec)
    e, ep = D.ppg2 - ppg, D.pps2 - pps
    pred_change, act_change = ppg - D.ppg1, D.ppg2 - D.ppg1
    # efficiency: does the predicted change in points per chance show up? (slope 1 = right size)
    pc, ac = pps - D.pps1, D.pps2 - D.pps1
    slope = float(np.polyfit(pc, ac, 1)[0]) if spec else float("nan")
    big = pred_change.abs() >= 1.5
    res = {
        "ppg_rmse": float(np.sqrt((e**2).mean())),
        "ppg_mae": float(e.abs().mean()),
        "pps_rmse": float(np.sqrt((ep**2).mean())),
        "pps_slope": slope,
        "big_n": int(big.sum()),
        "big_direction_right": float((np.sign(pred_change[big]) == np.sign(act_change[big])).mean()) if big.any() else None,
        "big_pred_change": float(pred_change[big].abs().mean()) if big.any() else None,
        "big_actual_change_same_direction": float((act_change[big] * np.sign(pred_change[big])).mean()) if big.any() else None,
    }
    R["methods"][name] = res
    tail = (
        f"n={res['big_n']:3}  {res['big_direction_right']:.0%}  {res['big_pred_change']:.2f} vs {res['big_actual_change_same_direction']:.2f}"
        if res["big_n"]
        else ""
    )
    print(f"{name:32} {res['ppg_rmse']:9.3f} {res['ppg_mae']:8.3f} {res['pps_rmse']:16.4f} {slope:6.2f}   {tail}")

# examples: 2024-25 at 1 January, true-talent all-shots projection
ppg, _ = projected(D, "true", ("3", "FT"))
D["proj"] = ppg
x = D[D.season == 2025].assign(delta=lambda d: d.proj - d.ppg1)
ex = {}
for label, part in (("bounce_back", x.sort_values("delta", ascending=False).head(8)), ("cool_off", x.sort_values("delta").head(8))):
    ex[label] = [
        f"{r.name}: {r.ppg1:.1f} PPG to 1 Jan, projected {r.proj:.1f} (3P {r.sum3_1 / max(r.count3_1, 1):.1%} vs true {r.p_true3:.1%}); "
        f"actual after: {r.ppg2:.1f} (3P {r.sum3_2 / max(r.count3_2, 1):.1%})"
        for r in part.itertuples()
    ]
    print(f"\n2024-25 {label.replace('_', ' ')} (3s+FTs, true talent):")
    for line in ex[label]:
        print("  " + line)
R["examples_2025"] = ex
(parse.ROOT / "results" / "players-pilot.json").write_text(json.dumps(R, indent=2, default=str))
