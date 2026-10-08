"""Real numbers for the style-guide mock-up (docs/bucketweights-style-guide.html): 2024-25 as of 1 Jan 2025."""
import json
import numpy as np
import pandas as pd
from scipy.stats import beta
import parse, model

K3, KFT = 300, 25
all_shots = pd.concat([parse.parse(y)[0] for y in range(2015, 2027)], ignore_index=True)
names = {}
raw = pd.read_parquet(parse.RAW / "espn_pbp" / "play_by_play_2025.parquet", columns=["athlete_id_1", "athlete_name_1", "team_id", "home_team_id", "home_team_abbrev", "away_team_id", "away_team_abbrev"]).dropna(subset=["athlete_id_1"])
names = dict(zip(raw.athlete_id_1.astype("int64"), raw.athlete_name_1))
_, games = parse.parse(2025)
sp = model.make_probabilities(all_shots, 2025, K3, KFT)
g = pd.read_parquet(parse.DERIVED / "pilot_games.parquet")
g = g[g.season == 2025]
out = {"games": []}
picks = [("2024-12-26", "ORL"), ("2025-03-16", "CLE"), ("2024-12-19", "DET")]
for date, home in picks:
    r = g[(g.date == date) & (g.home_abbrev == home)].iloc[0]
    s = sp[(sp.game_id == r.game_id) & sp.p_A.notna()].copy()
    s["v"] = s.kind.map({"3": 3, "FT": 1})
    s["luck"] = (s.made - s.p_A) * s.v
    winner_home = r.home_pts > r.away_pts
    w_id = r.home_id if winner_home else r.away_id
    by_team = s.groupby("team_id").apply(lambda x: pd.Series({
        "3pm": int(x[x.kind == "3"].made.sum()), "3pa": int((x.kind == "3").sum()), "x3pm": float(x[x.kind == "3"].p_A.sum()),
        "ftm": int(x[x.kind == "FT"].made.sum()), "fta": int((x.kind == "FT").sum()), "xftm": float(x[x.kind == "FT"].p_A.sum()),
        "luck": float(x.luck.sum())}))
    pl = s.groupby(["team_id", "shooter"]).agg(luck=("luck", "sum"), m3=("made", lambda m: int(m[s.loc[m.index, "kind"] == "3"].sum())), a3=("kind", lambda k: int((k == "3").sum()))).reset_index()
    top = pl.reindex(pl.luck.abs().sort_values(ascending=False).index).head(2)
    wc = r.deserved_A if winner_home else 1 - r.deserved_A
    out["games"].append({
        "date": date, "away": r.away_abbrev, "home": r.home_abbrev, "away_pts": int(r.away_pts), "home_pts": int(r.home_pts),
        "ot": bool(r.periods > 4), "winner_chance": float(wc),
        "teams": {("home" if t == r.home_id else "away"): v for t, v in by_team.to_dict("index").items()},
        "swing": [{"name": names.get(int(x.shooter)), "team": r.home_abbrev if x.team_id == r.home_id else r.away_abbrev, "luck": float(x.luck), "3pm": int(x.m3), "3pa": int(x.a3)} for x in top.itertuples()],
    })
# Paul George 3P% meter at 1 Jan 2025 (80% interval of true talent)
pid = [k for k, v in names.items() if v == "Paul George"][0]
x = all_shots[(all_shots.shooter == pid) & (all_shots.kind == "3") & (all_shots.zone != "heave")]
cur = x[(x.season == 2025) & (x.date < "2025-01-01")]
pr = x[(x.season < 2025) & (x.season >= 2022)]
w = model.DECAY ** (2025 - pr.season)
prev = all_shots[(all_shots.season == 2024) & (all_shots.kind == "3") & (all_shots.zone != "heave")]
mu = prev.made.mean()
a = (pr.made * w).sum() + cur.made.sum() + K3 * mu
b = (w.sum() - (pr.made * w).sum()) + (len(cur) - cur.made.sum()) + K3 * (1 - mu)
out["meter"] = {"name": "Paul George", "actual": float(cur.made.mean()), "made": int(cur.made.sum()), "att": int(len(cur)),
                "true": float(a / (a + b)), "lo80": float(beta.ppf(0.1, a, b)), "hi80": float(beta.ppf(0.9, a, b)), "league": float(mu)}
print(json.dumps(out, indent=1))
(parse.ROOT / "results" / "mock-data.json").write_text(json.dumps(out, indent=1))
