"""
The one look at 2025-26, the season held out of every fit: are the shooting levels, the
recalibrated game verdicts and the player projections as good there as in the seasons they were
fitted on? Nothing is re-fitted here. Writes results/v1-holdout.json (players: v1-holdout-players.json).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pilot"))
import games as G  # noqa: E402
import parse  # noqa: E402
import talent as T  # noqa: E402

HOLD = 2026
R = {}

# 1. shooting levels: log loss per shot in 2025-26 with the fitted k, w
shots = pd.concat([parse.parse(y)[0] for y in range(2015, HOLD + 1)], ignore_index=True)
games_all = pd.concat([parse.parse(y)[1] for y in range(2015, HOLD + 1)], ignore_index=True)
s = shots.merge(games_all[["game_id", "home_id"]], on="game_id")
s["home"] = (s.team_id == s.home_id).astype(int)
s = s[s.shooter.notna()]
T.EVAL = [HOLD]
R["levels"] = {}
for kind in ("3", "FT", "2"):
    pg = T.player_games(s, kind)
    k, w = G.FIT[kind]["k"], G.FIT[kind]["w"]
    R["levels"][kind] = {"fitted": T.logloss(pg, T.level(pg, k, w)), "league average": T.logloss(pg, pg.mu.to_numpy())}
    print(f"levels {kind}: fitted {R['levels'][kind]['fitted']:.5f} vs league average {R['levels'][kind]['league average']:.5f}")

# 2. game verdicts: v1 probabilities, recalibrated with (a, b) fitted on the seven earlier seasons
ab = json.loads((parse.ROOT / "results" / "v1-games.json").read_text())["versions"]["v1"]["recalibration_ab_all_seasons"]
g = games_all[games_all.season == HOLD]
df = G.game_rows(G.shot_probs(shots, games_all, HOLD, "v1"), g)
df["reg_result"] = np.where(df.home_reg > df.away_reg, 1.0, np.where(df.home_reg < df.away_reg, 0.0, 0.5))
raw = df.p.to_numpy()
rc = G.inv(ab[0] + ab[1] * G.logit(np.clip(raw, 1e-4, 1 - 1e-4)))
R["games"] = {"n": len(df), "raw": G.score(df, raw), "recalibrated": G.score(df, rc), "ab": ab}
for name in ("raw", "recalibrated"):
    r = R["games"][name]
    print(f"games {name}: Brier {r['brier']:.4f}, mean pred {r['mean_pred']:.3f} vs actual {r['mean_actual']:.3f}, winner <35% {r['winner_under_35']:.3f}")
    print("   " + "  ".join(f"[{p}→{a}, n={n}]" for _, n, p, a in r["table"]))
(parse.ROOT / "results" / "v1-holdout.json").write_text(json.dumps(R, indent=2))

# 3. player projections and ranges
print(subprocess.run([sys.executable, "-W", "ignore", str(HERE / "players.py"), str(HOLD), "v1-holdout-players.json"], capture_output=True, text=True).stdout)
