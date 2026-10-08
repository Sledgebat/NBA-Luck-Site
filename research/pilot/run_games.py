"""Compute every game's deserved chances (all four variants) for the pilot seasons; cache to data/derived."""
import time
import pandas as pd
import parse, model

K3, KFT = 300, 25  # from shrinkage_k on 2017-2025 player seasons (3P: 270-420, FT: 18-27)
SEASONS = [2017, 2018, 2019, 2022, 2023, 2024, 2025]

all_shots = pd.concat([parse.parse(y)[0] for y in range(2015, 2027)], ignore_index=True)
out = []
for y in SEASONS:
    t = time.time()
    _, games = parse.parse(y)
    sp = model.make_probabilities(all_shots, y, K3, KFT)
    g = model.season_games(sp, games)
    out.append(g)
    print(y, len(g), f"{time.time()-t:.0f}s", flush=True)
pd.concat(out, ignore_index=True).to_parquet(parse.DERIVED / "pilot_games.parquet")
