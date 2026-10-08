"""
Turn one season of ESPN play-by-play (sportsdataverse release) into two tidy tables:

  shots  every field-goal attempt and free throw: game, team, shooter, kind (2/3/FT), made,
         period, distance, zone
  games  one row per regular-season game: teams, regulation and final scores, closing spread

Regular season only (season_type 2). Raw files live in data/raw/espn_pbp; parsed tables are cached
in data/derived.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"

RIM_WORDS = "Dunk|Layup|Tip|Alley Oop|Putback|Finger Roll"


def zone_of(kind: pd.Series, dist: pd.Series, type_text: pd.Series, y_raw: pd.Series) -> pd.Series:
    """Coarse shot zones (pilot): rim / short / mid / long 2s; corner / above-break / heave 3s."""
    z = pd.Series("FT", index=kind.index, dtype=object)
    two = kind == "2"
    rim = two & (type_text.str.contains(RIM_WORDS, na=False) | (dist <= 4))
    z[two] = np.select(
        [rim[two], dist[two] <= 10, dist[two] <= 16],
        ["rim", "short", "mid"],
        "long",
    )
    three = kind == "3"
    heave = three & (type_text.str.contains("Heave", na=False) | (dist >= 36))
    corner = three & ~heave & (y_raw <= 14) & (dist < 23.6)
    z[three] = np.select([heave[three], corner[three]], ["heave", "corner"], "above")
    return z


def parse(season: int, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    DERIVED.mkdir(parents=True, exist_ok=True)
    s_path, g_path = DERIVED / f"shots_{season}.parquet", DERIVED / f"games_{season}.parquet"
    if use_cache and s_path.exists() and g_path.exists():
        return pd.read_parquet(s_path), pd.read_parquet(g_path)

    df = pd.read_parquet(RAW / "espn_pbp" / f"play_by_play_{season}.parquet")
    df = df[df.season_type == 2].sort_values(["game_id", "game_play_number"])

    # --- games ---
    first = df.groupby("game_id").first()
    final = df.groupby("game_id")[["home_score", "away_score"]].max()
    reg = df[df.period_number <= 4].groupby("game_id")[["home_score", "away_score"]].max()
    games = pd.DataFrame(
        {
            "season": season,
            "date": pd.to_datetime(first.game_date.astype(str)),
            "home_id": first.home_team_id,
            "away_id": first.away_team_id,
            "home_abbrev": first.home_team_abbrev,
            "away_abbrev": first.away_team_abbrev,
            "home_pts": final.home_score,
            "away_pts": final.away_score,
            "home_reg": reg.home_score,
            "away_reg": reg.away_score,
            "periods": df.groupby("game_id").period_number.max(),
            # ESPN's home_team_spread is positive when the home side is favoured (checked: r = +0.52 with margin)
            "spread_home": first.home_team_spread.where(first.game_spread_available),
        }
    )
    games.index.name = "game_id"
    games = games.reset_index()

    # --- shots ---
    # Before 2024-25 the feed has no (or an empty) points_attempted column. Rebuild it: free throws
    # are 1; a field goal is a 3 when the text says "three point", when it scored 3, when the text
    # gives 23+ feet, or (no distance in the text) when the coordinates put it beyond the arc.
    if "points_attempted" not in df.columns or df.loc[df.shooting_play, "points_attempted"].isna().mean() > 0.5:
        ft_row = df.type_text.str.startswith("Free Throw", na=False)
        feet = df.text.str.extract(r"(\d+)-foot")[0].astype(float)
        xd = np.hypot(df.coordinate_x_raw - 25, df.coordinate_y_raw - 4)
        xy_three = (xd >= 23.5) | ((df.coordinate_y_raw <= 14) & ((df.coordinate_x_raw - 25).abs() >= 21.5))
        guess = (
            df.text.str.contains("three point", case=False, na=False)
            | (feet >= 23)
            | (feet.isna() & xy_three & (df.coordinate_x_raw > -1000))
        )
        three_row = np.where(df.scoring_play, df.score_value == 3, guess)  # a make's value is known
        df["points_attempted"] = np.where(ft_row, 1, np.where(three_row, 3, 2))
    s = df[df.shooting_play & df.points_attempted.isin([1, 2, 3])].copy()
    is_ft = s.type_text.str.startswith("Free Throw", na=False)
    s["kind"] = np.where(is_ft, "FT", s.points_attempted.astype(str))
    s = s[~((s.kind == "FT") ^ (s.points_attempted == 1))]  # drop the odd mis-coded row
    s["made"] = s.scoring_play.astype(int)
    feet = s.text.str.extract(r"(\d+)-foot")[0].astype(float)
    xy_dist = np.hypot(s.coordinate_x_raw - 25, s.coordinate_y_raw - 4)
    bad_xy = (s.coordinate_x_raw < -1000) | s.coordinate_x_raw.isna()
    s["dist"] = feet.fillna(xy_dist.where(~bad_xy))
    s["zone"] = zone_of(s.kind, s.dist, s.type_text, s.coordinate_y_raw)
    shots = pd.DataFrame(
        {
            "game_id": s.game_id,
            "season": season,
            "team_id": s.team_id.astype("Int64"),
            "shooter": s.athlete_id_1.astype("Int64"),
            "kind": s.kind,
            "zone": s.zone,
            "made": s.made,
            "period": s.period_number,
            "dist": s.dist,
            "type_text": s.type_text,
        }
    ).reset_index(drop=True)
    shots = shots[shots.team_id.notna() & shots.game_id.isin(games.game_id)]
    shots = shots.merge(games[["game_id", "date"]], on="game_id")

    shots.to_parquet(s_path)
    games.to_parquet(g_path)
    return shots, games


def reconcile(season: int, shots: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Play-by-play totals vs ESPN's team box score, per team-game."""
    box = pd.read_parquet(RAW / "espn_team_box" / f"team_box_{season}.parquet")
    box = box[box.season_type == 2]
    agg = shots.assign(fg=shots.kind != "FT", three=shots.kind == "3", ft=shots.kind == "FT")
    t = agg.groupby(["game_id", "team_id"]).apply(
        lambda x: pd.Series(
            {
                "fga": x.fg.sum(),
                "fgm": (x.fg & (x.made == 1)).sum(),
                "tpa": x.three.sum(),
                "tpm": (x.three & (x.made == 1)).sum(),
                "fta": x.ft.sum(),
                "ftm": (x.ft & (x.made == 1)).sum(),
            }
        )
    ).reset_index()
    b = box.rename(
        columns={
            "field_goals_attempted": "b_fga",
            "field_goals_made": "b_fgm",
            "three_point_field_goals_attempted": "b_tpa",
            "three_point_field_goals_made": "b_tpm",
            "free_throws_attempted": "b_fta",
            "free_throws_made": "b_ftm",
        }
    )[["game_id", "team_id", "b_fga", "b_fgm", "b_tpa", "b_tpm", "b_fta", "b_ftm", "team_score"]]
    m = t.merge(b, on=["game_id", "team_id"], how="inner")
    out = {"team_games": len(m), "box_team_games": len(b), "pbp_games": len(games)}
    for c in ["fga", "fgm", "tpa", "tpm", "fta", "ftm"]:
        out[f"{c}_exact"] = round(float((m[c] == m["b_" + c]).mean()), 4)
        out[f"{c}_within1"] = round(float(((m[c] - m["b_" + c]).abs() <= 1).mean()), 4)
    pts = 2 * m.fgm + m.tpm + m.ftm
    out["points_exact"] = round(float((pts == m.team_score).mean()), 4)
    return out
