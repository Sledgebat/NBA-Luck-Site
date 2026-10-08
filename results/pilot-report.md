# Pilot: is NBA shooting luck worth a site? (8 Oct 2026)

A quick, deliberately rough version of the model, run on seven past regular seasons (2016-17,
2017-18, 2018-19, 2021-22 to 2024-25; 8,619 games) to decide whether the full methodology study is
worth doing. 2025-26 is untouched, kept as the final holdout. Reproduce:

```bash
.venv/bin/python research/pilot/run_games.py
.venv/bin/python research/pilot/analyze.py
```

Full numbers: `results/pilot.json` and `results/pilot-output.txt`.

## What was tested

Four versions of "what counts as luck":

| | Treated as luck | Make chance used |
| --- | --- | --- |
| **A** (Josh's proposal) | 3s and free throws | shooter's own rate, shrunk toward the league |
| B | A + every 2 | 2s at the league rate for their zone (rim, short, mid, long) |
| C | 3s and free throws | league rates only (corner/above-break/heave) |
| D | every shot | league/zone rates only |

Each game's chance comes from the exact distribution of points (no simulation). Only information
from before each game is used. The pilot ignores offensive rebounds after misses and shooting-foul
details, and counts a tie after regulation as half a win.

## Data checks

- **ESPN works; the NBA's own feeds don't** (from Josh's Mac): ESPN answered in 0.2 s;
  cdn.nba.com returned 403 Forbidden; stats.nba.com timed out. GitHub-hosted history downloads
  fine. Still to check from GitHub's servers (`.github/workflows/probe-sources.yml`).
- **Play-by-play matches the box scores** in most seasons: points agree in over 99% of team-games,
  3-point makes in over 99%. Before 2024-25 the feed doesn't say whether a *missed* shot was a 3,
  so it's inferred from the text and coordinates: right in 60–80% of team-games and within one
  attempt in about 95%. **2015-16 is missing shots in about half its games** and is left out.
- **2025-26 box scores leave out missed end-of-quarter heaves** (a league scoring change). The model
  keeps them, at their real make rate (about 3%).
- **How much shooting talent shows up in a season** (prior weight in attempts): about 270–420 for
  3P% and about 20–27 for FT%. That's lower than the ~750 often quoted for 3P%, probably because
  shot mix (corner specialists, pull-up shooters) is mixed in with talent here. The full study
  measures this properly.

## Results

**1. Luck flips plenty of games.** Under version A, the winner had less than a 35% chance in
**16% of games** (about one game a night), less than 25% in 9%, and less than 50% in 29%. So a
"Robbery" label would show up often enough to make nightly content.

**2. Counting 2-pointers as luck doesn't work (with public data).** Versions B and D are badly
off: games they gave the winner a 7% chance were actually won 40–46% of the time. How well a team
shoots its 2s, compared with the league average for that zone, is mostly skill. So Josh's choice
of 3s and free throws only is the right one. I suggested including 2s, and the data says no.

**3. Version A is roughly honest, with three fixes needed:**
- **Too sure at the extremes:** a side given 5% actually came out ahead 13% of the time; 95% → 91%.
- **Home teams beat the model** by 3 points of win chance (0.564 actual vs 0.534 predicted): home
  teams seem to shoot 3s or free throws a little better than their rates say.
- **Real games end regulation tied 2.2 times as often as the model says.** End-of-game behaviour
  again; HockeyWeights saw the same with ties (1.37×).

The spread of actual scores around the model's expectation is right, though (variance ratio 0.98).

**4. Season "luck wins" are too big to be all luck.** Actual minus deserved wins varies by
**±5.6 wins** (one standard deviation) across teams, but pure chance under the model would give
±3.4. So roughly 60% of what the model calls luck over a season is something that persists. The
likeliest cause is shot quality the play-by-play can't see: good offences create more open 3s,
good defences contest them. The 2021-22 Suns at "+13 lucky wins" (64 W vs 51 deserved), despite
+7.5 point differential, shows the problem.

**5. Luck-adjusted ratings don't predict better than plain point differential.** Next-game log loss
(lower is better; 7,515 games, leave-one-season-out):

| Rating built from | Log loss | vs point differential |
| --- | --- | --- |
| Point differential (raw) | 0.6271 | — |
| Win % | 0.6302 | worse |
| Luck-adjusted margin, A | 0.6324 | worse (CI excludes 0) |
| Opponents' luck removed only (defence) | 0.6284 | no different |
| Raw + luck-adjusted A, blended | 0.6262 | −0.0009 (CI −0.0019 to +0.0001): not significant |
| *Betting market (closing spread)* | *0.5966* | *much better* |

Rest-of-season win % from the first 10, 20, 30 or 41 games tells the same story: luck-adjusted
versions are a little worse than raw point differential at every point.

## What it means

- **Game verdicts: go.** Shooting luck often decides who wins, and version A's probabilities are
  close to honest. The full study fixes the overconfidence, the home edge and the ties, and adds
  offensive rebounds and shooting fouls.
- **"Luck-adjusted standings predict better": not yet.** Over a season, a large part of the
  "luck" is persistent shot quality. The site shouldn't claim this until a better model beats raw
  point differential out of sample. Playoff odds should be built on point differential, or a blend
  if one is shown to help.
- **For the full study's priority list:** use "assisted" (in the play text) and shot type to
  stand in for how open a shot was; estimate team offence and defence shot-quality effects; home
  shooting edge; end-game rules; calibration at the extremes. Then re-run this test. If no version
  beats raw point differential, the honest framing is "this game swung on shooting", not "this
  team is better than its record".

---

# Pilot 2: player bounce-backs (8 Oct 2026)

"X is averaging 18, but at his true shooting he'd be scoring 21." Tested on 1,031 player-seasons
(same seven seasons; players with 15+ games on both sides of 1 January and 8+ shots a game). On
1 January, each player's shooting luck is taken out and the result compared with what he actually
did from January on. Reproduce: `.venv/bin/python research/pilot/players.py`.

How quickly a player's own rate becomes trustworthy (prior weight, attempts): **2P% 87, 3P% 292,
FT% 23**.

| Projection | PPG error (RMSE) | Points-per-chance error | Predicted change shows up at (1.0 = right size) |
| --- | --- | --- | --- |
| Keep scoring what he's scoring | 2.66 | 0.082 | — |
| Back to his usual % (recent seasons, raw), 3s+FTs | 2.65 | 0.074 | 0.74 (overshoots) |
| Back to his usual %, all shots | 2.68 | 0.074 | 0.63 (overshoots) |
| **True talent (shrunk), 3s+FTs** | **2.61** | **0.070** | **1.03** |
| **True talent (shrunk), all shots** | **2.61** | **0.065** | **0.91** |

- **The honest version works.** Taking shooting luck out using a shrunk true-talent estimate cuts
  the error in scoring efficiency by 14% (3s+FTs) to 21% (all shots), and the predicted change is
  the right size. At the player level, using his *own* shrunk 2P% helps (unlike team-level zone
  averages, which failed).
- **"Back to his usual %" overshoots**: only about two-thirds of the predicted swing happens. The
  site should use the shrunk estimate and say so.
- **Points per game improve only a little** (2.66 → 2.61), because minutes, role, trades and
  injuries move PPG more than shooting does. So "could bounce back to X" should be framed as
  "at the same shots and minutes".
- **Big calls are right about three times in four:** of 50 players projected to move 1.5+ PPG
  (all-shots version), 74% moved that way, by 1.43 PPG on average against 1.79 predicted.
- **2024-25 examples at 1 January:** Jokic 52.5% from three → true 39.5% (shot 38.7% after);
  Paul George 32.0% → 38.5% (shot 39.9% after; PPG 15.4 → 17.0, projected 16.9). Misses come from
  role changes (Anthony Edwards took more shots; Fred VanVleet fewer).
