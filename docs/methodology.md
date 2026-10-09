# BucketWeights methodology, version 1 (for opening night, 2026-27)

Status: **approved by Josh ("for now")**, 9 October 2026. Every number below comes from a script in
`research/` and can be reproduced (commands at the end). Seasons are named by the year they end in
(2025 = 2024-25).

## What the site says, and how sure it is

| Claim on the site | How it's made | Tested how | Result |
| --- | --- | --- | --- |
| A player's **true level** for 3s, 2s and free throws, with an 80% range | Shrunk player rates, past seasons fading | Predicting each shot before it happens; does the 80% range hold 80% of outcomes? | Beats every simpler rule; ranges hold 78–84% |
| "**Averaging X, about Y at his true shooting**, same shots and minutes" | His makes reset to his level | Does the predicted change in efficiency happen later in the season? | Right size (slope 0.96–0.99); big calls right 3 times in 4 |
| Each game: **Robbery / Coin flip / Earned it**, with the winner's chance | Exact score chances with 3s and free throws as luck, then calibrated | Do 20% chances come true 20% of the time? | Yes, after calibration, including on the held-out season |
| A team's **points from shooting luck** | Sum of its games | Described, not predicted | Shown as what happened, not a forecast (see "What we don't claim") |

## 1. Data

- **Source:** ESPN's public play-by-play: every shot with shooter, type, distance or coordinates,
  and made or missed. History from sportsdataverse's copies of the same feed, 2014-15 to
  2025-26. The nightly run reads ESPN directly.
- **Checks:** every season's play-by-play is matched to ESPN's box scores. Points agree in over
  99% of team-games. Before 2024-25 the feed doesn't mark whether a *missed* shot was a 3, so it's
  read from the text and coordinates (right within one attempt in about 95% of team-games); from
  2024-25 the feed says so directly. **2015-16 is left out** (shots missing in about half its
  games). From 2025-26 box scores leave out missed end-of-quarter heaves; we keep them, at their
  real make rate (about 3%).
- **No look-ahead:** every estimate for a game uses only games before it.
- **Held out:** 2025-26 was not used to fit anything; it was checked once, at the end (section 6).

## 2. A player's true level

For each shot type (3s, 2s, free throws):

> level = (his makes over the last three seasons, each older season counting *w* times the one
> after it, + his makes this season so far + *k* × the league rate) ÷ (the same with attempts + *k*)

*k* is how many attempts' worth of "league average" every player starts with: it decides how much
evidence it takes before his own shooting dominates. *w* is how fast old seasons fade. Both were
**fitted, not assumed**: we picked the values that best predicted every shot from 2017-18 to
2024-25, using only what was known before each game (log loss per shot, lower is better).

| Shot | Fitted *k* | Fitted *w* | League average | His raw rate this season | Raw rate, 4 seasons | **Fitted** |
| --- | --- | --- | --- | --- | --- | --- |
| 3s (642,905 shots) | 150 | 0.8 | 0.6559 | 0.7209 | 0.6710 | **0.6549** |
| Free throws (424,397) | 20 | 0.5 | 0.5325 | 0.6118 | 0.5353 | **0.5163** |
| 2s (1,035,739) | 50 | 0.5 | 0.6911 | 0.7268 | 0.6967 | **0.6862** |

- **The often-quoted "3P% needs 750 attempts" isn't what we found.** The best *k* for threes is
  150, and anything from 60 to 500 does almost as well: three-point shooting is so noisy that the
  exact amount of shrinkage barely matters. What matters is shrinking at all. A player's raw rate
  this season is the worst predictor tested, worse than assuming everyone is league average.
- **Free throws are the opposite:** about 20 attempts and his own rate takes over.
- **Where a 3 was taken:** corner 3s go in more often than above-the-break 3s. Shifting a player's
  level by the league's corner/above-break gap improves predictions slightly, so it's used. Heaves
  get the league heave rate, never the shooter's.
- **Home and away:** home teams shoot slightly better (3s 36.7% vs 35.9%, free throws 77.2% vs
  77.0%), but adding that to each shot didn't improve shot-level predictions. The home edge is
  handled once, in the game calibration (section 4).
- **The 80% range** is the middle 80% of the beta distribution behind the level. Checked by asking
  whether each player's actual rest-of-season makes landed inside the 80% range the level implied:
  83% did for 3s, 80% for free throws, 79% for 2s (3,178 player-cutoffs). The ranges are honest.

## 3. "Averaging X, about Y at his true shooting"

His points a game so far, with every make reset to his level and nothing else changed: same
shots, same minutes. Tested at 1 December, 1 January and 1 February of seven seasons (3,178
player-cutoffs, at least 10 games either side and 8 shots a game):

| Projection | Points-per-chance error after the cut | Predicted change shows up at |
| --- | --- | --- |
| He keeps doing what he's doing | 0.0857 | — |
| 3s and free throws reset to his level | 0.0750 | 0.99 (right size) |
| **All shots reset to his level** | **0.0679** | **0.96** |
| *Pilot: back to his "usual %" (raw, recent seasons)* | *0.074* | *0.63–0.74 (overshoots)* |

- The site uses **all shots**. Players projected to move 1.5+ points a game moved that way 75%
  of the time.
- **Points a game improve only a little** (error 2.74 → 2.70), because minutes, role, trades and
  injuries move scoring more than shooting does. That's why every projection says "same shots and
  minutes".
- **"Back to his usual %" overshoots**, which is the gambler's fallacy in numbers: a cold shooter
  isn't owed makes. The site never says "due".

## 4. Game verdicts

**What's luck:** every 3-pointer and free throw is a coin flip at the shooter's level. Everything
else (2s, turnovers, rebounds, fouls drawn) stays as it happened.

**Why not 2s:** in the pilot, treating 2s as luck at league rates for their zone made the chances
badly wrong (games given 7% were won about 40% of the time). How well a team finishes inside is
mostly skill that public data can't separate from luck.

**Exact, not simulated:** each team's points are an exact distribution (every combination of makes
and misses, added up), so a game's chances never change between runs. Regulation only; a tie after
regulation counts as half a win, since overtime is close to a coin flip.

**Calibration:** the raw counterfactual is too sure of itself. Over 2017-18 to 2024-25, when it
gave a home side 2.5%, that side came out ahead 9% of the time, and home teams did 3 points of
chance better overall than it said. So every chance is passed through a correction fitted on past seasons (fitted on six
seasons, tested on the seventh, in turn):

> logit(chance) = 0.186 + 0.727 × logit(raw chance) (home side)

The 0.186 is the home-court edge; the 0.727 pulls overconfident chances toward 50%. After it:

| Predicted (home) | 2.8% | 10.4% | 23.2% | 37.8% | 50.2% | 62.6% | 77.5% | 89.7% | 97.3% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Actual | 2.8% | 12.7% | 23.3% | 37.7% | 48.2% | 62.0% | 79.8% | 88.9% | 95.1% |

**Labels**, by the winner's calibrated chance: **Robbery** under 35% (12% of games, about one a
night), **Coin flip** 35–65%, **Earned it** 65% or more.

**Known simplifications, kept for version 1:**
- A missed 3 can be rebounded and scored; the model keeps the real rebounds and putbacks rather
  than modelling them. The calibration absorbs the net effect.
- Shooting fouls stay as they happened (an and-one stays an and-one).
- End-of-game fouling and garbage time stay in. Real games end regulation tied 2.2 times as often
  as the raw model says (teams play for the tie late); calibration again absorbs most of it.
- No defender data: if a team's shots were unusually open or contested, the model can't see it.

## 5. What we don't claim

In the pilot, **luck-adjusted team ratings did not predict future results better than plain point
differential**, and teams' "luck wins" over a season spread far wider (±5.6) than luck alone
explains (±3.4). Much of what looks like a team's shooting luck over months is real shot quality
we can't see. So team luck is shown as **what happened**, never as "this team is better than its
record", and playoff odds (later in the season) will be built on point differential.

## 6. The held-out season (2025-26), checked once

Nothing was re-fitted.

| Check | 2025-26 result |
| --- | --- |
| Shooting levels vs league average (log loss) | 3s 0.6524 vs 0.6532 · FT 0.5093 vs 0.5233 · 2s 0.6838 vs 0.6882 |
| Game verdicts, calibrated | home share 54.9% predicted, 55.0% actual; Brier 0.1657 raw → 0.1634 calibrated; Robbery 10.5% of games |
| Player projections | predicted change shows up at 0.96; error 0.0851 → 0.0667 points per chance |
| 80% ranges | 3s 82.7%, free throws 78.0%, 2s 83.7% inside |

## 7. Plain-English version (for the "About the model" page)

**What's a true level?** Every shooter has a real make rate we can't see directly; we can only see
his makes, and they bounce around. His true level is our best estimate of the real rate: his last
three seasons and this one, recent games counting more, pulled toward the league average until
he's taken enough shots to trust. For threes that takes a while; for free throws, about 20 shots.

**Is he due?** No. Nobody is owed buckets. A cold shooter's next three goes in at his true level,
however cold he's been. When we say "about 20 points a game at his true shooting", we mean: if he
keeps getting the same shots and minutes, and they go in at his level, that's what he'd score.

**What's a Robbery?** We replay every three-pointer and free throw at each shooter's true level,
keep everything else as it happened, and work out exactly how often each team would have won.
If the winner would have won less than 35% of the time, it's a Robbery. We've checked these
chances against seven past seasons: when we say 20%, it happens about 20% of the time.

**What can't it see?** How open the shots were. Public data doesn't track defenders, so a player
whose shots got harder will look unlucky when he isn't. We say so wherever it matters.

## Reproduce

```bash
.venv/bin/python research/pilot/run_games.py     # pilot (sections 4–5 background)
.venv/bin/python research/pilot/analyze.py
.venv/bin/python research/v1/talent.py           # section 2
.venv/bin/python research/v1/games.py            # section 4
.venv/bin/python research/v1/players.py          # section 3
.venv/bin/python research/v1/holdout.py          # section 6 (run once)
```
