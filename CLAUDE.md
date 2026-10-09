# BucketWeights: notes for Claude

NBA shooting-luck site. Owner: **Josh**. He is not a developer: explain in plain language, give
exact Terminal commands when he needs to run something, don't assume he knows git or Python.
Independent fan site; the footer disclaimer stays on every page. Started 8 Oct 2026; target:
**live for opening night, Tuesday 20 October 2026** (2026-27 season = ESPN season 2027).

## Hard rules

- **Never touch HockeyWeights** (`/Users/westlake/AI Coding Projects/Edmonton Oilers Fansite/hockeyweights`).
  Reading it for reference is fine; copy patterns into this repo instead of sharing files. Don't
  run npm or git there. It's live and working.
- **Claude can't push.** Commit locally (end messages with the Co-Authored-By line); Josh pushes in
  GitHub Desktop. Repo: `Sledgebat/NBA-Luck-Site`, **public** since 9 Oct 2026.
- **Not publishing to Cloudflare yet** (Josh, 9 Oct 2026). The workflow's publish step is skipped
  until the `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_ACCOUNT_ID` secrets exist; don't ask him to add
  them until he says it's time. Pages project name will be `bucketweights`; no domain yet.
- **Honest stats:** never say a player is "due" (see Vocabulary). Never present team "luck" as a
  forecast or "better than their record" (it isn't predictive; see Model).

## How it runs

- **Nightly:** `.github/workflows/update-site.yml` at 03:45, 06:30, 10:20 UTC (+ "Run workflow",
  with a "preseason" tick box for rehearsals). Steps: tests → download DB from the `data` release
  (or backfill 2024–2026 if missing) → `python -m bucketweights.update` → `python -m
  bucketweights.site` → install Chromium → `python -m bucketweights.cards` (`continue-on-error`) →
  publish to Cloudflare (skipped without secrets) → keep the card images as the `social-cards`
  artifact (3 days) → save DB to the release → keep-alive commit after 45 quiet days. Runner pinned to `ubuntu-24.04` (ubuntu-latest moves to a
  new version from 19 Oct 2026). A run takes ~1 min (2.4 min the first time). Results show on the
  run's summary page, but **only when signed in** (checked 9 Oct 2026: signed out, GitHub shows no
  summary and no logs), so Josh has to read them out.
- **Fails (GitHub emails Josh) only when:** ESPN unreachable, a game can't be read, or no new
  regular-season game for 8+ days mid-season. Box-score mismatches don't fail; the game is flagged
  (`games.check_status`) and re-read for 3 days.
- **Idempotent:** each game is replaced whole in one transaction (`db.write_game`), only when its
  content hash changed; the last 3 days are always re-read (ESPN stat corrections).
- **GitHub free plan (checked 9 Oct 2026):** this repo is public → Actions minutes and artifact
  storage are free. HockeyWeights (`Sledgebat/Edmonton-Weights`) is now **private**: it alone uses
  the account's 2,000 free minutes a month (~600 for its 3 nightly runs, plus ~6 min for every push
  to its main branch). Artifacts here are uploaded only on failure (3 days). Release assets have no
  storage cap (DB ~24 MB, grows ~6 MB a season). If this repo ever goes private: ~225 min/month.

## Data

- **Live:** ESPN `site.api.espn.com` (fallback host `site.web.api.espn.com`): `/scoreboard?dates=`
  and `/summary?event=`. Works from GitHub's servers. cdn.nba.com returns 403 and stats.nba.com
  times out from Josh's Mac; **cdn.nba.com is 403 from GitHub too** (probe, 9 Oct 2026: schedule
  and scoreboard both 403). So there's no NBA fallback; ESPN is the only live source.
- **History:** sportsdataverse's copies of the same ESPN feed (GitHub release `espn_nba_pbp`,
  `play_by_play_{season}.parquet`). DB holds 2023–2026 + this season. 2015-16 is broken (missing
  shots) and never used.
- **Same rules for both:** `bucketweights/shots.py` (kind 2/3/FT, distance, zone). Live parse ==
  history parse exactly (tested on 2025-26 games).
- Season types: 1 preseason, 2 regular, 3 playoffs, 5 play-in. **NBA Cup final** = competition
  type `CC` (excluded from levels/standings; history finals hard-coded in `backfill.CUP_FINALS`).
  **All-Star games** (team ids > 30) are skipped and cleaned out (`db.drop_non_nba_games`).
- From 2025-26 box scores omit missed end-of-quarter heaves; we keep them at the league heave rate.
- Before 2024-25 a missed shot's 3-or-2 is inferred from text/coordinates (within one attempt in
  ~95% of team-games).
- Player's current team = team in his latest stored game (so off-season moves show from preseason).

## Model v1 (`bucketweights/params.json`, method in `docs/methodology.md`, approved by Josh 9 Oct 2026 "for now")

- **True level** per kind: (faded makes over the last 3 seasons + this season + k·league) ÷ (same
  attempts + k). Fitted by held-out log loss: 3s k=150 w=0.8; FT k=20 w=0.5; 2s k=50 w=0.5.
  Corner/above-break shift for 3s; heaves at the league heave rate. 80% range from the beta
  posterior (coverage checked: 78–84%).
- **"At his true shooting":** PPG with every make (2s, 3s, FTs) reset to his level, same shots and
  minutes. Predicted change is right-sized (slope 0.96–0.99); big calls right ~75%.
- **Game verdicts:** 3s and FTs are luck at the shooters' levels known before the game, everything
  else as it happened; exact score distributions; regulation only, a tie counts half; then
  calibrated: logit(home) = 0.1863 + 0.7269·logit(raw). Labels by the winner's chance: Robbery
  < 35% (~10–12% of games), Coin flip 35–65%, Earned it ≥ 65%. Holdout 2025-26 passed.
- **Not claimed:** team luck-adjusted ratings didn't beat point differential; season "luck wins"
  spread is too wide to be luck. Teams page shows shooting-luck points and robbery counts only
  ("wins at true shooting" was removed on purpose).
- Research scripts: `research/pilot/` (pilot), `research/v1/` (talent, games, players, holdout);
  results in `results/`. 2025-26 was the holdout and has now been used once.

## Code map

| Path | What |
| --- | --- |
| `bucketweights/espn.py` | ESPN fetch (retries, two hosts) and `parse_summary` → game, shots, players |
| `bucketweights/backfill.py` | History seasons into the DB (`python -m bucketweights.backfill 2024 2025 2026`) |
| `bucketweights/db.py` | SQLite schema (`games`, `shots`, `players`, `appearances`, `runs`), `write_game`, hashes |
| `bucketweights/model.py` | levels, `shot_luck`, `game_verdicts`, `player_table`, `hot_cold`, calibration |
| `bucketweights/update.py` | nightly fetch + checks + writes `build/data/*.json` (meta, players, hotcold, last_night, games, teams) |
| `bucketweights/cards.py` | the day's social cards: plan → card pages (`/social/cards/<slug>/`) → Playwright screenshots → `/social/` kit page + `posts.json` |
| `bucketweights/site.py` | Jinja build into `build/site` (home, players + `/players/{id}/`, games + `/games/{id}/`, one Teams table (no per-team pages: Josh, 9 Oct 2026), guide, 404) |
| `bucketweights/teams.json` | the 30 teams: ESPN id/abbrev, NBA abbrev, names, conference, division |
| `site/templates/` | `base.html` (header, Teams dialog, bottom nav, footer), `macros.html`, one template per page, `social.html` (kit), `cards/` (one per card) |
| `site/static/` | `site.css`, `site.js` (theme, Teams dialog, sortable tables, player filters), `cards.css` (social cards), fonts, `icon.svg` |
| `tests/test_pipeline.py` | exact maths, shot rules, ESPN fixture parse, idempotent writes |
| `docs/` | `methodology.md`, `design-references.md`, `bucketweights-style-guide.html` (approved mock-up), `bucketweights-social-cards.html` (card concepts) |

## Commands (local: Python 3.9 venv; GitHub uses 3.12, so keep code 3.9-compatible)

```bash
.venv/bin/python -m bucketweights.update                  # fetch + data (add --preseason, --no-fetch, --season 2026)
.venv/bin/python -m bucketweights.site                    # build (add --season 2026 for last season's full site)
python3 scripts/serve.py                                  # http://localhost:4200
.venv/bin/python -m unittest discover tests
```

For a realistic full-season preview: `update --no-fetch --season 2026` then `site --season 2026`.
Preview in the app's browser: launch config **`bucketweights-site`** (port 4200) lives in the
Claude Work folder's `.claude/launch.json`; `bucketweights-docs` (4100) serves `docs/`. Python's
`http.server -m` fails there because the session's working directory is in iCloud; `scripts/serve.py`
avoids that.

## Design (Josh's calls, 8–9 Oct 2026; details in `docs/design-references.md`)

- **"The type is the photo":** taken from real 1990s SLAM covers and vintage print, not generic
  retro. Josh rejected AI clichés: no tape, stamps, stickers, starbursts, grain, halftone overlays,
  tilted boxes, marker fonts, Anton/Bowlby. **Only ornaments: ★ stars and stacked stripe rules.**
  One hard drop shadow per screen at most. Square corners.
- **Fonts (free, self-hosted):** League Gothic (logo/headlines), Libre Franklin (text; Black
  Italic for big names), Courier Prime (card backs, notes), Big Shoulders Stencil (rotating
  coverline face). No Google Fonts at runtime.
- **Colour:** SLAM red #E2231A, yellow #FFD200, cyan #00A3E0, ink, page #FAF8F2, newsprint
  #EFEADB, card board #CFC8B8. **Hot (#E8410C) / cold (#0A6FD6) instead of green/red** (text
  versions hot-ink/cold-ink). Each night is an "issue" with its own flat cover colour, rotating
  yellow, white, cyan, black, orange, stone (`site.ISSUE_COLOURS`). Paper / Blacktop (dark) toggle.
- **Header (like HockeyWeights):** sticky strap in the issue colour: "30 Teams ▾" (table of
  contents dialog by conference/division) on the left; main menu Home ★ Players ★ Games ★ Teams ★
  Guide on the right; on phones a bottom icon bar instead.
  Each team in it opens the Players page filtered to that team (`players/?team=BOS`: whole roster,
  with the team's line from the Teams table on top); there are no team pages.
- **Home:** cover = logo (no player name over it: Josh removed that) + dateline, then **Heat Check
  / Defrost first** (Josh: hottest and coldest players at the top), then **Last night** (agate,
  robberies first), then **Last Shot**. Before opening night the lists show last season's
  players, unlinked.
- **Inside pages are print:** agate tables with dot leaders, player pages as card backs, Guide as
  a letters page.
- **No logos or headshots.** Players are their names set big; teams are names/abbrevs. **Team
  colours only on the social cards about one player or one game** (Josh, 9 Oct 2026): the cover
  and the card back in the featured player's team colours, the robbery card in the winning team's (`teams.json` `colours`: main, second; ESPN's
  with Warriors, Knicks, 76ers and Heat corrected). `cards.team_inks` nudges a colour lighter or
  darker (keeping its hue) until it reads, else white/ink; a test checks all 30 teams. The site,
  the agate and the yearbook keep their own colours.
- **Social cards (built 9 Oct 2026, `bucketweights/cards.py`; Josh posts by hand):** the five approved
  looks at 1080 × 1350 (authored at that size, shot at 1×). After a night with games: **cover**
  (tonight's issue in the issue colour; cover story = Defrost No. 1; coverlines = Heat check No. 1
  and the night's robbery or closest game; the main line and its typeface rotate by issue,
  `COVERLINES` / `.face-0`–`3`), **last night** (agate, every game, lowest winner's chance first;
  fits 15 games), **robbery of the night** (winner's colours, only when there was one), **one card
  back** (Defrost on odd issues, Heat check on even; the highest on that list without a card back
  in the last 14 days, kept in the DB's `cards` table), and the **yearbook** top 5 (Heat check
  Mondays, Defrost Thursdays, by the morning after). Nothing is made when the latest games are
  older than yesterday. Before opening night the player cards use last season's lists (labelled
  "Last season"). Long names/titles shrink to fit (`data-fit-width`); content that runs into the
  footer gets tighter spacing (`data-fit`) and is flagged "CHECK". Images and captions at
  `/social/` (noindex, not in the menus); until the site is live Josh downloads them from the run
  page's `social-cards` artifact. Previews: `cards --any-date --all` after a `--season 2026` build.

### Vocabulary

| Say | Don't say |
| --- | --- |
| Defrost · bounce-back · better than this | Due · owed · bound to |
| Heat check · cool-off · won't last | Fluke · fraud |
| True level · his level | True talent (on the site) |
| Robbery · coin flip · earned it | Rigged · deserved to lose |
| At the same shots and minutes | Will score |

## Next steps (as of 9 Oct 2026)

1. Josh to push, then run "Update site" with **preseason** ticked and download the `social-cards`
   artifact to look at the real cards (first run with Playwright on GitHub).
2. Josh's review of the cards: which to post, the weekly days, the cover's main lines, the
   Defrost yearbook's red stock (the approved mock only had Heat check).
3. Polish: Open Graph images/meta for sharing (a card could double as the page's share image), a
   sitemap, the empty "Last night" date line before opening night, preseason games' pages.
4. When Josh says go: Cloudflare secrets + domain (bucketweights.com?) → publish. Change
   `cards.SITE_NAME` if the domain differs.
5. Later in the season: playoff/Play-In odds on point differential (Cup games count except the
   final), and the "what's at stake" swings; possibly the NBA Cup knockout placeholders.
