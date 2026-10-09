"""
The day's social cards: 1080 × 1350 images for Josh to post by hand.

    python -m bucketweights.cards                    # after update and site, as the nightly run does
    python -m bucketweights.cards --any-date --all   # every card type from the latest games (previews)
    python -m bucketweights.cards --no-images        # card pages only, no screenshots

Reads the data files from update.py, builds each card as its own page (build/site/social/cards/),
screenshots them with Playwright into build/site/social/<date>/, and builds the /social/ page
with the images and a caption for each. The looks are the five Josh approved on 8 Oct 2026
(docs/bucketweights-social-cards.html).

Which cards, after a night with games:
  every day     the cover (tonight's issue) and last night (agate)
  robberies     robbery of the night (the lowest winner's chance), only when there was one
  every day     one card back: Defrost on odd issues, Heat check on even; the highest on that
                list without a card back in the last REPEAT_DAYS days
  Mon / Thu     the yearbook: Heat check / Defrost top 5 (by the morning after the games)
Nothing is made when the latest games are older than yesterday (an off night).
"""
from __future__ import annotations

import argparse
import colorsys
import functools
import http.server
import json
import os
import shutil
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from . import db, model, site, update

SITE_NAME = "bucketweights.com"  # until the domain is set
OUT = site.OUT / "social"
CARD_W, CARD_H = 1080, 1350
REPEAT_DAYS = 14
YEARBOOK = {0: "heat_check", 3: "defrost"}  # weekday of the morning after: Monday, Thursday
LISTS = {"heat_check": "Heat check", "defrost": "Defrost"}
KIND = {  # label, "52.5% ___", plural
    "3": ("3PT", "from three", "threes"),
    "FT": ("FT", "at the line", "free throws"),
    "2": ("2PT", "on twos", "twos"),
}
POSITION = {"G": "Guard", "F": "Forward", "C": "Center"}
# The cover's main line, one per issue in turn. The cover story is always a cold shooter.
COVERLINES = ["cold, not broke.", "better than this.", "the shots are fine.", "a slump, not a ceiling.", "bounce-back watch."]
FACES = 4  # the main line's typeface rotates too (cards.css .face-0 to .face-3)
FIT_STEPS = ("tight", "tighter")


# ---------------------------------------------------------------- words


def pct(x, digits=1) -> str:
    return site.pct(x, digits)


def chance(c: float) -> str:
    n = round(100 * c)
    return "under 1%" if n < 1 else f"{n}%"


def chance_short(c: float) -> str:
    n = round(100 * c)
    return "<1%" if n < 1 else f"{n}%"


def nickname(abbrev: str) -> str:
    return site.team_name(abbrev)


def location(abbrev: str) -> str:
    t = site.BY_ABBREV.get(abbrev)
    return t["location"] if t else abbrev


def by_margin(n: int) -> str:
    return "one" if n == 1 else str(n)


def sides(g: dict) -> tuple[str, str]:
    """(winner, loser) as "home"/"away"."""
    return ("home", "away") if g["home_pts"] > g["away_pts"] else ("away", "home")


def score_line(g: dict) -> str:
    w, l = sides(g)
    ot = " (OT)" if g.get("periods", 4) > 4 else ""
    return f"{nickname(g[w])} {g[w + '_pts']}, {nickname(g[l])} {g[l + '_pts']}{ot}"


def key_kind(p: dict, sign: int) -> str:
    """The kind of shot that moves his scoring most in the list's direction (+1 Defrost, −1 Heat check)."""
    def pull(k):
        d = p[k]
        if d["pct"] is None or not p["gp"]:
            return 0.0
        return model.VALUE[k] * (d["level"] - d["pct"]) * d["a"] / p["gp"] * sign
    return max(KIND, key=pull)


def shooting_line(p: dict, k: str) -> str:
    """'32.0% from three, level 38.5%'"""
    return f"{pct(p[k]['pct'])} {KIND[k][1]}, level {pct(p[k]['level'])}"


def ppg_line(p: dict) -> str:
    return f"{p['ppg']:.1f} → {p['proj_ppg']:.1f}"


def last_name(name: str) -> str:
    parts = name.split()
    if len(parts) > 2 and parts[-1].rstrip(".") in ("Jr", "Sr", "II", "III", "IV"):
        return parts[-2]
    return parts[-1] if parts else name


def night_story(g: dict) -> str:
    """One sentence on why the lowest-chance winner got there: the bigger three-point surprise."""
    w, l = sides(g)
    margin = g[w + "_pts"] - g[l + "_pts"]
    wm, wa, wx = g["teams"][w]["three"]
    lm, la, lx = g["teams"][l]["three"]
    if (lx - lm) >= (wm - wx):
        return f"{location(g[l])} shot {lm}-for-{la} from three ({lx:.1f} expected) and lost by {by_margin(margin)}."
    return f"{location(g[w])} shot {wm}-for-{wa} from three ({wx:.1f} expected) and won by {by_margin(margin)}."


# ---------------------------------------------------------------- team colours
# The player cards (cover, card back) are printed in the player's team colours (Josh, 9 Oct 2026).
# Team pairs often clash, so a colour is lightened or darkened, keeping its hue, until it reads.

BOARD, PAPER, INK, WHITE = "#CFC8B8", "#FAF8F2", "#141414", "#FFFFFF"


def _rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _hex(rgb) -> str:
    return "#" + "".join(f"{round(max(0, min(1, v)) * 255):02X}" for v in rgb)


def luminance(h: str) -> float:
    lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in _rgb(h)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def readable(colour: str, against: str, target: float, max_shift: float = 1.0, fallback: str | None = None) -> str:
    """The colour, made lighter or darker (away from the background) until it reaches the contrast.
    If that takes more than `max_shift` of lightness it would stop looking like itself: `fallback`
    (else plain ink or white) instead."""
    if contrast(colour, against) >= target:
        return colour
    h, l0, sat = colorsys.rgb_to_hls(*_rgb(colour))
    step = 0.01 if luminance(colour) >= luminance(against) else -0.01
    l = l0
    while 0.0 < l < 1.0 and abs(l - l0) <= max_shift:
        l = max(0.0, min(1.0, l + step))
        c = _hex(colorsys.hls_to_rgb(h, l, sat))
        if contrast(c, against) >= target:
            return c
    return fallback or max((INK, WHITE), key=lambda x: contrast(x, against))


def is_grey(colour: str) -> bool:
    r, g, b = _rgb(colour)
    return max(r, g, b) - min(r, g, b) < 0.12


def team_inks(abbrev: str) -> dict:
    """CSS variables for a team's cover and card back."""
    t = site.BY_ABBREV.get(abbrev)
    if not t:
        return {}
    main, second = t["colours"]
    ink = max((WHITE, INK), key=lambda x: contrast(x, main))
    # cover: the second colour for the wordmark if it reads (3:1 for big type) with a small nudge, else the ink
    logo = readable(second, main, 3.0, max_shift=0.12, fallback=ink)
    # card back: the darker-reading colour prints the text, the other fills the badge and the thick rule.
    # A coloured fill stays as it is (gold on grey board reads by its hue); only a grey one is shifted.
    text, fill = sorted((main, second), key=lambda c: contrast(c, BOARD), reverse=True)
    ink1 = readable(text, BOARD, 4.5)
    ink2 = readable(fill, BOARD, 1.6) if is_grey(fill) else fill
    return {
        "cover": {"--issue": main, "--ink": ink, "--logo": logo},
        "back": {"--ink1": ink1, "--ink2": ink2,
                 "--ink2-text": max((ink1, WHITE, INK), key=lambda x: contrast(x, ink2)),
                 "--accent": readable(fill, BOARD, 4.5, max_shift=0.15, fallback=ink1)},
    }


def style(vars_: dict | None) -> str:
    return "; ".join(f"{k}: {v}" for k, v in (vars_ or {}).items())


# ---------------------------------------------------------------- the plan


def recent_backs(con, day: str) -> set[int]:
    since = (date.fromisoformat(day) - timedelta(days=REPEAT_DAYS)).isoformat()
    rows = con.execute("SELECT subject FROM cards WHERE card = 'back' AND day >= ? AND day < ?", (since, day)).fetchall()
    return {r["subject"] for r in rows}


def card_back(con, lists: dict, which: str, season: int, day: str, lines) -> dict | None:
    """One player's card back: the highest on the list who hasn't had one lately."""
    ranked = lists[which]
    if not ranked:
        return None
    skip = recent_backs(con, day)
    rank, p = next(((i + 1, x) for i, x in enumerate(ranked) if x["id"] not in skip), (1, ranked[0]))
    sign = 1 if which == "defrost" else -1
    k = key_kind(p, sign)
    label, words, plural = KIND[k]
    mine = lines[lines.player_id == p["id"]].sort_values("season")
    rows = []
    for r in mine.itertuples():
        m, a = int(getattr(r, f"m{k}", 0)), int(getattr(r, f"a{k}", 0))
        t = site.BY_ESPN.get(int(r.team_id)) if r.team_id else None
        rows.append({"season": f"{int(r.season) - 1}-{str(int(r.season))[2:]}", "now": int(r.season) == season,
                     "team": t["abbrev"] if t else "", "ppg": r.pts / r.gp if r.gp else 0, "m": m, "a": a,
                     "pct": m / a if a else None})
    before = [r for r in rows if not r["now"]]
    m0, a0 = sum(r["m"] for r in before), sum(r["a"] for r in before)
    level, now = p[k]["level"], p[k]["pct"]
    if a0 >= 50:
        n = len(before)
        span = "last season" if n == 1 else f"the {['', '', 'two', 'three', 'four'][n] if n <= 4 else n} seasons before this one"
        trivia = (f"He made {pct(m0 / a0)} of his {plural} over {span}. That's why his level sits at "
                  f"{pct(level)}, not {pct(now)}.")
    else:
        trivia = (f"With few {plural} on record, his level leans on the league average: {pct(level)}, "
                  f"not this season's {pct(now)}.")
    team = site.BY_ABBREV.get(p["team"])
    con.execute("INSERT OR REPLACE INTO cards VALUES (?, 'back', ?)", (day, p["id"]))
    con.commit()
    title = f"{LISTS[which]} No. {rank}"
    return {
        "style": style(team_inks(p["team"]).get("back")),
        "slug": f"back-{which.replace('_', '-')}", "template": "cards/back.html", "title": f"Card back · {title}: {p['name']}",
        "p": p, "rank": rank, "which": which, "list_title": LISTS[which], "kind": k, "label": label, "rows": rows,
        "bio": " · ".join(x for x in (POSITION.get(p["pos"], p["pos"]), team["name"] if team else p["team"]) if x).upper(),
        "range": f"{label} level {pct(level)} (likely {pct(p[k]['lo'])}–{pct(p[k]['hi'])})".upper(),
        "trivia": trivia, "to": f"To {site.short_date(day)}" if season == model.season_of(day) else f"{rows[-1]['season'] if rows else ''} season",
        "caption": (f"{title}: {p['name']}. {pct(now)} {words} this season, his level {pct(level)}. "
                    f"{p['ppg']:.1f} points a game now, {p['proj_ppg']:.1f} at his true shooting, with the same shots "
                    f"and minutes. {SITE_NAME}"),
    }


def yearbook(lists: dict, which: str, season_label: str, day: str) -> dict | None:
    ranked = lists[which][:5]
    if not ranked:
        return None
    sign = 1 if which == "defrost" else -1
    items = [{"p": p, "line": f"{shooting_line(p, key_kind(p, sign))} · {ppg_line(p)} PPG"} for p in ranked]
    cap = "; ".join(f"{i + 1}. {x['p']['name']}, {x['p']['ppg']:.1f} → {x['p']['proj_ppg']:.1f}" for i, x in enumerate(items))
    lead = ("Shooting above their level. Expect a cool-off." if which == "heat_check"
            else "Shooting below their level. Expect a bounce-back.")
    return {
        "slug": f"yearbook-{which.replace('_', '-')}", "template": "cards/yearbook.html", "title": f"Yearbook · {LISTS[which]}",
        "which": which, "list_title": LISTS[which], "items": items, "season_label": season_label,
        "sub": "Above their level · Expect a cool-off" if which == "heat_check" else "Below their level · Expect a bounce-back",
        "caption": f"{LISTS[which]}. {lead} Points a game now → at their true shooting, same shots and minutes: {cap}. {SITE_NAME}",
    }


def plan(con, any_date: bool = False, every: bool = False) -> dict:
    meta, hot, ln = site.read("meta.json"), site.read("hotcold.json"), site.read("last_night.json")
    season = meta["season"]
    games = sorted(ln.get("games") or [], key=lambda g: g["winner_chance"])
    if not games:
        return {"date": None, "cards": [], "skipped": ["No games yet: nothing to post."]}
    day = ln["date"]
    if not any_date and date.fromisoformat(day) < update.today_et() - timedelta(days=1):
        return {"date": day, "cards": [], "skipped": [f"The latest games are from {site.long_date(day)}: nothing new to post."]}
    morning = date.fromisoformat(day) + timedelta(days=1)
    lists, list_season, carried = hot, season, False
    if not hot["heat_check"] and not hot["defrost"]:
        lists = site.carried_lists(con, season)
        list_season, carried = lists["season"], True
    issue = meta.get("issue") or 0
    colour = site.ISSUE_COLOURS[issue % len(site.ISSUE_COLOURS)] if issue else "yellow"
    issue_label = f"No. {issue}" if issue else "Preseason"
    cards, skipped = [], []

    # 1. the cover
    if lists["defrost"]:
        star = lists["defrost"][0]
        k = key_kind(star, 1)
        robbery = next((g for g in games if g["label"] == "Robbery"), None)
        lines = []
        if lists["heat_check"]:
            h = lists["heat_check"][0]
            hk = key_kind(h, -1)
            lines.append(("Heat check", f"{last_name(h['name'])}: {pct(h[hk]['pct'])} {KIND[hk][1]} won't last"))
        lines.append(("Robbery!", score_line(robbery)) if robbery else (games[0]["label"], score_line(games[0])))
        cards.append({
            "slug": "cover", "template": "cards/cover.html", "title": "The cover · tonight's issue",
            "style": style(team_inks(star["team"]).get("cover")),
            "colour": colour, "issue_label": issue_label, "p": star, "team": nickname(star["team"]),
            "who": f"{KIND[k][0]} {pct(star[k]['pct'])}, his level {pct(star[k]['level'])}",
            "coverlines": lines, "main": COVERLINES[issue % len(COVERLINES)], "face": issue % FACES,
            "season_note": lists["season_label"] if carried else "",
            "caption": (f"BucketWeights {issue_label} ★ {site.long_date(day)}. Cover story: {star['name']} is shooting "
                        f"{pct(star[k]['pct'])} {KIND[k][1]}; his level is {pct(star[k]['level'])}. At his true shooting "
                        f"that's {star['proj_ppg']:.1f} points a game, not {star['ppg']:.1f}, with the same shots and minutes. "
                        f"{SITE_NAME}"),
        })

    # 2. last night, agate
    n_rob = sum(g["label"] == "Robbery" for g in games)
    n_coin = sum(g["label"] == "Coin flip" for g in games)
    best = games[0]
    cards.append({
        "slug": "last-night", "template": "cards/agate.html", "title": "Last night · the agate",
        "games": games, "best_title": "Robbery of the night" if best["label"] == "Robbery" else "Closest call",
        "best": night_story(best),
        "caption": (f"Last night, weighed: {len(games)} game{'s' if len(games) != 1 else ''}, {n_rob} robber{'y' if n_rob == 1 else 'ies'}, "
                    f"{n_coin} coin flip{'s' if n_coin != 1 else ''}. The % is the winner's chance at both teams' true shooting. {SITE_NAME}"),
    })

    # 3. robbery of the night
    if best["label"] == "Robbery":
        w, l = sides(best)
        hero = max((s for s in best["swing"] if s["team"] == best[w] and s["luck_pts"] > 0 and s["three"][1]),
                   key=lambda s: s["luck_pts"], default=None)
        goat = min((s for s in best["swing"] if s["team"] == best[l] and s["luck_pts"] < 0 and s["three"][1]),
                   key=lambda s: s["luck_pts"], default=None)
        cards.append({
            "slug": "robbery", "template": "cards/robbery.html", "title": "Robbery of the night",
            "g": best, "w": w, "l": l, "score": score_line(best), "chance": chance_short(best["winner_chance"]),
            "winner_place": location(best[w]), "hero": hero, "goat": goat,
            "caption": (f"Robbery! {score_line(best)}. At both teams' true shooting, the {nickname(best[w])} win this "
                        f"{chance(best['winner_chance'])} of the time. {night_story(best)} {SITE_NAME}"),
        })
    else:
        skipped.append("Robbery of the night: no robbery last night.")

    # 4. a card back
    lines_df = site.season_lines(con, list_season)
    backs = ["defrost", "heat_check"] if every else (["defrost"] if issue % 2 == 1 or not issue else ["heat_check"])
    for which in backs:
        c = card_back(con, lists, which, list_season, day, lines_df)
        if c:
            c["carried"] = carried
            cards.append(c)
        else:
            skipped.append(f"Card back: the {LISTS[which]} list is empty.")

    # 5. the yearbook
    books = list(YEARBOOK.values()) if every else ([YEARBOOK[morning.weekday()]] if morning.weekday() in YEARBOOK else [])
    for which in books:
        label = lists["season_label"] if carried else meta["season_label"]
        c = yearbook(lists, which, label, day)
        if c:
            cards.append(c)
    if not books:
        skipped.append("Yearbook: Heat check on Mondays, Defrost on Thursdays.")

    for c in cards:
        c["image"] = f"{day}/{c['slug']}.png"
    return {"date": day, "issue_label": issue_label, "colour": colour, "carried": carried,
            "season_label": meta["season_label"], "cards": cards, "skipped": skipped}


# ---------------------------------------------------------------- pages and images


def write_pages(p: dict, built: str) -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    # the card styles, in case they changed since the site was built
    shutil.copy2(site.SITE / "static" / "cards.css", site.OUT / "static" / "cards.css")
    e = site.env()
    e.globals.update(score_line=score_line, chance_short=chance_short)
    common = {"meta": site.read("meta.json"), "issue_colour": p.get("colour", "yellow"), "built": built,
              "SITE_NAME": SITE_NAME, "day": p["date"], "plan": p}
    for c in p["cards"]:
        f = OUT / "cards" / c["slug"] / "index.html"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(e.get_template(c["template"]).render(root="../../../", c=c, **common))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(e.get_template("social.html").render(root="../", nav="", **common))


def serve(root: Path):
    handler = functools.partial(QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def shoot(p: dict) -> tuple[list[str], list[str]]:
    """Screenshot every card. Content that runs into the bottom lines gets tighter spacing first."""
    from playwright.sync_api import sync_playwright

    made, check = [], []
    server = serve(site.OUT)
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": CARD_W, "height": CARD_H}, device_scale_factor=1, color_scheme="light")
            for c in p["cards"]:
                page.goto(f"{origin}/{OUT.relative_to(site.OUT).as_posix()}/cards/{c['slug']}/", wait_until="networkidle")
                page.evaluate("document.fonts.ready.then(() => window.fitWidths && window.fitWidths())")
                card = page.locator("[data-card]")
                over = lambda: card.evaluate(
                    """el => { const f = el.querySelector('.flow'), floor = [...el.querySelectorAll('[data-floor]')];
                         if (!f || !floor.length) return 0;
                         return Math.ceil(f.getBoundingClientRect().bottom - Math.min(...floor.map(x => x.getBoundingClientRect().top)) + 16); }""")
                o = over()
                for step in FIT_STEPS:
                    if o <= 0:
                        break
                    card.evaluate("(el, s) => el.setAttribute('data-fit', s)", step)
                    o = over()
                if o > 0:
                    check.append(f"{c['slug']}: runs {o}px into the bottom lines, even tightened")
                f = OUT / c["image"]
                f.parent.mkdir(parents=True, exist_ok=True)
                card.screenshot(path=str(f))
                made.append(c["image"])
            browser.close()
    finally:
        server.shutdown()
    return made, check


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--any-date", action="store_true", help="make cards even when the latest games are old")
    ap.add_argument("--all", action="store_true", help="every card type, whatever the day (for checking)")
    ap.add_argument("--no-images", action="store_true", help="card pages only")
    args = ap.parse_args(argv)
    started = time.time()
    con = db.connect()
    p = plan(con, args.any_date, args.all)
    built = datetime.now(update.ET).strftime("%-d %b %Y, %-I:%M %p ET")
    write_pages(p, built)
    made, check = ([], []) if args.no_images or not p["cards"] else shoot(p)
    (OUT / "posts.json").write_text(json.dumps(
        {"date": p["date"], "cards": [{k: c[k] for k in ("slug", "title", "image", "caption")} for c in p["cards"]],
         "skipped": p["skipped"], "check": check}, indent=1, ensure_ascii=False))
    print(f"Cards for {p['date']}: {len(made)} images in {time.time() - started:.1f} s")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:  # shown on the run's page on GitHub
        with open(summary, "a") as f:
            f.write(f"### Social cards for {p['date']}\n\n" + "".join(f"- {c['title']}\n" for c in p["cards"])
                    + "".join(f"- Not made: {x}\n" for x in p["skipped"]) + "".join(f"- **Check:** {x}\n" for x in check))
    for f in made:
        print(f"  made     {f}")
    for s in p["skipped"]:
        print(f"  skipped  {s}")
    for s in check:
        print(f"  CHECK    {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
