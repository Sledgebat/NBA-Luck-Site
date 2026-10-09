"""
Offline tests: python -m unittest discover tests

The exact score maths, the shot rules, reading a saved ESPN game, and that storing a game twice
changes nothing.
"""
import gzip
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bucketweights import db, espn, model
from bucketweights import shots as S

FIX = Path(__file__).with_name("fixtures")


class ExactMaths(unittest.TestCase):
    def test_points_pmf_matches_brute_force(self):
        p = np.array([0.3, 0.8, 0.45, 0.9])
        v = np.array([3, 1, 3, 1])
        pmf = model.points_pmf(p, v)
        brute = np.zeros(int(v.sum()) + 1)
        for outcome in itertools.product([0, 1], repeat=len(p)):
            o = np.array(outcome)
            brute[int((o * v).sum())] += np.prod(np.where(o == 1, p, 1 - p))
        np.testing.assert_allclose(pmf, brute, atol=1e-12)
        self.assertAlmostEqual(pmf.sum(), 1.0)

    def test_chance_symmetry_and_ties(self):
        # identical teams: the home side's raw chance is exactly a half
        p, v = [0.36] * 30 + [0.78] * 20, [3] * 30 + [1] * 20
        chance, tie = model.raw_home_chance(80, p, v, 80, p, v)
        self.assertAlmostEqual(chance, 0.5, places=12)
        self.assertGreater(tie, 0)

    def test_calibration_and_labels(self):
        self.assertGreater(model.calibrate(0.5), 0.5)   # home edge
        self.assertGreater(model.calibrate(0.02), 0.02)  # pulled toward a half
        self.assertEqual(model.label(0.2), "Robbery")
        self.assertEqual(model.label(0.5), "Coin flip")
        self.assertEqual(model.label(0.8), "Earned it")


class ShotRules(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(S.kind_of("Free Throw - 1 of 2", "", None, True, 1, None, None), "FT")
        self.assertEqual(S.kind_of("Jump Shot", "", 3, False, 0, None, None), "3")
        self.assertEqual(S.kind_of("Jump Shot", "X misses 26-foot three point jumper", None, False, 0, None, None), "3")
        self.assertEqual(S.kind_of("Jump Shot", "X makes 18-foot jumper", None, True, 2, None, None), "2")

    def test_zones(self):
        self.assertEqual(S.zone_of("2", 2.0, "Driving Layup Shot", 5), "rim")
        self.assertEqual(S.zone_of("3", 22.0, "Jump Shot", 3), "corner")
        self.assertEqual(S.zone_of("3", 25.0, "Jump Shot", 25), "above")
        self.assertEqual(S.zone_of("3", 60.0, "Jump Shot", 70), "heave")
        self.assertEqual(S.zone_of("3", 30.0, "Heave Jump Shot", 30), "heave")


class EspnGame(unittest.TestCase):
    def setUp(self):
        self.j = json.loads(gzip.open(FIX / "summary_401810723.json.gz", "rt").read())

    def test_parse_and_box_check(self):
        from bucketweights.update import check_game
        game, shots, players = espn.parse_summary(self.j, "2026-03-01", 2026, 2, "STD")
        self.assertEqual(game["game_id"], 401810723)
        self.assertEqual(check_game(game, shots, espn.box_totals(self.j)), "ok")
        self.assertTrue(players)
        self.assertEqual(game["home_reg"] + game["away_reg"] > 150, True)

    def test_storing_twice_changes_nothing(self):
        game, shots, players = espn.parse_summary(self.j, "2026-03-01", 2026, 2, "STD")
        with tempfile.TemporaryDirectory() as d:
            con = db.connect(Path(d) / "t.sqlite")
            for _ in range(2):
                db.write_game(con, game, shots, players, "espn", "ok", "now")
            n_games = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            n_shots = con.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
            self.assertEqual(n_games, 1)
            self.assertEqual(n_shots, len(shots))
            self.assertEqual(db.stored_hash(con, game["game_id"]), db.content_hash(game, shots))



class CardWords(unittest.TestCase):
    """The social cards' sentences (bucketweights/cards.py); the images themselves need a browser."""

    def game(self, hp, ap, h3, a3, chance=0.2, periods=4):
        return {"home": "BOS", "away": "ORL", "home_pts": hp, "away_pts": ap, "periods": periods,
                "winner_chance": chance, "teams": {"home": {"three": h3}, "away": {"three": a3}}}

    def test_score_line_puts_the_winner_first(self):
        from bucketweights import cards
        self.assertEqual(cards.score_line(self.game(108, 113, [12, 43, 15.0], [19, 50, 17.9])), "Magic 113, Celtics 108")
        self.assertEqual(cards.score_line(self.game(110, 109, [1, 1, 1], [1, 1, 1], periods=5)), "Celtics 110, Magic 109 (OT)")

    def test_night_story_tells_the_bigger_surprise(self):
        from bucketweights import cards
        cold_loser = self.game(113, 108, [19, 50, 17.9], [12, 43, 15.0])
        self.assertEqual(cards.night_story(cold_loser), "Orlando shot 12-for-43 from three (15.0 expected) and lost by 5.")
        hot_winner = self.game(124, 123, [17, 27, 9.3], [10, 30, 11.0])
        self.assertEqual(cards.night_story(hot_winner), "Boston shot 17-for-27 from three (9.3 expected) and won by one.")

    def test_chance_and_names(self):
        from bucketweights import cards
        self.assertEqual(cards.chance_short(0.004), "<1%")
        self.assertEqual(cards.chance_short(0.312), "31%")
        self.assertEqual(cards.last_name("Tim Hardaway Jr."), "Hardaway")
        self.assertEqual(cards.last_name("Nikola Jokić"), "Jokić")

    def test_key_kind_follows_the_list(self):
        from bucketweights import cards
        p = {"gp": 10, "3": {"pct": 0.30, "level": 0.38, "a": 60}, "2": {"pct": 0.56, "level": 0.52, "a": 100},
             "FT": {"pct": 0.80, "level": 0.80, "a": 30}}
        self.assertEqual(cards.key_kind(p, 1), "3")    # Defrost: the threes are below his level
        self.assertEqual(cards.key_kind(p, -1), "2")   # Heat check: the twos are above it


if __name__ == "__main__":
    unittest.main()
