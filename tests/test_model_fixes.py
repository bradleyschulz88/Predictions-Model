"""Regression tests for scoring-path correctness fixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlb_predictions  # noqa: E402
from mlb_predictions import (  # noqa: E402
    _last_five_pct,
    _lineup_logit_adjustment,
    apply_predictions,
    extract_spread_line,
    extract_spread_price,
    extract_total_price,
    kelly_band_probability,
    predict_spread,
)
from shared_utils import win_pct_from_record, win_pct_or_none  # noqa: E402


class SpreadSignTests(unittest.TestCase):
    """extract_spread_line promises the home spread; it must not return away's."""

    def _lines(self, current: dict) -> list[dict]:
        return [{"viewType": "Spread", "currentLine": current}]

    def test_reads_home_spread_directly(self) -> None:
        self.assertEqual(extract_spread_line(self._lines({"home": "-3.5", "away": "+3.5"})), -3.5)

    def test_flips_sign_when_only_away_is_present(self) -> None:
        # Away +7 means the home side is a 7-point favourite: -7, not +7.
        self.assertEqual(extract_spread_line(self._lines({"away": "+7"})), -7.0)

    def test_handles_unicode_minus(self) -> None:
        self.assertEqual(extract_spread_line(self._lines({"home": "−2.5"})), -2.5)

    def test_returns_none_without_spread_lines(self) -> None:
        self.assertIsNone(extract_spread_line([{"viewType": "MoneyLine", "currentLine": {}}]))


class SidePriceTests(unittest.TestCase):
    """SBR's totals/spread lines are bare numbers with nowhere to carry a
    price; ESPN core odds embed it as parenthetical text on the line itself.
    These extractors must read that text and stay silent when it isn't there."""

    def test_total_price_reads_the_espn_core_parenthetical(self) -> None:
        lines = [{"viewType": "Total", "currentLine": {"over": "o8.5 (-110)", "under": "u8.5 (-110)"}}]
        self.assertEqual(extract_total_price(lines, "over"), -110)
        self.assertEqual(extract_total_price(lines, "under"), -110)

    def test_total_price_is_none_on_an_sbr_bare_number(self) -> None:
        lines = [{"viewType": "Total", "currentLine": {"over": "o8.5", "under": "u8.5"}}]
        self.assertIsNone(extract_total_price(lines, "over"))

    def test_total_price_rejects_an_invalid_side(self) -> None:
        lines = [{"viewType": "Total", "currentLine": {"over": "o8.5 (-110)"}}]
        self.assertIsNone(extract_total_price(lines, "push"))

    def test_spread_price_reads_the_espn_core_parenthetical(self) -> None:
        lines = [{"viewType": "Spread", "currentLine": {"home": "-1.5 (+105)", "away": "+1.5 (-125)"}}]
        self.assertEqual(extract_spread_price(lines, "home"), 105)
        self.assertEqual(extract_spread_price(lines, "away"), -125)

    def test_spread_price_is_none_on_an_sbr_bare_number(self) -> None:
        lines = [{"viewType": "Spread", "currentLine": {"home": "-1.5", "away": "+1.5"}}]
        self.assertIsNone(extract_spread_price(lines, "home"))

    def test_spread_price_ignores_non_spread_lines(self) -> None:
        lines = [{"viewType": "Total", "currentLine": {"home": "-1.5 (+105)"}}]
        self.assertIsNone(extract_spread_price(lines, "home"))


class KellyBandProbabilityTests(unittest.TestCase):
    """Kelly stake sizing should defer to the graded record for a confidence
    level once there is enough of one, and otherwise leave the model alone."""

    def _report(self, reliability: list[dict]) -> dict:
        return {"reliability": reliability}

    def test_a_reliable_band_overrides_with_its_actual_win_rate(self) -> None:
        report = self._report(
            [{"range": "85-90", "picks": 40, "actualWinPct": 52.9, "avgPredictedPct": 87.9}]
        )
        with mock.patch.object(mlb_predictions, "_get_evaluation_report", return_value=report):
            self.assertAlmostEqual(kelly_band_probability(87.0), 0.529)

    def test_a_thin_band_defers_to_the_model(self) -> None:
        """Fewer picks than MIN_KELLY_BAND_SAMPLE is noise, not signal."""
        report = self._report(
            [{"range": "85-90", "picks": 4, "actualWinPct": 25.0}]
        )
        with mock.patch.object(mlb_predictions, "_get_evaluation_report", return_value=report):
            self.assertIsNone(kelly_band_probability(87.0))

    def test_no_matching_bucket_defers_to_the_model(self) -> None:
        report = self._report([{"range": "55-60", "picks": 100, "actualWinPct": 58.0}])
        with mock.patch.object(mlb_predictions, "_get_evaluation_report", return_value=report):
            self.assertIsNone(kelly_band_probability(87.0))

    def test_confidence_below_half_is_not_looked_up(self) -> None:
        """The reliability curve is folded onto the picked side, so it never
        carries data below 50% -- there is nothing to match against."""
        report = self._report([{"range": "55-60", "picks": 100, "actualWinPct": 58.0}])
        with mock.patch.object(mlb_predictions, "_get_evaluation_report", return_value=report):
            self.assertIsNone(kelly_band_probability(40.0))

    def test_missing_report_defers_to_the_model(self) -> None:
        with mock.patch.object(mlb_predictions, "_get_evaluation_report", return_value={}):
            self.assertIsNone(kelly_band_probability(87.0))

    def test_none_confidence_defers_to_the_model(self) -> None:
        self.assertIsNone(kelly_band_probability(None))


class LastFiveSentinelTests(unittest.TestCase):
    """A -1.0 sentinel reaching the logit is indistinguishable from real form."""

    def test_zero_zero_record_returns_none(self) -> None:
        self.assertIsNone(_last_five_pct("0-0"))

    def test_unparseable_record_returns_none(self) -> None:
        self.assertIsNone(_last_five_pct("no games"))
        self.assertIsNone(_last_five_pct(None))

    def test_normal_record_still_parses(self) -> None:
        self.assertAlmostEqual(_last_five_pct("3-2"), 0.6)

    def test_never_returns_negative(self) -> None:
        for record in ("0-0", "0-0-0", "", "—", "5-0"):
            value = _last_five_pct(record)
            if value is not None:
                self.assertGreaterEqual(value, 0.0, msg=record)


class SeasonStartRecordTests(unittest.TestCase):
    """An 0-0 club is not a .500 club.

    win_pct_from_record folds "0-0" into its 0.5 default, which is right for a
    display cell and wrong for a model input. Fed into a difference it yields
    exactly 0.0, indistinguishable from two genuinely average teams -- and
    splitDiff's home-field centring then turns that fabricated 0.0 into a
    measured 1.04pt edge against the home side on every opening-week game.
    NFL and NBA both open within ten weeks, so every game on the board would
    have carried it at once.
    """

    def test_win_pct_or_none_is_none_before_any_games(self) -> None:
        for record in ("0-0", "0-0-0", "", None, "—"):
            self.assertIsNone(win_pct_or_none(record, league="nfl"), msg=repr(record))

    def test_win_pct_or_none_matches_the_default_helper_once_played(self) -> None:
        for record, league in (("11-3-1", "nfl"), ("40-15", "nba"), ("20-8-10", "epl")):
            self.assertAlmostEqual(
                win_pct_or_none(record, league=league),
                win_pct_from_record(record, league=league),
                msg=record,
            )

    def test_a_genuine_five_hundred_club_is_not_confused_with_an_unplayed_one(self) -> None:
        """The whole point: 8-8 and 0-0 both read 0.5 through the old helper."""
        self.assertAlmostEqual(win_pct_or_none("8-8", league="nfl"), 0.5)
        self.assertIsNone(win_pct_or_none("0-0", league="nfl"))

    def _inputs(self, **over):
        game = {
            "league": "nfl", "homeTeam": "H", "awayTeam": "A",
            "homeRecord": "0-0", "awayRecord": "0-0",
            "homeHomeRecord": "0-0", "awayRoadRecord": "0-0",
            "enrichment": {},
        }
        game.update(over)
        return mlb_predictions.extract_model_inputs(game)

    def test_opening_night_logs_no_strength_features(self) -> None:
        features = self._inputs()
        self.assertIsNone(features["recordDiff"])
        self.assertIsNone(features["splitDiff"])

    def test_one_unplayed_side_is_still_unknown(self) -> None:
        """0-0 against 6-2 is not a 0.75 edge to the away side."""
        features = self._inputs(homeRecord="0-0", awayRecord="6-2")
        self.assertIsNone(features["recordDiff"])

    def test_real_records_are_unaffected(self) -> None:
        features = self._inputs(
            homeRecord="6-2", awayRecord="2-6",
            homeHomeRecord="4-0", awayRoadRecord="1-3",
        )
        self.assertAlmostEqual(features["recordDiff"], 0.5)
        self.assertAlmostEqual(features["splitDiff"], 0.75)

    def test_the_phantom_home_bias_is_gone(self) -> None:
        """End to end: two identical unplayed teams on a pick-em market must
        not have strengthDiff invent an edge against the home side."""
        from model_fit import build_feature_dict

        collapsed = build_feature_dict(self._inputs(), split_diff_centre=0.0364)
        self.assertIsNone(
            collapsed["strengthDiff"],
            "no games played means no strength evidence, not a -0.0182 edge",
        )


class RecordFormatTests(unittest.TestCase):
    """Three-part records mean different things in soccer and football."""

    def test_soccer_reads_middle_column_as_draws(self) -> None:
        # 10W 5D 2L -> (10 + 2.5) / 17
        self.assertAlmostEqual(win_pct_from_record("10-5-2", league="epl"), 12.5 / 17)

    def test_nfl_reads_last_column_as_ties(self) -> None:
        # 10W 5L 2T -> (10 + 1) / 17
        self.assertAlmostEqual(win_pct_from_record("10-5-2", league="nfl"), 11.0 / 17)

    def test_two_part_record_unaffected_by_league(self) -> None:
        for league in (None, "mlb", "epl", "nfl"):
            self.assertAlmostEqual(win_pct_from_record("10-5", league=league), 10 / 15)

    def test_zero_games_falls_back_to_default(self) -> None:
        self.assertEqual(win_pct_from_record("0-0-0", 0.42, league="epl"), 0.42)


class LineupScaleTests(unittest.TestCase):
    """Comparing two different metrics manufactures edge from data coverage."""

    def test_no_edge_when_sides_use_different_metrics(self) -> None:
        # Home has confirmed batting averages, away only a season OPS proxy.
        game = {
            "league": "mlb",
            "homeLineup": {"batters": [{"order": index, "avg": 0.250} for index in range(1, 10)]},
            "awayLineup": {},
        }
        enrichment = {"awayAdvanced": {"opsProxy": 0.720}}
        self.assertEqual(_lineup_logit_adjustment(game, "mlb", enrichment), 0.0)

    def test_edge_when_both_sides_use_the_same_metric(self) -> None:
        game = {"league": "mlb", "homeLineup": {}, "awayLineup": {}}
        enrichment = {
            "homeAdvanced": {"opsProxy": 0.780},
            "awayAdvanced": {"opsProxy": 0.680},
        }
        adjustment = _lineup_logit_adjustment(game, "mlb", enrichment)
        self.assertGreater(adjustment, 0.0)
        self.assertLessEqual(adjustment, 0.35)

    def test_league_average_inputs_produce_no_edge(self) -> None:
        game = {"league": "mlb", "homeLineup": {}, "awayLineup": {}}
        enrichment = {
            "homeAdvanced": {"opsProxy": 0.720},
            "awayAdvanced": {"opsProxy": 0.720},
        }
        self.assertAlmostEqual(_lineup_logit_adjustment(game, "mlb", enrichment), 0.0)

    def test_missing_data_on_both_sides_is_neutral(self) -> None:
        game = {"league": "mlb", "homeLineup": {}, "awayLineup": {}}
        self.assertEqual(_lineup_logit_adjustment(game, "mlb", {}), 0.0)

    def test_adjustment_stays_clamped_for_extreme_inputs(self) -> None:
        game = {"league": "nba", "homeLineup": {}, "awayLineup": {}}
        enrichment = {
            "homeAdvanced": {"pointsPerGame": 200.0},
            "awayAdvanced": {"pointsPerGame": 40.0},
        }
        self.assertLessEqual(_lineup_logit_adjustment(game, "nba", enrichment), 0.35)


class PublishableIdentityTests(unittest.TestCase):
    """Ranking must track object identity, not dict equality."""

    def test_identical_games_are_ranked_independently(self) -> None:
        def make_game() -> dict:
            return {
                "league": "mlb",
                "eventId": "same",
                "homeTeam": "Team A",
                "awayTeam": "Team B",
                "homeRecord": "60-40",
                "awayRecord": "40-60",
                "enrichment": {},
                "lines": [],
            }

        # Enrichment reaches the network; this test is about ranking only.
        with mock.patch.object(
            mlb_predictions, "enrich_games_with_providers", side_effect=lambda games, **_: games
        ):
            games = apply_predictions([make_game(), make_game()])

        # Two games that serialise identically must both keep their rank rather
        # than the second being demoted by an equality-based membership test.
        ranks = [game.get("predictionRank") for game in games]
        publishable = [(game.get("prediction") or {}).get("publishable") for game in games]
        self.assertEqual(len([rank for rank in ranks if rank is not None]), sum(1 for flag in publishable if flag))



class ScoringPaceTests(unittest.TestCase):
    """Pace must be per game, to compare against a combined total line."""

    def _form(self, scores):
        return {"homeLastFive": {"games": [{"score": s} for s in scores]}, "awayLastFive": {}}

    def test_pace_is_the_combined_score(self) -> None:
        from mlb_predictions import _scoring_pace_from_form

        # Two games totalling 8 and 6 -> 7.0, not 3.5 (the per-team average).
        self.assertAlmostEqual(_scoring_pace_from_form(self._form(["5-3", "4-2"])), 7.0)

    def test_over_is_reachable(self) -> None:
        """The over branch was dead code while pace ran 2x low."""
        from mlb_predictions import predict_total

        lines = [{"viewType": "Total", "currentLine": {"over": 8.5, "under": 8.5}}]
        hot = {"homeLastFive": {"games": [{"score": "9-7"}, {"score": "8-6"}]}, "awayLastFive": {}}
        self.assertEqual(predict_total({"league": "mlb"}, lines, hot)["pickSide"], "over")

    def test_under_still_reachable(self) -> None:
        from mlb_predictions import predict_total

        lines = [{"viewType": "Total", "currentLine": {"over": 8.5, "under": 8.5}}]
        cold = {"homeLastFive": {"games": [{"score": "2-1"}, {"score": "3-2"}]}, "awayLastFive": {}}
        self.assertEqual(predict_total({"league": "mlb"}, lines, cold)["pickSide"], "under")

    def test_malformed_scores_are_skipped(self) -> None:
        from mlb_predictions import _scoring_pace_from_form

        self.assertIsNone(_scoring_pace_from_form(self._form(["", "postponed", "7"])))


class SpreadModelTests(unittest.TestCase):
    def _spread(self, home_prob, league="nba", line="-6.5"):
        from mlb_predictions import predict_spread

        lines = [{"viewType": "Spread", "currentLine": {"home": line}}]
        prediction = {"probabilities": {"true": {"home": home_prob, "away": 1 - home_prob}}}
        return predict_spread({"league": league}, lines, prediction)

    def test_even_matchup_gives_a_pick_em_line(self) -> None:
        self.assertAlmostEqual(self._spread(0.5)["modelLine"], 0.0, places=1)

    def test_probability_actually_moves_the_line(self) -> None:
        """The old version always produced exactly 0.0 regardless of input."""
        self.assertLess(self._spread(0.8)["modelLine"], self._spread(0.6)["modelLine"])

    def test_line_stays_realistic_at_the_extremes(self) -> None:
        # A linear points-per-percent map produced -21 here; the correct
        # answer for an 80% NBA favourite is around -10.
        self.assertGreater(self._spread(0.8)["modelLine"], -12.0)
        self.assertLess(self._spread(0.8)["modelLine"], -7.0)

    def test_away_pick_uses_the_opposite_number(self) -> None:
        result = self._spread(0.30, line="-6.5")
        self.assertEqual(result["pickSide"], "away")
        self.assertIn("+6.5", result["pick"])

    def test_no_spread_for_fixed_line_leagues(self) -> None:
        """A fixed handicap has no line to solve for.

        Baseball's runline never moves off +/-1.5 and soccer uses Asian
        handicaps, so inverting a normal curve is the wrong tool for both. The
        measured margin distribution agrees: 29% of 503 graded MLB games were
        decided by exactly one run, a spike a normal curve cannot represent.
        Baseball is served by predict_runline instead.
        """
        self.assertIsNone(self._spread(0.7, league="mlb"))
        self.assertIsNone(self._spread(0.7, league="epl"))

    def test_cover_probability_comes_from_the_same_margin_model(self) -> None:
        """A spread with no probability cannot be priced, so it could never be
        ranked against the moneyline. It is derived rather than invented:
        modelLine is -inv_cdf(p_home)*sigma, so P(home covers L) is
        Phi((L - modelLine)/sigma) from that same normal.
        """
        # 0.7 against a -6.5 line lands inside the "No lean" band, which
        # legitimately has no probability -- use a gap that takes a side.
        result = self._spread(0.8)
        self.assertEqual(result["pickSide"], "home")
        self.assertIsNotNone(result["confidence"])
        self.assertGreater(result["confidence"], 50.0, "the picked side is the likelier one")
        # Still not a calibrated number, and nothing downstream may treat it as one.
        self.assertTrue(result["unvalidated"])

    def test_at_a_pick_em_line_it_agrees_exactly_with_the_moneyline(self) -> None:
        """The strongest available check on the derivation.

        At a line of 0 there is no handicap, so covering and winning are the
        same event -- the spread model and the moneyline model must return the
        same number. A sign error or a stray sigma would break this and nothing
        else would notice.
        """
        for home_prob in (0.62, 0.70, 0.85):
            result = predict_spread(
                {"league": "nba", "homeTeam": "H", "awayTeam": "A"},
                [{"viewType": "Spread", "currentLine": {"home": "0", "away": "0"}}],
                {"probabilities": {"true": {"home": home_prob, "away": 1 - home_prob}}},
            )
            self.assertEqual(result["pickSide"], "home")
            self.assertAlmostEqual(result["confidence"], round(home_prob * 100, 1), places=1)

    def test_a_favourite_covers_a_small_line_more_often_than_a_huge_one(self) -> None:
        def cover(line: float) -> float:
            r = predict_spread(
                {"league": "nba", "homeTeam": "H", "awayTeam": "A"},
                [{"viewType": "Spread", "currentLine": {"home": str(line), "away": str(-line)}}],
                {"probabilities": {"true": {"home": 0.85, "away": 0.15}}},
            )
            # Express both as "chance the home side covers" regardless of pick.
            return r["confidence"] if r["pickSide"] == "home" else 100.0 - r["confidence"]

        self.assertGreater(cover(-2.5), cover(-9.5))
        self.assertGreater(cover(-9.5), cover(-15.5))

    def test_a_push_pick_carries_no_probability(self) -> None:
        """"No lean" is not a position, so it has nothing to be confident about."""
        result = predict_spread(
            {"league": "nba", "homeTeam": "H", "awayTeam": "A"},
            [{"viewType": "Spread", "currentLine": {"home": "0", "away": "0"}}],
            {"probabilities": {"true": {"home": 0.5, "away": 0.5}}},
        )
        self.assertEqual(result["pickSide"], "push")
        self.assertIsNone(result["confidence"])

    def test_missing_probabilities_yield_no_pick(self) -> None:
        from mlb_predictions import predict_spread

        lines = [{"viewType": "Spread", "currentLine": {"home": "-6.5"}}]
        self.assertIsNone(predict_spread({"league": "nba"}, lines, {}))


class EnrichmentPreservationTests(unittest.TestCase):
    """apply_predictions must not clobber schedule-derived enrichment."""

    def test_already_enriched_games_are_not_re_enriched(self) -> None:
        import mlb_predictions

        game = {
            "league": "mlb",
            "eventId": "1",
            "homeTeam": "A",
            "awayTeam": "B",
            "homeRecord": "55-45",
            "awayRecord": "45-55",
            "enrichment": {
                "sources": ["ESPN standings"],
                "restDays": {"home": 1, "away": 3},
                "homeScheduleFlags": {"backToBack": True},
            },
        }
        with mock.patch.object(mlb_predictions, "enrich_games_with_providers") as enrich:
            mlb_predictions.apply_predictions([game])
        enrich.assert_not_called()
        self.assertEqual(game["enrichment"]["restDays"], {"home": 1, "away": 3})
        self.assertTrue(game["enrichment"]["homeScheduleFlags"]["backToBack"])

    def test_unenriched_games_are_enriched_once_per_league(self) -> None:
        import mlb_predictions

        games = [
            {"league": "mlb", "eventId": str(i), "homeTeam": "A", "awayTeam": "B", "enrichment": {}}
            for i in range(5)
        ]
        with mock.patch.object(
            mlb_predictions, "enrich_games_with_providers", side_effect=lambda g, **k: g
        ) as enrich:
            mlb_predictions.apply_predictions(games)
        # Batched: one call for the slate, not one per game.
        self.assertEqual(enrich.call_count, 1)
        self.assertEqual(len(enrich.call_args[0][0]), 5)

    def test_enrichment_failure_does_not_fail_predictions(self) -> None:
        import mlb_predictions

        game = {"league": "mlb", "eventId": "1", "homeTeam": "A", "awayTeam": "B", "enrichment": {}}
        with mock.patch.object(
            mlb_predictions, "enrich_games_with_providers", side_effect=RuntimeError("provider down")
        ):
            result = mlb_predictions.apply_predictions([game])
        self.assertIsNotNone(result[0].get("prediction"))


if __name__ == "__main__":
    unittest.main()
