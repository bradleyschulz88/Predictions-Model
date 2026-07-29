"""Tests for accuracy tracking and prediction enhancements."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from accuracy_tracker import grade_predictions, record_predictions
from mlb_predictions import confidence_label, extract_total_line, predict_game
from sports_config import list_league_ids


class PredictionEnhancementTests(unittest.TestCase):
    def test_confidence_labels(self) -> None:
        self.assertEqual(confidence_label(70), "Strong pick")
        self.assertEqual(confidence_label(58), "Lean")
        self.assertEqual(confidence_label(52), "Coin flip")

    def test_extract_total_line(self) -> None:
        lines = [{"viewType": "Total", "currentLine": {"over": "o8.5 (-110)", "under": "u8.5 (-110)"}}]
        self.assertEqual(extract_total_line(lines), 8.5)

    def test_predict_game_includes_model_fields(self) -> None:
        game = {
            "league": "mlb",
            "homeTeam": "Home",
            "awayTeam": "Away",
            "homeRecord": "30-20",
            "awayRecord": "20-30",
            "enrichment": {"homeMajorInjuries": [], "awayMajorInjuries": [{"player": "X", "status": "Out"}]},
        }
        prediction = predict_game(game)
        self.assertIn("confidenceLabel", prediction)
        self.assertIn("dataSources", prediction)
        self.assertIn("features", prediction)
        self.assertIn("pick", prediction["probabilities"])
        self.assertIn("implied", prediction["probabilities"])
        self.assertIn("dataCoverage", prediction["features"])


class AccuracyTrackerTests(unittest.TestCase):
    def test_record_and_grade_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            record_predictions(data_dir, {"mlb": {"scheduleDate": "2026-06-16", "fetchedAt": "now", "games": []}})
            accuracy = grade_predictions(data_dir)
            self.assertIn("summary", accuracy)
            self.assertEqual(accuracy.get("skippedDates"), [])
            self.assertTrue((data_dir / "predictions_log.json").is_file())

    def test_record_predictions_stores_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            payload = {
                "league": "mlb",
                "scheduleDate": "2026-06-16",
                "fetchedAt": "now",
                "games": [
                    {
                        "eventId": "1",
                        "matchup": "A @ B",
                        "prediction": {
                            "predictedWinner": "B",
                            "predictedSide": "home",
                            "outcomeLabel": "B to win",
                            # Above MLB's 65 bar. At 62 this pick is now
                            # correctly withheld, which is the point of the
                            # per-league threshold.
                            "confidence": 71.0,
                            "features": {"recordDiff": 0.1, "league": "mlb"},
                        },
                    }
                ],
            }
            record_predictions(data_dir, [payload])
            log = json.loads((data_dir / "predictions_log.json").read_text(encoding="utf-8"))
            self.assertEqual(log["predictions"]["1"]["features"]["recordDiff"], 0.1)


class LeagueConfigTests(unittest.TestCase):
    def test_includes_new_leagues(self) -> None:
        leagues = set(list_league_ids())
        self.assertTrue({"mlb", "nfl", "nba", "wnba", "epl", "afl"}.issubset(leagues))
        # Retired after the 2026 tournament; the graded history is kept, the fetch is not.
        self.assertNotIn("worldcup", leagues)


class BuildPagesTests(unittest.TestCase):
    def test_build_overview_sorts_top_picks(self) -> None:
        from scripts.build_pages_data import build_overview

        payloads = {
            "mlb": {
                "leagueLabel": "MLB",
                "scheduleDate": "2026-06-16",
                "gameCount": 2,
                "topPick": "Yankees ML",
                "games": [
                    {"matchup": "A @ B", "eventId": "1", "prediction": None},
                    {"matchup": "C @ D", "eventId": "2", "prediction": {"outcomeLabel": "C ML", "confidence": 71, "confidenceLabel": "Strong pick", "predictedWinner": "C"}},
                ],
            },
            "nba": {
                "leagueLabel": "NBA",
                "scheduleDate": "2026-06-16",
                "gameCount": 1,
                "topPick": "Lakers ML",
                "games": [
                    {"matchup": "E @ F", "eventId": "3", "prediction": {"outcomeLabel": "F ML", "confidence": 68, "confidenceLabel": "Lean", "predictedWinner": "F"}},
                ],
            },
        }
        overview = build_overview(payloads)
        self.assertEqual(len(overview["leagues"]), 2)
        self.assertGreaterEqual(overview["topPicksOverall"][0]["confidence"], overview["topPicksOverall"][1]["confidence"])

    def test_build_overview_ranks_by_ev_not_confidence(self) -> None:
        """The whole point of the landing page: price decides, not confidence.

        The MLB pick is more confident but priced at what the model already
        thinks, so it must rank below the less confident NBA pick that is
        priced generously.
        """
        from scripts.build_pages_data import build_overview

        def game(event_id, winner, side, confidence, odds, market_pct):
            prediction = {
                "predictedWinner": winner,
                "predictedSide": side,
                "outcomeLabel": f"{winner} to win",
                "confidence": confidence,
                "confidenceLabel": "Strong pick",
                "homeWinPct": confidence if side == "home" else 100 - confidence,
                "awayWinPct": confidence if side == "away" else 100 - confidence,
            }
            if odds is not None:
                prediction["value"] = {"evPct": None, "odds": odds, "kellyPct": 1.0}
                # evPct is what ranks; assert on the value the builder copies up.
                prediction["value"]["evPct"] = {"-500": -3.0, "200": 12.0}[str(odds)]
            if market_pct is not None:
                prediction["probabilities"] = {"implied": {"consensus": {f"{side}Pct": market_pct}}}
            return {"matchup": f"X @ {winner}", "eventId": event_id, "prediction": prediction}

        overview = build_overview(
            {
                "mlb": {"leagueLabel": "MLB", "gameCount": 1, "games": [game("1", "Yankees", "home", 88, -500, 84.0)]},
                "nba": {"leagueLabel": "NBA", "gameCount": 1, "games": [game("2", "Celtics", "away", 61, 200, 33.0)]},
                "afl": {"leagueLabel": "AFL", "gameCount": 1, "games": [game("3", "Blues", "home", 77, None, None)]},
            }
        )

        self.assertEqual([play["pick"] for play in overview["worthBacking"]], ["Celtics"])
        self.assertEqual([play["pick"] for play in overview["passedOn"]], ["Yankees"])
        self.assertEqual([play["pick"] for play in overview["unpriced"]], ["Blues"])

        # A pick with no price is never counted as priced, however confident.
        summary = overview["summary"]
        self.assertEqual(summary["picks"], 3)
        self.assertEqual(summary["priced"], 2)
        self.assertEqual(summary["positiveEv"], 1)
        self.assertEqual(summary["unpriced"], 1)
        self.assertEqual(summary["bestEvPct"], 12.0)

        # Each league leads with its best-priced play; unpriced leagues fall
        # back to confidence rather than showing nothing.
        best = {league["id"]: league["best"] for league in overview["leagues"]}
        self.assertEqual(best["nba"]["evPct"], 12.0)
        self.assertEqual(best["mlb"]["evPct"], -3.0)
        self.assertIsNone(best["afl"]["evPct"])
        self.assertEqual(best["afl"]["confidence"], 77)

        # The market side the pick is on, not the home side by default.
        self.assertEqual(best["nba"]["marketPct"], 33.0)

    def test_build_overview_survives_empty_input(self) -> None:
        from scripts.build_pages_data import build_overview

        overview = build_overview({})
        self.assertEqual(overview["worthBacking"], [])
        self.assertIsNone(overview["summary"]["bestEvPct"])
        self.assertEqual(overview["summary"]["suggestedUnits"], 0.0)

    def test_include_enrichment_for_all_dates(self) -> None:
        from scripts.build_pages_data import include_enrichment_for_date

        self.assertTrue(include_enrichment_for_date("2026-06-16", "2026-06-16"))
        self.assertTrue(include_enrichment_for_date("2026-06-13", "2026-06-16"))
        self.assertTrue(include_enrichment_for_date("2026-06-20", "2026-06-16"))

    def test_build_league_payload_resilient_retries_on_ssl(self) -> None:
        from unittest.mock import patch

        from scripts.build_pages_data import build_league_payload_resilient

        calls: list[bool] = []

        def fake_build(*_args, verify_ssl=True, **_kwargs):
            calls.append(verify_ssl)
            if verify_ssl:
                raise RuntimeError("SSL: CERTIFICATE_VERIFY_FAILED")
            return {"gameCount": 1, "games": [{"eventId": "1", "prediction": {"outcomeLabel": "Test"}}]}

        with patch("scripts.build_pages_data.build_league_payload", side_effect=fake_build):
            payload = build_league_payload_resilient(
                "mlb",
                "2026-06-21",
                include_enrichment=True,
                include_odds=False,
            )

        self.assertEqual(calls, [True, False])
        self.assertEqual(payload["gameCount"], 1)


if __name__ == "__main__":
    unittest.main()


class OddsCoverageWarningTests(unittest.TestCase):
    """A configured odds source that prices nothing is broken, not unpriced.

    merge_sbr_odds_into_games swallows SBR errors on purpose, so a missing board
    cannot destroy the schedule. The cost is that a permanently broken slug or
    team-name match fails silently -- WNBA logged 115 picks without a single
    price while carrying a valid sbr_odds_slug, and nothing said so.
    """

    def _summary(self, games: int, priced: int) -> dict:
        return {
            "gameCount": games,
            "counts": {"espnPredictor": games, "impliedOdds": priced},
            "pct": {"espnPredictor": 100.0 if games else 0.0},
        }

    def test_priced_league_with_no_prices_warns(self) -> None:
        from data_coverage import coverage_warnings

        warnings = coverage_warnings({"wnba": self._summary(5, 0)})
        self.assertTrue(any("odds source configured" in w for w in warnings))

    def test_unpriced_league_stays_silent(self) -> None:
        """AFL has no odds slug; zero prices is the correct, expected state."""
        from data_coverage import coverage_warnings

        warnings = coverage_warnings({"afl": self._summary(4, 0)})
        self.assertFalse(any("odds source configured" in w for w in warnings))

    def test_working_league_stays_silent(self) -> None:
        from data_coverage import coverage_warnings

        warnings = coverage_warnings({"mlb": self._summary(15, 14)})
        self.assertFalse(any("odds source configured" in w for w in warnings))

    def test_empty_slate_stays_silent(self) -> None:
        from data_coverage import coverage_warnings

        warnings = coverage_warnings({"nba": self._summary(0, 0)})
        self.assertFalse(any("odds source configured" in w for w in warnings))

    def test_retired_league_carries_no_expectation(self) -> None:
        from data_coverage import coverage_warnings

        warnings = coverage_warnings({"worldcup": self._summary(3, 0)})
        self.assertFalse(any("odds source configured" in w for w in warnings))


class EmptyBoardWarningTests(unittest.TestCase):
    """A publish bar that swallows the slate is a broken bar, not a quiet day.

    MLB carried a 65 threshold measured against a confidence distribution that
    then shifted 9.6 points down underneath it. The bar went from excluding the
    bottom quartile to withholding 90% of games -- 100% on some days -- and
    nothing said a word, because every build stayed green. An empty board is not
    an error, so it needed to be made into a warning.
    """

    def _summary(self, games: int, published: int) -> dict:
        return {
            "gameCount": games,
            "published": published,
            "counts": {"espnPredictor": games, "impliedOdds": games},
            "pct": {"espnPredictor": 100.0 if games else 0.0},
        }

    def _fired(self, summary: dict, league: str = "mlb") -> bool:
        from data_coverage import coverage_warnings

        return any("publish bar" in w for w in coverage_warnings({league: summary}))

    def test_a_swallowed_slate_warns(self) -> None:
        """The exact regression: 2 of 16 published."""
        self.assertTrue(self._fired(self._summary(16, 2)))

    def test_a_completely_empty_board_warns(self) -> None:
        self.assertTrue(self._fired(self._summary(15, 0)))

    def test_a_healthy_board_stays_silent(self) -> None:
        self.assertFalse(self._fired(self._summary(16, 9)))

    def test_a_small_slate_is_not_judged(self) -> None:
        """1 of 3 is 33% and means nothing; small slates swing wildly."""
        self.assertFalse(self._fired(self._summary(3, 0)))

    def test_absent_published_count_is_not_an_error(self) -> None:
        """Older snapshots carry no `published` key; that is not a failure."""
        summary = self._summary(16, 0)
        summary.pop("published")
        self.assertFalse(self._fired(summary))

    def test_summarize_coverage_reports_what_reached_the_board(self) -> None:
        from data_coverage import summarize_coverage

        games = [
            {"league": "mlb", "prediction": {"predictedWinner": "A", "confidence": 71.0}},
            {"league": "mlb", "prediction": {"predictedWinner": "B", "confidence": 60.0}},
            {"league": "mlb", "prediction": {"predictedWinner": "C", "confidence": 51.0}},
        ]
        summary = summarize_coverage(games)
        self.assertEqual(summary["gameCount"], 3)
        # 71 and 60 clear the global floor; 51 does not.
        self.assertEqual(summary["published"], 2)


class PredictorCoverageExpectationTests(unittest.TestCase):
    """Warn where a feed is broken, not where it never existed.

    AFL logged a 0% ESPN-predictor coverage warning on every build. Measured
    across the whole history: MLB 183 of 623 games carried a predictor and WNBA
    30 of 120, while AFL managed 0 of 55 and the World Cup 0 of 74. ESPN
    publishes it for the US major leagues and not for Australian football or
    soccer, so the AFL warning was permanent noise -- worse than silence,
    because it trains you to skip the annotations that matter.
    """

    def _summary(self, games: int, predictor: int) -> dict:
        return {
            "gameCount": games,
            "published": games,
            "counts": {"espnPredictor": predictor, "impliedOdds": games},
            "pct": {"espnPredictor": round(predictor / games * 100, 1) if games else 0.0},
        }

    def _fired(self, league: str, games: int, predictor: int) -> bool:
        from data_coverage import coverage_warnings

        return any(
            "predictor coverage" in w
            for w in coverage_warnings({league: self._summary(games, predictor)})
        )

    def test_afl_no_longer_warns(self) -> None:
        """The exact noise: one game, no predictor, every single build."""
        self.assertFalse(self._fired("afl", 1, 0))
        self.assertFalse(self._fired("afl", 9, 0))

    def test_soccer_no_longer_warns(self) -> None:
        self.assertFalse(self._fired("epl", 10, 0))

    def test_a_genuinely_broken_us_feed_still_warns(self) -> None:
        """MLB does carry a predictor, so zero coverage there is a real break."""
        self.assertTrue(self._fired("mlb", 15, 0))
        self.assertTrue(self._fired("wnba", 12, 0))

    def test_healthy_coverage_stays_silent(self) -> None:
        self.assertFalse(self._fired("mlb", 15, 15))

    def test_a_retired_league_carries_no_expectation(self) -> None:
        self.assertFalse(self._fired("worldcup", 8, 0))

    def test_the_flag_matches_what_was_measured(self) -> None:
        from sports_config import get_league

        for league in ("mlb", "nfl", "nba", "wnba"):
            self.assertTrue(get_league(league).supports_espn_predictor, league)
        for league in ("epl", "afl"):
            self.assertFalse(get_league(league).supports_espn_predictor, league)
