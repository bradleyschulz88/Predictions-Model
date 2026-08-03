"""Every shipped dashboard file must reach the published site.

The copy step in pages.yml used to enumerate five filenames. Adding a sixth
asset -- which the redesign did, twice -- produced a green build that deployed a
page referencing a stylesheet and a script that were never copied, so the site
rendered unstyled with no console error to explain it. Nothing caught that,
because the workflow succeeded and the tests never looked at the workflow.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "dashboard"
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"


def _prepare_step() -> str:
    """The body of the 'Prepare docs site' step."""
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("Prepare docs site")
    # Ends at the next step in the job.
    end = text.index("- name:", start + 10)
    return text[start:end]


class DashboardAssetTests(unittest.TestCase):
    def test_every_dashboard_file_is_copied_to_docs(self) -> None:
        step = _prepare_step()
        copied = set(re.findall(r"dashboard/([A-Za-z0-9._-]+)", step))
        shipped = {p.name for p in DASHBOARD.iterdir() if p.is_file()}
        missing = shipped - copied
        self.assertFalse(
            missing,
            f"these dashboard files are never copied into docs/, so the deployed "
            f"site will 404 on them: {sorted(missing)}",
        )

    def test_the_copy_step_does_not_reference_a_file_that_is_gone(self) -> None:
        step = _prepare_step()
        copied = set(re.findall(r"dashboard/([A-Za-z0-9._-]+)", step))
        shipped = {p.name for p in DASHBOARD.iterdir() if p.is_file()}
        stale = copied - shipped
        self.assertFalse(stale, f"the workflow copies files that no longer exist: {sorted(stale)}")


class EntryPointTests(unittest.TestCase):
    """The front page must load the assets it actually needs."""

    def setUp(self) -> None:
        self.index = (DASHBOARD / "index.html").read_text(encoding="utf-8")

    def test_index_loads_the_new_shell(self) -> None:
        self.assertIn('href="board.css"', self.index)
        self.assertIn('src="board.js"', self.index)

    def test_index_does_not_load_the_legacy_bundle(self) -> None:
        """styles.css and app.js belong to tools.html now, not the front page."""
        self.assertNotRegex(self.index, r'href="\./?styles\.css"')
        self.assertNotRegex(self.index, r'src="\./?app\.js"')

    def test_the_four_views_exist(self) -> None:
        for view in ("v-board", "v-sport", "v-accuracy", "v-dig"):
            self.assertIn(f'id="{view}"', self.index)

    def test_the_legacy_tools_page_is_reachable(self) -> None:
        """The bet tracker was not deleted, so it must be linked."""
        self.assertIn('href="tools.html"', self.index)
        tools = (DASHBOARD / "tools.html").read_text(encoding="utf-8")
        self.assertRegex(tools, r'src="\./?app\.js"')
        self.assertIn('href="index.html"', tools)

    def test_the_service_worker_caches_the_new_assets(self) -> None:
        sw = (DASHBOARD / "sw.js").read_text(encoding="utf-8")
        for asset in ("./board.css", "./board.js", "./tools.html"):
            self.assertIn(asset, sw)

    def test_the_publish_bar_matches_the_python_side(self) -> None:
        """A dashboard bar that drifts from calibration_params shows picks the
        record does not contain, or hides ones it does."""
        from calibration_params import MIN_PICK_CONFIDENCE

        source = (DASHBOARD / "board.js").read_text(encoding="utf-8")
        match = re.search(r"MIN_PUBLISHABLE_CONFIDENCE\s*=\s*([0-9.]+)", source)
        self.assertIsNotNone(match, "board.js must state its publish bar")
        self.assertEqual(float(match.group(1)), float(MIN_PICK_CONFIDENCE))


class DateNavigationTests(unittest.TestCase):
    """The redesign shipped without any way to move off today's slate --

    the model publishes three days out per league, and there was no prev/next,
    no date picker, nothing. The Sports view could only ever load
    `L.scheduleDate` from overview.json, which is always today. These pin the
    pieces that fix it, statically, so the capability cannot quietly drop out
    of a future edit the way it dropped out of the first one.
    """

    def setUp(self) -> None:
        self.js = (DASHBOARD / "board.js").read_text(encoding="utf-8")
        self.html = (DASHBOARD / "index.html").read_text(encoding="utf-8")

    def test_the_date_bar_markup_exists(self) -> None:
        for control in ("datePrev", "dateNext", "dateChips"):
            self.assertIn(f'id="{control}"', self.html)

    def test_the_slate_loader_reads_manifest_date_files(self) -> None:
        """Without this, a date change has no path to a different day's JSON."""
        self.assertIn("manifestLeague(S.sport)?.dateFiles?.[date]", self.js)

    def test_slates_are_cached_per_date_not_per_league(self) -> None:
        """The pre-fix cache was S.slates[league] -- one slot per league, so a
        second date could never coexist with the first and every navigation
        re-fetched, or worse, silently returned the wrong day from cache."""
        self.assertIn("S.slates[cacheKey]", self.js)
        self.assertNotIn("S.slates[S.sport]", self.js)

    def test_next_and_prev_are_wired_to_a_handler(self) -> None:
        self.assertIn('$("#datePrev").addEventListener', self.js)
        self.assertIn('$("#dateNext").addEventListener', self.js)

    def test_readings_are_recomputed_for_the_loaded_slate(self) -> None:
        """The pre-fix Published/Priced counts came from the overview's
        per-league summary, which describes only today -- so browsing to
        tomorrow kept showing today's counts beside tomorrow's games."""
        self.assertIn("publishedCount", self.js)
        self.assertIn("pricedCount", self.js)
        self.assertNotIn("L.pickCount ?? ", self.js)


class FailureBannerTests(unittest.TestCase):
    """A second audit pass on the date-nav fix found the failure banner had
    two bugs of its own, both caught by actually rendering it rather than by
    reading the code: manifest.json was fetched with a bare
    `catch(() => null)`, so if it ever failed in production the date bar
    would vanish with zero indication -- the exact silent-degradation this
    session exists to fix, reintroduced one function away. And the banner
    itself, once added, was inserted as a sibling of `.wrap` rather than a
    child, so it rendered 104px wider than every panel beneath it and flush
    against the first one with no gap.
    """

    def setUp(self) -> None:
        self.js = (DASHBOARD / "board.js").read_text(encoding="utf-8")

    def test_a_missing_manifest_is_reported_not_swallowed(self) -> None:
        manifest_fetch = self.js.index('getJson("data/manifest.json")')
        clause = self.js[manifest_fetch : manifest_fetch + 400]
        self.assertIn("S.failures.push", clause,
            "manifest.json must not be the one fetch with a bare catch(() => null) "
            "-- losing it silently drops date navigation to today-only")

    def test_the_banner_is_inserted_inside_wrap_not_before_it(self) -> None:
        """A sibling of .wrap sits outside its padding and max-width."""
        self.assertIn('$("#boardBody").prepend(b)', self.js)
        self.assertNotIn('$("#boardBody").before(b)', self.js)

    def test_the_banner_is_inserted_after_renderboard_clears_the_host(self) -> None:
        """renderBoard() sets #boardBody's innerHTML = "" at its own top. A
        banner prepended into #boardBody before that call runs is erased by
        it a moment later -- the insertion must come after."""
        render_call = self.js.index("renderBoard();")
        banner_insert = self.js.index('$("#boardBody").prepend(b)')
        self.assertLess(render_call, banner_insert,
            "the banner is prepended before renderBoard() runs, so its own "
            "innerHTML clear will wipe the banner out immediately")


class AblationRecheckWiringTests(unittest.TestCase):
    """The queued-candidate ablation (h2h, handedness, bullpen, elo, ...) used
    to mean someone remembering to run `model_fit.py --ablate` by hand and
    reading a terminal table. It has to actually be rebuilt on a schedule and
    read on the dashboard, or "visible on a schedule" is just a comment."""

    def setUp(self) -> None:
        self.js = (DASHBOARD / "board.js").read_text(encoding="utf-8")
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_the_build_reruns_the_ablation_every_time(self) -> None:
        self.assertIn("model_fit.py --ablate --write", self.workflow)

    def test_the_recheck_step_runs_after_the_refit_that_feeds_it(self) -> None:
        refit = self.workflow.index("Refit model and gate on quality")
        recheck = self.workflow.index("model_fit.py --ablate --write")
        self.assertLess(refit, recheck)

    def test_the_dashboard_fetches_the_ablation_report(self) -> None:
        self.assertIn('getJson("data/ablation.json")', self.js)
        self.assertIn("S.ablation = ablation", self.js)

    def test_the_dig_view_renders_it(self) -> None:
        render_call = self.js.index("function renderDig()")
        body = self.js[render_call:]
        self.assertIn("ablationPanel(S.ablation)", body)

    def test_a_missing_report_does_not_crash_the_panel(self) -> None:
        """A build from before this shipped, or one with too little graded
        data, must degrade to a message rather than throwing on rows[-1]."""
        panel = self.js[self.js.index("function ablationPanel("):self.js.index("function renderDig()")]
        self.assertIn("ablation?.rows", panel)
        self.assertIn("if (!rows.length)", panel)


class BorrowedCalibrationTests(unittest.TestCase):
    """Every calibration reading on the board comes from one pooled reliability
    curve built across all leagues at once. That is the right way to build it --
    no single league has enough graded games for its own curve -- but it means a
    band reading "n=139" can be almost entirely another sport.

    NBA and NFL are the live case: both open a season with zero graded games
    here, so until they build a record every calibration number shown against
    one of their picks is borrowed from baseball. The board has to say so, in
    every place it presents one, or it is asserting a record that does not
    exist.
    """

    def setUp(self) -> None:
        self.js = (DASHBOARD / "board.js").read_text(encoding="utf-8")

    def test_the_helper_reads_the_leagues_own_graded_count(self) -> None:
        self.assertIn("function leagueGraded(league)", self.js)
        self.assertIn("summary?.byLeague", self.js)

    def test_zero_history_and_thin_history_read_differently(self) -> None:
        """'None yet' and 'not many yet' are different facts."""
        self.assertIn("function borrowedCalibrationNote(league)", self.js)
        self.assertIn("borrowed entirely from other leagues", self.js)
        self.assertIn("mostly other leagues", self.js)

    def test_every_surface_that_shows_a_calibration_number_is_covered(self) -> None:
        """The four places a band reaches the reader. Missing one puts an
        unlabelled borrowed number on the page, which is the whole bug."""
        # 1. the per-game calibration check panel
        self.assertIn("const borrowed = borrowedCalibrationNote(play.league)", self.js)
        # 2. the hero's calibration tile, which must know whose record it is
        self.assertIn("function calibrationShort(confidence, league)", self.js)
        self.assertIn("calibrationShort(play.confidence, play.league)", self.js)
        # 3. the Kelly stake note, which now sizes off that band
        self.assertIn("that band is borrowed from other leagues", self.js)
        # 4. the Dig checklist
        self.assertIn('out.push(["unproven league", borrowed])', self.js)

    def test_the_league_tile_flags_it_before_you_open_a_game(self) -> None:
        self.assertIn("No graded record yet", self.js)
        self.assertIn("leagueGraded(L.id) === 0", self.js)

    def test_the_threshold_is_named_not_inlined(self) -> None:
        match = re.search(r"MIN_LEAGUE_HISTORY\s*=\s*([0-9]+)", self.js)
        self.assertIsNotNone(match, "board.js must state its own history bar")
        self.assertEqual(int(match.group(1)), 30)

    def test_a_league_with_its_own_record_gets_no_warning(self) -> None:
        """Guards the guard: the note must be conditional, not always-on.

        An unconditional warning is the same failure as no warning -- it stops
        carrying information and trains the reader to skip it.
        """
        self.assertIn("if (graded >= MIN_LEAGUE_HISTORY) return null;", self.js)


if __name__ == "__main__":
    unittest.main()


class WhichBetPanelTests(unittest.TestCase):
    """The card named one bet and printed the other two markets as inert text
    with no pick, price or edge, so "is the moneyline even the best bet here"
    had no answer on the page. The panel that replaces it has to rank all
    priced markets, mark the winner, and be honest about the gate."""

    def setUp(self) -> None:
        self.js = (DASHBOARD / "board.js").read_text(encoding="utf-8")

    def test_the_panel_exists_and_replaces_the_inert_box(self) -> None:
        self.assertIn("function marketsPanel(play)", self.js)
        self.assertNotIn("hit rate only — no price logged for these", self.js,
                         "the old inert side-markets box should be gone")

    def test_it_renders_full_width_not_in_the_context_sidebar(self) -> None:
        """A five-column table in the narrow right column pushed the model and
        edge figures off screen -- the two numbers the panel exists to show."""
        body_append = self.js.index("body.appendChild(why);")
        tail = self.js[body_append:body_append + 500]
        self.assertIn("body.appendChild(markets)", tail)

    def test_the_backed_market_is_marked(self) -> None:
        self.assertIn("Back this", self.js)

    def test_an_unvalidated_market_is_marked_but_not_hidden(self) -> None:
        """Hiding a real edge would be its own dishonesty."""
        self.assertIn("unproven", self.js)
        self.assertIn("o.validated", self.js)

    def test_a_held_back_market_is_explained(self) -> None:
        self.assertIn("best.heldBack", self.js)
        self.assertIn("priced graded pick", self.js)

    def test_the_headline_is_the_bet_not_the_winner(self) -> None:
        """Once the board ranks on best market, printing the moneyline team
        above a total's price attributes one market's edge to another."""
        self.assertIn("function betHeadline(play)", self.js)
        self.assertIn("betHeadline(play)", self.js)

    def test_the_play_carries_the_best_bet_through(self) -> None:
        self.assertIn("bestBet: prediction.bestBet", self.js)
        self.assertIn("betMarket:", self.js)

    def test_the_pick_column_states_its_alignment(self) -> None:
        """board.css sets `th, td { text-align:right }` with only :first-child
        left, so without this the values sit right-aligned under a left-aligned
        heading."""
        self.assertIn('padding:7px 9px;text-align:left', self.js)

    def test_no_priced_market_still_names_what_picks_exist(self) -> None:
        self.assertIn("no price reached the build", self.js)


class PlayedGameResultTests(unittest.TestCase):
    """A finished game said "Final" and stopped.

    accuracy.json has carried the final score, whether the moneyline was
    correct, and the graded outcome of the totals and spread picks since side
    markets were scored -- keyed by the same eventId the board already had --
    and board.js had zero references to it. The result is the most useful thing
    about a played game and it was the one thing not shown.
    """

    def setUp(self) -> None:
        self.js = (DASHBOARD / "board.js").read_text(encoding="utf-8")

    def test_the_board_reads_the_graded_record(self) -> None:
        self.assertIn("function resultFor(eventId)", self.js)
        self.assertIn("picksByEventId", self.js)

    def test_only_graded_rows_are_treated_as_results(self) -> None:
        """A pending row has no score and no outcome to report."""
        self.assertIn('row.status === "graded"', self.js)

    def test_scores_are_coerced_rather_than_trusted(self) -> None:
        """The scoreboard feed returns them as strings."""
        self.assertIn("function finalScore(result)", self.js)
        self.assertIn("Number.isFinite(home)", self.js)

    def test_a_push_is_neither_a_win_nor_a_loss(self) -> None:
        """Counting a returned stake as a win would flatter the record."""
        self.assertIn('push: "Push"', self.js)

    def test_the_verdict_follows_the_bet_that_was_recommended(self) -> None:
        """A card headlining a total must not report won or lost against the
        moneyline -- that is a different market than the one it advised."""
        self.assertIn('marketOutcome(result, play.betMarket || "moneyline")', self.js)

    def test_each_market_is_settled_separately(self) -> None:
        self.assertIn("marketOutcome(result, o.market)", self.js)

    def test_the_result_column_only_exists_once_there_is_one(self) -> None:
        """An unplayed game must not show an empty column."""
        self.assertIn('result ? `<th style="text-align:right;padding:0 9px 5px">RESULT</th>` : ""', self.js)
