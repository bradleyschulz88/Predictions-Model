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


if __name__ == "__main__":
    unittest.main()
