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


if __name__ == "__main__":
    unittest.main()
