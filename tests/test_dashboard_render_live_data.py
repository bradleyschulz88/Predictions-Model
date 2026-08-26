"""Render every view against the payloads actually committed, not a fixture.

`test_dashboard_render.py` builds its own fixture by hand. That is the right
call for geometry -- it pins the exact overflow-prone shape -- but it means the
page has never been rendered against the real `accuracy.json` or
`overview.json`, and those are precisely where this project's silent failures
have lived. Both instances of the `pricedPct` rename bug were consumers reading
a key the payload no longer had: nothing threw, the number just stopped
appearing, and a green build reported success. A hand-written fixture cannot
catch that, because it is written to match whatever the code currently expects.

So this file feeds the committed files in, walks all four views, and fails on
anything the page throws or on a panel that renders empty where the data says
it should not.

It is deliberately tolerant about content. Live data changes every half hour,
and a test that asserts today's numbers is a test that fails tomorrow for no
reason -- the same countdown that took 30 scheduled builds down on 2026-08-22.
What it asserts is structural: no exception, no unhandled rejection, and the
panels whose inputs are present are not blank.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_dashboard_render import (  # noqa: E402
    RENDER_TIMEOUT_SECONDS,
    STUB,
    _require_browser,
    find_chromium,
)

DATA = ROOT / "docs" / "data"

# Walks every view the nav offers. A view that throws on real data is the
# failure this file exists for, so each is entered and given time to settle.
PROBE = """
<script>
(async () => {
  const errors = [];
  addEventListener("error", (e) => errors.push(String(e.message || e.error)));
  addEventListener("unhandledrejection", (e) => errors.push(String(e.reason)));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const out = document.createElement("pre");
  out.id = "__probe";
  document.body.appendChild(out);
  const seen = {};

  try {
    for (let i = 0; i < 200 && !(typeof S !== "undefined" && S.overview); i++) {
      await sleep(20);
    }
    for (const view of ["overview", "sport", "accuracy", "about"]) {
      try {
        go(view);
      } catch (err) {
        errors.push(view + ": " + String(err && err.message ? err.message : err));
        continue;
      }
      await sleep(120);
      const main = document.querySelector("main") || document.body;
      seen[view] = {
        // Text length is a blunt instrument, but a view that renders nothing
        // at all is the failure mode a string assertion would miss.
        textLength: (main.textContent || "").trim().length,
        // Every "—" on a page whose payload has the number is a consumer
        // reading a key that moved.
        dashes: ((main.textContent || "").match(/—/g) || []).length,
      };
    }
  } catch (err) {
    errors.push("probe: " + String(err && err.message ? err.message : err));
  }

  out.textContent = "PROBE:" + JSON.stringify({ errors, seen });
})();
</script>
"""


def _fixture() -> dict:
    """Every committed payload, keyed by the filename the page fetches."""
    payload: dict = {}
    for source in sorted(DATA.glob("*.json")):
        try:
            payload[source.name] = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A malformed published file is its own bug and
            # test_publishable_json.py owns it; do not mask it here.
            continue
    return payload


def _render(target: Path) -> dict:
    for name in ("index.html", "board.js", "board.css", "app.js", "styles.css",
                 "manifest.json", "tools.html"):
        source = ROOT / "dashboard" / name
        if source.is_file():
            (target / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    index = target / "index.html"
    html = index.read_text(encoding="utf-8")
    stub = STUB % json.dumps(_fixture(), default=str)
    html = html.replace('<script src="board.js"></script>',
                        stub + '<script src="board.js"></script>')
    index.write_text(html.replace("</body>", PROBE + "</body>"), encoding="utf-8")

    argv = [find_chromium(), "--headless", "--no-sandbox", "--disable-gpu",
            "--hide-scrollbars", "--window-size=1200,1000",
            "--dump-dom", "--virtual-time-budget=8000", index.as_uri()]
    for attempt in (1, 2):
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=RENDER_TIMEOUT_SECONDS
            )
            break
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
    match = re.search(r"PROBE:(\{.*?\})</pre>", result.stdout, re.S)
    if not match:
        raise AssertionError(
            "the probe never reported; the page likely failed to boot on real data.\n"
            + result.stdout[-2000:] + "\n" + result.stderr[-2000:]
        )
    return json.loads(match.group(1))


@unittest.skipIf(
    find_chromium() is None and not _require_browser(),
    "no Chromium available to render with",
)
class LiveDataRenderTests(unittest.TestCase):
    probe: dict

    @classmethod
    def setUpClass(cls) -> None:
        if find_chromium() is None:
            raise AssertionError(
                "REQUIRE_RENDER_TESTS is set but no Chromium was found, so this "
                "would silently cover nothing."
            )
        if not (DATA / "accuracy.json").is_file():
            raise unittest.SkipTest("no committed accuracy.json to render against")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.probe = _render(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_no_view_throws_on_the_real_payloads(self) -> None:
        self.assertEqual(
            self.probe["errors"], [],
            "the page raised on data it is actually served in production",
        )

    def test_every_view_renders_something(self) -> None:
        """A view that throws inside a render function often catches nothing
        and simply leaves the panel empty."""
        for view, seen in (self.probe.get("seen") or {}).items():
            with self.subTest(view=view):
                self.assertGreater(
                    seen["textLength"], 200,
                    f"the {view} view rendered {seen['textLength']} characters",
                )

    def test_all_four_views_were_reached(self) -> None:
        self.assertEqual(
            sorted(self.probe.get("seen") or {}),
            ["about", "accuracy", "overview", "sport"],
        )

    def test_the_accuracy_view_is_not_mostly_placeholders(self) -> None:
        """The `pricedPct` rename broke two consumers, and the only visible
        symptom was a number replaced by a dash. Some dashes are legitimate --
        an unpriced league has no ROI -- so this catches a page of them, not
        the presence of any.
        """
        seen = (self.probe.get("seen") or {}).get("accuracy")
        if not seen:
            self.skipTest("accuracy view not reached")
        ratio = seen["dashes"] / max(1, seen["textLength"] / 100)
        self.assertLess(
            ratio, 3.0,
            f"the accuracy view shows {seen['dashes']} placeholders across "
            f"{seen['textLength']} characters, which suggests consumers reading "
            "keys the payload no longer has",
        )


if __name__ == "__main__":
    unittest.main()
