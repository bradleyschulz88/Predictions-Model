"""Render the dashboard in a real browser and check what static tests cannot.

Every layout bug this project has hit was found by a person looking at the
page, never by the suite:

  * the failure banner inserted as a sibling of `.wrap`, so it rendered 104px
    wider than every panel beneath it
  * the "Which bet" table crammed into the narrow context column, with the
    model and edge figures -- the two numbers the panel exists to show --
    pushed off the right edge
  * that table's PICK column right-aligning under a left-aligned heading,
    because board.css sets `th, td { text-align:right }`

String assertions against board.js structurally cannot catch any of those.
The markup was correct in every case; the geometry was not.

This drives headless Chromium through `--dump-dom`, which runs the page's
JavaScript and returns the resulting DOM. A probe script injected into the
page measures the geometry after boot and stamps the numbers back into the
DOM, where this test reads them. That keeps the project stdlib-only -- no
Playwright, no Selenium, no driver library -- while still asserting on real
layout rather than on source text.

Skips cleanly wherever no Chromium is present, so the suite still runs on a
machine without a browser. CI sets REQUIRE_RENDER_TESTS=1, which turns that
skip into a failure -- a silent skip there would mean layout regressions had
quietly stopped failing the build, which is the one thing this file exists to
prevent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DASHBOARD = ROOT / "dashboard"

# Playwright's bundled builds, then anything on PATH.
CHROMIUM_CANDIDATES = (
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "/opt/pw-browsers/chromium/chrome-linux/chrome",
)


def find_chromium() -> str | None:
    for candidate in CHROMIUM_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


# Injected BEFORE board.js. Serves the fixture straight from memory instead of
# over HTTP, because --virtual-time-budget fast-forwards timers while real
# network requests still take real time: the page would dump its DOM before
# any slate arrived. Stubbing fetch removes the race and the server both, so
# the test is hermetic and runs in a couple of seconds.
STUB = """
<script>
window.__FIXTURE__ = %s;
window.fetch = (input) => {
  const url = String(input && input.url ? input.url : input);
  const key = Object.keys(window.__FIXTURE__).find((name) => url.includes(name));
  if (!key) return Promise.resolve({ ok: false, status: 404, json: () => Promise.reject() });
  return Promise.resolve({ ok: true, status: 200,
    json: () => Promise.resolve(window.__FIXTURE__[key]) });
};
</script>
"""

# Injected after board.js. Waits for the board to boot, drives it to the state
# worth measuring, then reports geometry. Anything thrown on the way is
# captured rather than silently producing an empty page.
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

  const report = (extra) => {
    const d = document.documentElement;
    out.textContent = "PROBE:" + JSON.stringify({
      errors,
      // Positive means the page scrolls sideways, which it never should.
      bodyOverflowPx: Math.max(0, d.scrollWidth - d.clientWidth),
      ...extra,
    });
  };

  try {
    // Wait on the app's own state rather than a DOM proxy. boot() is async,
    // and go("sport") before S.overview lands leaves S.sport defaulted to mlb
    // and renders "could not load the slate" for a league with no fixture.
    for (let i = 0; i < 200 && !(typeof S !== "undefined" && S.overview); i++) {
      await sleep(20);
    }
    go("sport");
    for (let i = 0; i < 200; i++) {
      if (document.querySelector(".ghead")) break;
      await sleep(20);
    }
    const head = document.querySelector(".ghead");
    if (head) head.click();
    await sleep(120);

    // The markets table: every column must sit inside its scroll container,
    // or the edge figures are invisible without a sideways drag.
    let table = null;
    for (const h4 of document.querySelectorAll(".subh h4")) {
      if (/which bet/i.test(h4.textContent)) {
        table = h4.closest("div").parentElement.querySelector("table");
        break;
      }
    }
    const measured = {};
    if (table) {
      const box = table.parentElement;
      measured.tableClippedPx = Math.max(0, table.scrollWidth - box.clientWidth);
      const headers = [...table.querySelectorAll("th")].map((th) => th.textContent.trim());
      measured.headers = headers;
      // Header and its column's cells must share an alignment.
      const rows = [...table.querySelectorAll("tbody tr")];
      measured.alignmentMismatches = headers.map((name, i) => {
        const th = table.querySelectorAll("th")[i];
        const td = rows.length ? rows[0].children[i] : null;
        if (!td) return null;
        const a = getComputedStyle(th).textAlign, b = getComputedStyle(td).textAlign;
        return a === b ? null : `${name}: header ${a} vs cell ${b}`;
      }).filter(Boolean);
      // Is the rightmost column actually on screen?
      const lastCell = rows.length ? rows[0].children[headers.length - 1] : null;
      if (lastCell) {
        measured.lastColumnVisible =
          lastCell.getBoundingClientRect().right <= box.getBoundingClientRect().right + 1;
      }
    }
    measured.hasMarketsTable = !!table;
    report(measured);
  } catch (err) {
    errors.push("probe threw: " + err);
    report({});
  }
})();
</script>
"""


def build_fixture(target: Path) -> None:
    """A one-game slate with all three markets priced, so the panel renders."""
    from unittest.mock import patch

    import mlb_predictions

    for name in ("index.html", "board.css", "board.js"):
        shutil.copy(DASHBOARD / name, target / name)

    game = {
        "eventId": "g1", "league": "wnba", "matchup": "Away Club @ Home Club",
        "homeTeam": "Home Club", "awayTeam": "Away Club",
        "startDate": "2026-08-02T23:00:00Z",
        "homeRecord": "20-8", "awayRecord": "8-20",
        "homeHomeRecord": "12-2", "awayRoadRecord": "3-11",
        "enrichment": {},
        "lines": [
            {"viewType": "MoneyLine", "currentLine": {"home": "-450", "away": "+350"}},
            {"viewType": "Total", "currentLine": {"over": "o173.5 (-108)", "under": "u173.5 (-112)"}},
            {"viewType": "Spread", "currentLine": {"home": "-11.5 (-105)", "away": "+11.5 (-115)"}},
        ],
    }
    # A record that exercises both gate branches: totals validated, spread not.
    report = {"summary": {
        "totals": {"graded": 75, "decided": 73, "priced": 64, "pct": 61.6,
                   "stdErrPct": 5.7, "breakEvenPct": 52.3, "beatsBreakEven": False,
                   "pricedRoiPct": 10.5},
        "spreads": {"graded": 75, "decided": 75, "priced": 10, "pct": 58.7,
                    "stdErrPct": 5.7, "breakEvenPct": 52.9, "beatsBreakEven": False,
                    "pricedRoiPct": 32.5},
    }}
    with patch.object(mlb_predictions, "_get_accuracy_report", return_value=report):
        mlb_predictions.apply_predictions([game])

    prediction = game["prediction"]
    headline = (prediction.get("bestBet") or {}).get("pick") or {}
    slate = {"league": "wnba", "games": [game]}
    play = {
        "league": "wnba", "eventId": "g1", "matchup": game["matchup"],
        "pick": prediction["predictedWinner"], "pickSide": "home",
        "confidence": prediction["confidence"], "marketPct": 81.8,
        "evPct": headline.get("evPct"), "kellyPct": headline.get("kellyPct"),
        "odds": headline.get("odds"), "breakEvenPct": headline.get("breakEvenPct"),
        "bestBet": prediction.get("bestBet"), "betMarket": headline.get("market"),
        "betLabel": headline.get("pick"), "startDate": game["startDate"],
    }
    fixture = {
        "wnba_2026-08-02.json": slate,
        "wnba.json": slate,
        "overview.json": {
            "builtAt": "2026-08-02T18:00:00Z",
            "summary": {"picks": 1, "priced": 1, "positiveEv": 1,
                        "bestEvPct": headline.get("evPct")},
            "leagues": [{"id": "wnba", "label": "WNBA Basketball", "gameCount": 1,
                         "pickCount": 1, "pricedCount": 1, "scheduleDate": "2026-08-02",
                         "best": {"pick": prediction["predictedWinner"],
                                  "matchup": game["matchup"],
                                  "confidence": prediction["confidence"],
                                  "evPct": headline.get("evPct"),
                                  "odds": headline.get("odds")}}],
            "worthBacking": [play], "passedOn": [], "unpriced": [],
        },
        "manifest.json": {"leagues": [{
            "id": "wnba", "defaultDate": "2026-08-02", "availableDates": ["2026-08-02"],
            "dateFiles": {"2026-08-02": "data/wnba_2026-08-02.json"},
        }]},
        "accuracy.json": {
            "updatedAt": "2026-08-02T18:00:00Z",
            "summary": {"allTime": {"total": 774, "correct": 472, "pct": 61.0,
                                    "units": 16.5, "roiPct": 2.1},
                        "byLeague": {"wnba": {"total": 71, "pct": 60.0}},
                        **report["summary"]},
            "picksByEventId": {},
        },
    }
    for name in ("evaluation.json", "model_weights.json"):
        source = ROOT / "docs" / "data" / name
        if source.is_file():
            fixture[name] = json.loads(source.read_text(encoding="utf-8"))

    # The stub must be defined before board.js runs; the probe after it, so the
    # globals it drives exist.
    index = target / "index.html"
    html = index.read_text(encoding="utf-8")
    stub = STUB % json.dumps(fixture, default=str)
    html = html.replace('<script src="board.js"></script>',
                        stub + '<script src="board.js"></script>')
    index.write_text(html.replace("</body>", PROBE + "</body>"), encoding="utf-8")


def _require_browser() -> bool:
    """Whether a missing browser should fail rather than skip.

    Skipping keeps the suite runnable on a machine with no Chromium, which is
    the right default. It is the wrong default in CI: a silent skip there
    means layout regressions stop failing the build, which is the one thing
    this file exists to prevent. The workflow sets this, so a runner that
    loses its browser breaks loudly instead of quietly covering nothing.
    """
    return os.environ.get("REQUIRE_RENDER_TESTS", "").strip().lower() in {"1", "true", "yes"}


@unittest.skipIf(
    find_chromium() is None and not _require_browser(),
    "no Chromium available to render with",
)
class DashboardRenderTests(unittest.TestCase):
    """Geometry, not markup. These are the checks a string match cannot make."""

    probe: dict

    @classmethod
    def setUpClass(cls) -> None:
        if find_chromium() is None:
            raise AssertionError(
                "REQUIRE_RENDER_TESTS is set but no Chromium was found, so the "
                "layout checks would silently cover nothing. Install a browser "
                "or unset the variable."
            )
        cls._tmp = tempfile.TemporaryDirectory()
        target = Path(cls._tmp.name)
        build_fixture(target)
        result = subprocess.run(
            [find_chromium(), "--headless", "--no-sandbox", "--disable-gpu",
             "--hide-scrollbars", "--window-size=1200,1000",
             "--dump-dom", "--virtual-time-budget=8000",
             (target / "index.html").as_uri()],
            capture_output=True, text=True, timeout=180,
        )
        match = re.search(r"PROBE:(\{.*?\})</pre>", result.stdout, re.S)
        if not match:
            raise AssertionError(
                "the probe never reported; the page likely failed to boot.\n"
                + result.stdout[-2000:] + "\n" + result.stderr[-2000:]
            )
        cls.probe = json.loads(match.group(1))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_page_raises_no_errors(self) -> None:
        self.assertEqual(self.probe["errors"], [])

    def test_the_page_never_scrolls_sideways(self) -> None:
        """Wide content belongs in its own scroll container, not the body."""
        self.assertEqual(
            self.probe["bodyOverflowPx"], 0,
            f"the page scrolls horizontally by {self.probe['bodyOverflowPx']}px",
        )

    def test_the_markets_table_actually_rendered(self) -> None:
        self.assertTrue(self.probe["hasMarketsTable"],
                        "the Which bet panel did not render, so the rest proves nothing")

    def test_no_column_is_clipped_out_of_view(self) -> None:
        """The bug this file exists for: five columns in the narrow context
        sidebar pushed MODEL and EDGE off the right edge."""
        self.assertEqual(
            self.probe["tableClippedPx"], 0,
            f"the markets table overflows its container by "
            f"{self.probe['tableClippedPx']}px, hiding its rightmost columns",
        )

    def test_the_edge_column_is_on_screen(self) -> None:
        self.assertTrue(self.probe.get("lastColumnVisible"),
                        "the rightmost column sits outside its container")

    def test_every_header_matches_its_column_alignment(self) -> None:
        """board.css sets `th, td { text-align:right }` with only :first-child
        left, so a column styled on the header alone silently disagrees with
        its own values."""
        self.assertEqual(self.probe["alignmentMismatches"], [])

    def test_the_expected_columns_are_present(self) -> None:
        self.assertEqual(
            [h.upper() for h in self.probe["headers"]],
            ["MARKET", "PICK", "PRICE", "MODEL", "EDGE"],
        )


if __name__ == "__main__":
    unittest.main()
