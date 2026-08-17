"""The cache name is the eviction trigger, and it was being forgotten.

`dashboard/sw.js` opened with "Bump on any app-shell change", because
`activate` deletes every cache whose name is not the current one. That comment
was the entire mechanism, and it failed the way comments-as-mechanisms do: the
constant sat at `edge-board-v53` from 2026-07-30 while 23 commits changed
`board.js` and `app.js` beneath it. The eviction path had not run in three
weeks.

Network-first fetching hid it. An online browser revalidates every shell asset
and every data file, so fresh content arrived regardless and nothing looked
wrong. It stops being hidden the moment the network does not answer, because
the handler then falls back to `caches.match` -- and what comes back is
whatever that browser cached, asset by asset, with no guarantee the pieces came
from one build. A July `board.js` reading an August `accuracy.json` is a
console error with no visible cause.

That is not hypothetical this week: GitHub Pages was down for over two hours on
2026-08-17 and failed three builds, which is exactly the window where every
request falls back to cache.

Fixed by deriving the name from the files instead of remembering it. These
tests hold the derivation honest: it covers everything the worker pre-caches,
it changes when any of them changes, and the build actually runs it.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.stamp_service_worker import SHELL_FILES, shell_digest, stamp

DASHBOARD = ROOT / "dashboard"
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"


def _staged() -> Path:
    """A throwaway copy of the shipped dashboard, as the build assembles it."""
    directory = Path(tempfile.mkdtemp())
    for name in (*SHELL_FILES, "sw.js"):
        shutil.copy(DASHBOARD / name, directory / name)
    return directory


class CoverageTests(unittest.TestCase):
    """A file pre-cached but not hashed reintroduces the same drift."""

    def test_everything_the_worker_pre_caches_is_hashed(self) -> None:
        source = (DASHBOARD / "sw.js").read_text(encoding="utf-8")
        assets = source[source.index("const ASSETS = ["):]
        assets = assets[: assets.index("];")]
        listed = {
            line.strip().strip(",").strip('"').lstrip("./")
            for line in assets.split("\n")
            if '"./' in line
        }
        # "./" is the directory itself, served as index.html, not a file.
        listed.discard("")
        self.assertTrue(listed, "could not read the ASSETS list")
        missing = listed - set(SHELL_FILES)
        self.assertEqual(missing, set(), f"pre-cached but not hashed: {sorted(missing)}")

    def test_every_hashed_file_actually_ships(self) -> None:
        for name in SHELL_FILES:
            self.assertTrue((DASHBOARD / name).is_file(), f"{name} is hashed but not shipped")


class DigestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = _staged()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_the_same_shell_gives_the_same_name(self) -> None:
        self.assertEqual(shell_digest(self.directory), shell_digest(self.directory))

    def test_changing_any_shipped_file_changes_the_name(self) -> None:
        before = shell_digest(self.directory)
        for name in SHELL_FILES:
            with self.subTest(changed=name):
                path = self.directory / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n/* touched */\n")
                self.assertNotEqual(before, shell_digest(self.directory), name)
                path.write_bytes(original)
        self.assertEqual(before, shell_digest(self.directory), "restore failed")

    def test_a_missing_file_is_an_error_not_a_silent_hash(self) -> None:
        """Hashing whatever happens to be present would produce a plausible
        name for a broken deploy."""
        (self.directory / "board.js").unlink()
        with self.assertRaises(FileNotFoundError):
            shell_digest(self.directory)

    def test_swapping_two_files_is_a_different_build(self) -> None:
        """Content alone would collide; the name is hashed in too."""
        before = shell_digest(self.directory)
        a, b = self.directory / "board.css", self.directory / "styles.css"
        a_body, b_body = a.read_bytes(), b.read_bytes()
        a.write_bytes(b_body)
        b.write_bytes(a_body)
        self.assertNotEqual(before, shell_digest(self.directory))


class StampTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = _staged()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_it_writes_the_derived_name_into_the_file(self) -> None:
        name = stamp(self.directory)
        self.assertIn(f'const CACHE = "{name}";', (self.directory / "sw.js").read_text(encoding="utf-8"))

    def test_the_name_is_recognisable_as_this_app(self) -> None:
        self.assertTrue(stamp(self.directory).startswith("edge-board-"))

    def test_stamping_twice_is_stable(self) -> None:
        """The worker's own text is not hashed, so a re-run cannot chase itself."""
        self.assertEqual(stamp(self.directory), stamp(self.directory))

    def test_it_leaves_the_rest_of_the_worker_alone(self) -> None:
        before = (self.directory / "sw.js").read_text(encoding="utf-8")
        stamp(self.directory)
        after = (self.directory / "sw.js").read_text(encoding="utf-8")
        self.assertEqual(
            len(before.split("\n")), len(after.split("\n")), "the stamp changed more than one line"
        )
        self.assertIn("self.addEventListener(\"fetch\"", after)

    def test_a_worker_with_no_cache_line_is_an_error(self) -> None:
        (self.directory / "sw.js").write_text("// nothing here\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            stamp(self.directory)


class WorkflowTests(unittest.TestCase):
    """A deriver the build never calls is the same bug with extra files."""

    def setUp(self) -> None:
        self.source = WORKFLOW.read_text(encoding="utf-8")

    def test_the_build_stamps_the_worker(self) -> None:
        self.assertIn("scripts/stamp_service_worker.py", self.source)

    def test_it_runs_after_the_worker_is_copied(self) -> None:
        self.assertLess(
            self.source.index("dashboard/sw.js docs/"),
            self.source.index("scripts/stamp_service_worker.py"),
            "the stamp would rewrite a file that is about to be overwritten",
        )

    def test_it_stamps_the_directory_that_is_published(self) -> None:
        published = self.source[self.source.index("upload-pages-artifact"):]
        published = published[: published.index("- name:")]
        self.assertIn("path: docs", published)
        self.assertIn("stamp_service_worker.py docs", self.source)

    def test_the_checked_in_value_is_only_a_local_fallback(self) -> None:
        """If it looked like a real version someone would bump it again."""
        source = (DASHBOARD / "sw.js").read_text(encoding="utf-8")
        self.assertIn('const CACHE = "edge-board-dev";', source)
        self.assertIn("build time", source)


if __name__ == "__main__":
    unittest.main()
