#!/usr/bin/env python3
"""Name the service worker's cache after the app shell it was built from.

`dashboard/sw.js` opens with "Bump on any app-shell change", because `activate`
deletes every cache whose name is not the current one -- so the version string
IS the eviction trigger. A manual step that must be remembered on every change
is a step that eventually is not, and this one was not: the constant sat at
`edge-board-v53` from 2026-07-30 while 23 commits changed `board.js` and
`app.js` beneath it. The eviction path had not run in three weeks.

The fetch handler is network-first, so an online browser still got fresh files
and the drift stayed invisible. It stops being invisible the moment the network
does not answer -- a dropped connection, or a Pages outage like the one on
2026-08-17 that failed three builds over two hours. Then every request falls
back to `caches.match`, and what comes back is whatever that browser happened
to cache, asset by asset, with no guarantee the pieces came from the same
build. A `board.js` from July reading an `accuracy.json` from August is a
console error with no obvious cause.

So the name is derived rather than remembered. The hash covers exactly the
files the worker pre-caches, which means it changes when and only when a
shipped asset changes, and it cannot be forgotten.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# The files whose content defines a build of the shell. Kept in step with the
# ASSETS list inside sw.js by test_service_worker_cache.py rather than by
# eye -- a file pre-cached but not hashed would reintroduce the same drift on
# a smaller scale.
SHELL_FILES = (
    "index.html",
    "board.css",
    "board.js",
    "tools.html",
    "app.js",
    "styles.css",
    "manifest.json",
)

CACHE_PATTERN = re.compile(r'(const CACHE = ")([^"]+)(";)')


def shell_digest(directory: Path, files: tuple[str, ...] = SHELL_FILES) -> str:
    """Short, stable hash over the shipped shell.

    Filenames are hashed alongside contents so that swapping two files' bodies
    is a different build, and the list is sorted so the result does not depend
    on directory order.
    """
    digest = hashlib.sha256()
    for name in sorted(files):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"app-shell file missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def stamp(directory: Path) -> str:
    """Rewrite sw.js in `directory` with a content-derived cache name."""
    worker = directory / "sw.js"
    source = worker.read_text(encoding="utf-8")
    if not CACHE_PATTERN.search(source):
        raise ValueError(f"no `const CACHE = \"...\";` line in {worker}")

    name = f"edge-board-{shell_digest(directory)}"
    worker.write_text(CACHE_PATTERN.sub(rf"\g<1>{name}\g<3>", source), encoding="utf-8")
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default="docs",
        type=Path,
        help="the assembled site directory (default: docs)",
    )
    args = parser.parse_args(argv)
    print(f"Service worker cache: {stamp(args.directory)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
