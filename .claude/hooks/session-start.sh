#!/bin/bash
# Prepare a Claude Code on the web session for this repo.
#
# The container is reclaimed after inactivity and the repo is cloned fresh, so
# anything installed or started by hand is gone next session. This puts back
# the two things that do not survive: the certifi bundle the build relies on,
# and the OmniRoute daemon.
#
# Nothing here may abort the session. A missing optional tool is a normal
# state, not a failure, so every optional step is guarded rather than trusted.
set -euo pipefail

# Local checkouts have their own environments; only the web sessions are
# starting from nothing.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# ---------------------------------------------------------------------------
# Python dependencies
#
# The runtime is stdlib-only by design -- requirements.txt carries certifi and
# nothing else, and a test enforces that no third-party package reaches the
# prediction path. This mirrors the workflow's install step so tests behave
# the same here as in CI.
# ---------------------------------------------------------------------------
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
  # --root-user-action=ignore because these containers run as root and the
  # venv advice would otherwise print on every single session start.
  PIP_FLAGS=(--quiet --disable-pip-version-check --root-user-action=ignore)
  if pip install "${PIP_FLAGS[@]}" -r "$PROJECT_DIR/requirements.txt" 2>/dev/null \
     || pip install "${PIP_FLAGS[@]}" --break-system-packages \
        -r "$PROJECT_DIR/requirements.txt" 2>/dev/null; then
    echo "session-start: python dependencies ready"
  else
    echo "session-start: could not install requirements.txt; continuing"
  fi
fi

# Point Python at certifi's bundle, exactly as .github/workflows/pages.yml
# does. Without it HTTPS to the data providers is less reliable.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  CERT_PATH="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  if [ -n "$CERT_PATH" ]; then
    echo "export SSL_CERT_FILE=$CERT_PATH" >> "$CLAUDE_ENV_FILE"
  fi
fi

# ---------------------------------------------------------------------------
# OmniRoute
#
# Installed globally rather than by this repo, so it may simply be absent --
# that is fine and must not fail the session. When it is present the daemon
# still dies with the container, so start it if nothing is listening.
# ---------------------------------------------------------------------------
OMNIROUTE_PORT=20128
if command -v omniroute >/dev/null 2>&1; then
  # Ask the port, not the process table. `pgrep -f omniroute` matches any
  # command line containing the word -- including the shell that invokes this
  # hook -- so it reported "already running" while the server was down, and
  # would have skipped starting it every time.
  if (exec 3<>"/dev/tcp/127.0.0.1/$OMNIROUTE_PORT") 2>/dev/null; then
    exec 3>&- 3<&-
    echo "session-start: omniroute already listening on $OMNIROUTE_PORT"
  else
    # --no-open and --no-tray because there is no desktop here. Failure is
    # non-fatal: the repo does not depend on it.
    if omniroute serve --daemon --no-open --no-tray >/dev/null 2>&1; then
      echo "session-start: omniroute started on http://localhost:$OMNIROUTE_PORT"
    else
      echo "session-start: omniroute present but would not start; continuing"
    fi
  fi
else
  echo "session-start: omniroute not installed; skipping"
fi

# ---------------------------------------------------------------------------
# graphify
#
# A CLI plus a /graphify skill that maps this repo into a queryable knowledge
# graph. Installed with `uv tool install`, so it lives in the container and
# dies with it -- the same problem this file exists to solve.
#
# Measured against an empty uv cache in this container: 7 seconds and ~175MB
# of cache for around thirty tree-sitter wheels. Cheap enough to do on every
# session start rather than leaving it to be re-run by hand.
#
# --native-tls with the agent proxy's CA bundle is required: uv does not read
# the certifi path exported above, and without it every PyPI fetch fails with
# "invalid peer certificate: UnknownIssuer".
# ---------------------------------------------------------------------------
if command -v graphify >/dev/null 2>&1; then
  echo "session-start: graphify already installed ($(graphify --version 2>/dev/null || echo present))"
elif command -v uv >/dev/null 2>&1; then
  GRAPHIFY_CA="/root/.ccr/ca-bundle.crt"
  [ -f "$GRAPHIFY_CA" ] || GRAPHIFY_CA="${SSL_CERT_FILE:-}"
  # The published package is `graphifyy`; the command it installs is
  # `graphify`. Not a typo -- checked against the project's own pyproject.
  if SSL_CERT_FILE="$GRAPHIFY_CA" uv tool install --native-tls graphifyy >/dev/null 2>&1; then
    # Registers the skill and writes ~/.claude/CLAUDE.md. Also container-local.
    if PATH="$HOME/.local/bin:$PATH" graphify install --platform claude >/dev/null 2>&1; then
      echo "session-start: graphify installed and /graphify skill registered"
    else
      echo "session-start: graphify installed but the skill would not register; continuing"
    fi
  else
    echo "session-start: graphify would not install; continuing"
  fi
else
  echo "session-start: uv not available, skipping graphify"
fi

echo "session-start: ready"
