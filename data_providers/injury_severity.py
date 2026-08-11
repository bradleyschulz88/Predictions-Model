"""Score how much an injury list actually costs a team.

The existing weighting (``_injury_role_weight`` in mlb_predictions) counts each
injury as roughly 1.0 and looks for the substring "pitcher" or "quarterback" in
text that does not contain a position at all -- ESPN's records carry only
player, status, detail and return date. Every absence therefore weighed about
the same, which is a plausible reason the injury feature measured a correlation
of -0.02 with outcomes: not that injuries do not matter, but that counting them
this way carries no information.

Two scorers live here:

* A deterministic one, always available, that reads availability from the
  status field and seriousness from the injury description.
* An optional LLM scorer that adds the one thing the feed cannot supply -- how
  important the player is to the team. It is off unless NVIDIA_API_KEY is set,
  cached per player, and falls back to the deterministic score on any failure.

Neither is trusted yet. The feature is wired through to the ablation so the
data can decide, which it cannot do until enough games carry it.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from typing import Any, Iterable

from mlb_cache import PROVIDER_CACHE

# How unavailable the player is, from the status field.
STATUS_AVAILABILITY = {
    "season": 1.0,
    "60-day": 1.0,
    "ir": 1.0,
    "suspended": 1.0,
    "out": 0.95,
    "45-day": 0.95,
    "15-day": 0.85,
    "10-day": 0.80,
    "7-day": 0.75,
    "doubtful": 0.65,
    "questionable": 0.40,
    "day-to-day": 0.25,
    "probable": 0.15,
}
DEFAULT_AVAILABILITY = 0.5

# How serious the described injury is, as a multiplier on availability.
INJURY_SEVERITY = (
    (("tommy john", "acl", "rupture", "torn", "tear", "surgery"), 1.30),
    (("fracture", "broken", "break", "dislocat"), 1.15),
    (("strain", "sprain", "pull"), 1.00),
    (("soreness", "tendinitis", "inflammation", "contusion", "bruise", "spasm"), 0.85),
    (("illness", "flu", "personal", "paternity", "bereavement"), 0.60),
)
DEFAULT_SEVERITY = 1.0

# Player importance, when nothing better is known. The LLM scorer replaces this.
DEFAULT_IMPORTANCE = 1.0

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# A small instruct model is the right tool: the task is "rate this player 0-3",
# not reasoning. A 550B reasoning model would be far slower, burn the free tier
# in a fraction of the calls, and answer no better.
NVIDIA_MODEL = os.environ.get("NVIDIA_INJURY_MODEL", "meta/llama-3.1-8b-instruct")
LLM_TIMEOUT_SECONDS = 20

# The free tier allows ~40 requests/minute across the whole key. A full build
# scores ~80 teams, so unthrottled it would trip the limit immediately and every
# team after the first 40 would silently fall back to the deterministic score.
# 1.6s between calls keeps it just under, at a cost of roughly two minutes.
MIN_SECONDS_BETWEEN_CALLS = 1.6

# Why the last call failed, for the build report. Every failure path used to
# return a bare None, so a rejected key, an exhausted quota, a blocked network
# and a malformed reply were indistinguishable -- the build could only say
# "absent, rejected or out of quota" and leave you to guess which. Rotating the
# key and seeing the same message is exactly the situation that needs a status
# code, so the reason is recorded here and reported once per run.
_last_failure: str | None = None


def last_failure() -> str | None:
    """Why the LLM scorer last failed, or None if it never did."""
    return _last_failure


def reset_failure() -> None:
    """Clear the recorded reason. For tests, and for a fresh build."""
    global _last_failure
    _last_failure = None


def _note_failure(reason: str) -> None:
    global _last_failure
    _last_failure = reason

# Hard ceiling per process, so an unusually large slate cannot stall a build.
MAX_CALLS_PER_RUN = 200

_last_call_at = 0.0
_calls_made = 0


def reset_llm_budget() -> None:
    """Test hook: clear the throttle and per-run call budget."""
    global _last_call_at, _calls_made
    _last_call_at, _calls_made = 0.0, 0


def _availability(status: str | None) -> float:
    text = (status or "").lower()
    for token, value in STATUS_AVAILABILITY.items():
        if token in text:
            return value
    return DEFAULT_AVAILABILITY


def _severity(detail: str | None) -> float:
    text = (detail or "").lower()
    for tokens, value in INJURY_SEVERITY:
        if any(token in text for token in tokens):
            return value
    return DEFAULT_SEVERITY


def _days_until_return(return_date: str | None, today: date | None = None) -> int | None:
    if not return_date:
        return None
    try:
        parsed = datetime.fromisoformat(str(return_date)[:10]).date()
    except (TypeError, ValueError):
        return None
    return (parsed - (today or date.today())).days


def _return_multiplier(return_date: str | None, today: date | None = None) -> float:
    """A player due back tomorrow costs less than one out for two months."""
    days = _days_until_return(return_date, today)
    if days is None:
        return 1.0
    if days <= 0:
        return 0.5
    if days <= 7:
        return 0.8
    if days <= 30:
        return 1.0
    return 1.2


def deterministic_injury_score(
    injury: dict[str, Any], *, today: date | None = None
) -> float:
    """Cost of one absence in [0, ~1.6], before any player-importance weighting."""
    return round(
        _availability(injury.get("status"))
        * _severity(injury.get("detail"))
        * _return_multiplier(injury.get("returnDate"), today),
        4,
    )


# --------------------------------------------------------------------------
# Optional LLM importance scoring
# --------------------------------------------------------------------------


def api_key() -> str | None:
    """The key with surrounding whitespace removed, or None if unusable.

    Stripping is not cosmetic. Pasting a key into the GitHub secrets field
    carries a trailing newline more often than not, and urllib rejects a header
    value containing one outright -- `ValueError: Invalid header value`. The
    request never leaves the machine, so the failure looks like an unreachable
    API or a rejected key when the key is perfectly good. A production build
    proved it: 0 of 34 teams scored, with a masked value ending in a backslash.

    A key with interior whitespace is a truncated or concatenated paste rather
    than a stray newline, so it is refused with a distinct reason instead of
    being silently mangled into something the API will reject.
    """
    raw = os.environ.get("NVIDIA_API_KEY")
    if raw is None:
        return None
    key = raw.strip()
    if not key:
        _note_failure("NVIDIA_API_KEY is set but contains only whitespace")
        return None
    if any(character.isspace() for character in key):
        _note_failure(
            "NVIDIA_API_KEY contains a space or line break inside it, so it is a "
            "partial or doubled paste -- copy the key again as a single line"
        )
        return None
    if _looks_like_an_unexpanded_variable(key):
        _note_failure(
            f"NVIDIA_API_KEY is set to the literal text {key!r}, not a key. NVIDIA's "
            "sample code writes api_key=\"$NVIDIA_API_KEY\" as a placeholder, and "
            "Python does not expand $NAME inside quotes -- paste the nvapi- key itself"
        )
        return None
    return key


# `$NVIDIA_API_KEY`, `${NVIDIA_API_KEY}`, `%NVIDIA_API_KEY%`: a variable reference
# that nothing ever substituted.
_UNEXPANDED = re.compile(r"^(\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|%[A-Za-z_][A-Za-z0-9_]*%)$")


def _looks_like_an_unexpanded_variable(key: str) -> bool:
    """Whether the "key" is really the name of where the key should have come from.

    NVIDIA's quick-start snippet reads `api_key = "$NVIDIA_API_KEY"`, which is a
    placeholder in shell clothing -- Python sends those fifteen characters
    verbatim. Copying that line into a secret is an easy mistake and produces a
    plain 401, indistinguishable from a revoked key, so the advice it earns
    ("rotate the key") is the one thing that cannot help.

    Deliberately narrow: an entire value that is nothing but a variable
    reference. A real key that merely contains a dollar sign is untouched.
    """
    return bool(_UNEXPANDED.match(key))


def llm_enabled() -> bool:
    return api_key() is not None


_PROMPT = (
    "You rate how much a team is hurt by a player's absence. "
    "For each player below, reply with how important they are to their team on a "
    "0-3 scale: 0 = fringe roster, 1 = regular contributor, 2 = key starter, "
    "3 = franchise player. Reply ONLY with a JSON object mapping each player "
    "name to its number. No prose.\n\n"
    "League: {league}\nTeam: {team}\nPlayers:\n{players}"
)


def _throttle() -> None:
    """Space calls so a build stays under the free tier's rate limit."""
    global _last_call_at
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def _call_nvidia(prompt: str, key: str) -> str | None:
    global _calls_made
    if _calls_made >= MAX_CALLS_PER_RUN:
        _note_failure(f"hit the {MAX_CALLS_PER_RUN}-call budget for this run")
        return None

    payload = json.dumps(
        {
            "model": NVIDIA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            # Deterministic output: the same slate must not score differently
            # between two builds half an hour apart.
            "temperature": 0.0,
            "max_tokens": 400,
        }
    ).encode("utf-8")

    for attempt in range(2):
        _throttle()
        _calls_made += 1
        request = urllib.request.Request(
            NVIDIA_BASE_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            # Rate limiting is worth one backoff; anything else is not
            # retryable and falls straight through to the deterministic score.
            if error.code == 429 and attempt == 0:
                time.sleep(4.0)
                continue
            _note_failure(_describe_http_error(error))
            return None
        except (urllib.error.URLError, ValueError, TimeoutError, OSError) as error:
            _note_failure(f"could not reach the API ({type(error).__name__}: {error})")
            return None

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            _note_failure(f"the reply had no message content (keys: {sorted(body)})")
            return None

    _note_failure("rate limited twice in a row (429)")
    return None


def _describe_http_error(error: urllib.error.HTTPError) -> str:
    """Turn a status code into the sentence that names the actual fix.

    401 and 403 are the two that matter here and they mean different things: a
    401 is a key the API does not accept at all, which a rotation fixes; a 403
    is a key it accepts but will not let near this model, which a rotation does
    not fix. Reporting them as one message sends you back to regenerate a key
    that was never the problem.
    """
    detail = ""
    try:
        body = error.read().decode("utf-8", errors="replace")[:200].strip()
        detail = f" -- {body}" if body else ""
    except Exception:  # noqa: BLE001 - the status code is the point, not the body
        detail = ""

    if error.code == 401:
        return (
            f"HTTP 401, the key was rejected: regenerate it at build.nvidia.com and "
            f"update the NVIDIA_API_KEY secret{detail}"
        )
    if error.code == 403:
        return (
            f"HTTP 403, the key is valid but not entitled to {NVIDIA_MODEL}: this is "
            f"a model-access problem, NOT a stale key, so rotating again will not fix "
            f"it{detail}"
        )
    if error.code == 429:
        return f"HTTP 429, out of free-tier quota -- it resets, so try later{detail}"
    if error.code == 404:
        return (
            f"HTTP 404, no such model as {NVIDIA_MODEL}: NVIDIA retired or renamed it, "
            f"set NVIDIA_INJURY_MODEL to a current one{detail}"
        )
    return f"HTTP {error.code} {error.reason}{detail}"


def _parse_importance(text: str | None) -> dict[str, float]:
    """Pull the JSON object out of a reply that may be wrapped in prose or fences."""
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}

    scores: dict[str, float] = {}
    for name, value in raw.items():
        try:
            rating = float(value)
        except (TypeError, ValueError):
            continue
        # Map the 0-3 rating onto a multiplier centred near 1.
        scores[str(name).strip().lower()] = round(0.5 + min(3.0, max(0.0, rating)) * 0.5, 3)
    return scores


def player_importance(
    injuries: Iterable[dict[str, Any]],
    *,
    league: str,
    team: str | None,
) -> dict[str, float]:
    """Importance multiplier per player name, or {} when unavailable.

    Cached per team so a slate costs one call per team rather than one per
    injury, which keeps a full multi-league build inside the free tier's
    40 requests per minute.
    """
    key = api_key()
    names = [str(injury.get("player") or "").strip() for injury in injuries]
    names = [name for name in names if name]
    if not key or not names:
        return {}

    cache_key = f"injury-importance:{league}:{team}:{','.join(sorted(names))}"
    cached = PROVIDER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    prompt = _PROMPT.format(
        league=league.upper(),
        team=team or "Unknown",
        players="\n".join(f"- {name}" for name in names),
    )
    scores = _parse_importance(_call_nvidia(prompt, key))
    PROVIDER_CACHE.set(cache_key, scores)
    return scores


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def team_injury_severity(
    injuries: list[dict[str, Any]] | None,
    *,
    league: str,
    team: str | None = None,
    use_llm: bool | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Total injury cost for one team, with the per-player breakdown."""
    injuries = injuries or []
    if not injuries:
        return {"score": 0.0, "count": 0, "source": "none", "players": []}

    should_use_llm = llm_enabled() if use_llm is None else use_llm
    importance = (
        player_importance(injuries, league=league, team=team) if should_use_llm else {}
    )

    players: list[dict[str, Any]] = []
    total = 0.0
    for injury in injuries:
        base = deterministic_injury_score(injury, today=today)
        name = str(injury.get("player") or "").strip().lower()
        weight = importance.get(name, DEFAULT_IMPORTANCE)
        cost = round(base * weight, 4)
        total += cost
        players.append(
            {
                "player": injury.get("player"),
                "base": base,
                "importance": weight,
                "cost": cost,
            }
        )

    return {
        "score": round(total, 4),
        "count": len(injuries),
        "source": "llm" if importance else "deterministic",
        "players": players,
    }
