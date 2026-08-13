"""Score how much an injury list actually costs a team.

The existing weighting (``_injury_role_weight`` in mlb_predictions) counts each
injury as roughly 1.0 and looks for the substring "pitcher" or "quarterback" in
text that does not contain a position at all -- ESPN's records carry only
player, status, detail and return date. Every absence therefore weighed about
the same, which is a plausible reason the injury feature measured a correlation
of -0.02 with outcomes: not that injuries do not matter, but that counting them
this way carries no information.

One scorer lives here: a deterministic one that reads availability from the
status field and seriousness from the injury description.

There used to be a second, an optional LLM pass over NVIDIA's API that added
the one thing the feed cannot supply -- how important the player is to the
team. It is gone, along with the key it needed. Two reasons, and the first is
the weaker one: the key was never successfully configured, so in practice every
score this project ever published came from the deterministic path anyway.

The real reason is the ablation. `injuryDiff` and `injurySeverityDiff` both
made walk-forward log loss worse, not better, at every sample size measured --
0.6438 to 0.6459 as they went in. Keeping a metered external dependency, a
rate limiter, a per-team cache and several hundred lines of key-handling to
feed a feature the data has repeatedly declined is not a trade worth making.

The deterministic score stays because it costs nothing, is already wired into
enrichment, and is what the board has always actually been showing.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

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
# Public entry point
# --------------------------------------------------------------------------


def team_injury_severity(
    injuries: list[dict[str, Any]] | None,
    *,
    league: str | None = None,
    team: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Total injury cost for one team, with the per-player breakdown.

    `league` and `team` are kept because callers pass them and because a future
    scorer would need them; nothing reads them now.
    """
    injuries = injuries or []
    if not injuries:
        return {"score": 0.0, "count": 0, "source": "none", "players": []}

    players: list[dict[str, Any]] = []
    total = 0.0
    for injury in injuries:
        cost = deterministic_injury_score(injury, today=today)
        total += cost
        players.append(
            {
                "player": injury.get("player"),
                "base": cost,
                "importance": DEFAULT_IMPORTANCE,
                "cost": cost,
            }
        )

    return {
        "score": round(total, 4),
        "count": len(injuries),
        "source": "deterministic",
        "players": players,
    }
