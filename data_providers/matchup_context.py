"""Plate umpire and pitcher handedness.

Umpire
------
Plate umpires have persistent, measurable strike-zone tendencies worth roughly a
third of a run per game between the extremes. It is a totals input far more than
a winner input -- a tight zone suppresses scoring for both sides equally, the
same argument that puts park factor in the totals model.

The assignment is published on the day, via the schedule's `officials`
hydration. Before that it is simply unknown, and this returns None rather than
guessing.

Handedness
----------
The platoon split is the oldest real edge in baseball: hitters do better against
opposite-handed pitching. What this provides is the *fact* of the matchup -- who
is throwing with which arm -- which is the input a split needs.

It deliberately stops short of applying a platoon adjustment. Doing that
properly needs each lineup's split against left- and right-handed pitching, and
lineups are confirmed late; a league-average platoon constant applied to every
game would add noise while looking like analysis.

Neither of these moves a published probability. Both are logged as ablation
candidates and ship only if they beat their own absence out of sample.
"""

from __future__ import annotations

from typing import Any

from data_providers.utils import fetch_json

# Tendencies are published as runs per game above or below average. The spread
# between the most and least permissive umpires is roughly a third of a run,
# which is real but small -- comparable to a mild park, not to Coors.
UMPIRE_RUNS_SPREAD = 0.35


def _fetch(url: str, *, cache_key: str, verify_ssl: bool = True) -> dict:
    return fetch_json(url, cache_key=cache_key, verify_ssl=verify_ssl)


def fetch_plate_umpire(game_pk: int | str | None, *, verify_ssl: bool = True) -> str | None:
    """Name of the home-plate umpire, or None if not yet assigned.

    Never raises: the assignment is often absent until hours before first pitch,
    and that is a normal state rather than a failure.
    """
    if not game_pk:
        return None
    try:
        payload = _fetch(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            cache_key=f"mlb:umpire:{game_pk}",
            verify_ssl=verify_ssl,
        )
    except Exception:
        return None

    officials = (((payload.get("liveData") or {}).get("boxscore") or {}).get("officials")) or []
    for official in officials:
        if not isinstance(official, dict):
            continue
        if str(official.get("officialType") or "").strip().lower() == "home plate":
            name = (official.get("official") or {}).get("fullName")
            if name:
                return str(name)
    return None


def pitcher_hand(pitcher: dict[str, Any] | None) -> str | None:
    """"L" or "R" for a probable starter, from whatever shape carries it.

    ESPN and the MLB Stats API spell this differently, and the same pipeline
    sees both depending on which enrichment step filled the pitcher in.
    """
    if not isinstance(pitcher, dict):
        return None

    for candidate in (
        (pitcher.get("pitchHand") or {}).get("code") if isinstance(pitcher.get("pitchHand"), dict) else None,
        pitcher.get("pitchHand") if isinstance(pitcher.get("pitchHand"), str) else None,
        pitcher.get("throws"),
        pitcher.get("hand"),
    ):
        if not candidate:
            continue
        code = str(candidate).strip().upper()[:1]
        if code in {"L", "R"}:
            return code
    return None


def handedness_matchup(game: dict[str, Any]) -> dict[str, Any] | None:
    """Which arm each starter throws with. None until both are known.

    Returns the facts, not an adjustment. A platoon edge needs each lineup's
    split against left- and right-handed pitching, and lineups are confirmed
    late; a league-average constant applied to every game would add noise while
    looking like analysis.
    """
    home = pitcher_hand(game.get("homePitcher"))
    away = pitcher_hand(game.get("awayPitcher"))
    if not home or not away:
        return None

    return {
        "homeStarterHand": home,
        "awayStarterHand": away,
        # True when the two starters throw with different arms, which is the
        # case where lineup construction matters most.
        "opposed": home != away,
        "note": (
            "Handedness only. No platoon adjustment is applied, because that "
            "needs lineup splits against each arm and lineups confirm late."
        ),
    }


def handedness_diff(game: dict[str, Any]) -> float | None:
    """Southpaw asymmetry, home minus away. Candidate feature only.

    +1 means only the home side starts a left-hander, -1 only the away side, 0
    means both or neither. Left-handed starting pitching is the scarcer
    commodity, so this asks whether one side has it and the other does not.
    """
    matchup = handedness_matchup(game)
    if not matchup:
        return None
    return float(
        (1 if matchup["homeStarterHand"] == "L" else 0)
        - (1 if matchup["awayStarterHand"] == "L" else 0)
    )
