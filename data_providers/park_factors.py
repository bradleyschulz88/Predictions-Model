"""MLB park factors -- how much a ballpark inflates or suppresses run scoring.

Why a static table
------------------
Park factor is the one major baseball input that is not in any feed we already
pull. ESPN gives the venue name and nothing about how it plays, and the metric
itself is a multi-season regression (a single year of one park is ~81 games,
far too noisy to use raw), so it is published as a slow-moving constant rather
than fetched per game. A table is the honest representation of that.

The numbers are runs indices on the usual scale where **100 is neutral**: 110
means roughly 10% more runs are scored there than in an average park, for both
teams alike. They are approximate multi-year values and are deliberately
conservative -- the gap between Coors Field and Oracle Park is enormous and
uncontroversial, while the difference between two mid-table parks is inside the
noise. Treat the extremes as signal and the middle as roughly neutral.

**These need review each season.** Parks change: Baltimore moved its left field
wall back in 2022 and dropped from one of the best home-run parks to below
average. Two clubs are in temporary homes, which is exactly when a stale table
does the most damage.

Why this is a totals input, not a moneyline one
-----------------------------------------------
A park inflates scoring for *both* teams, so it barely moves who wins -- it
moves how many runs are scored. It is offered to the moneyline ablation anyway,
because the honest way to find out is to measure it, but the expectation is that
it fails there and earns its place in the totals model.
"""

from __future__ import annotations

from typing import Any

NEUTRAL_PARK_FACTOR = 100.0

# Keyed by the home club, because the home club's park is where the game is
# played. ESPN's displayName is the key so this joins straight onto `homeTeam`.
PARK_FACTORS: dict[str, float] = {
    # Thin air. The largest park effect in professional sport, by a distance.
    "Colorado Rockies": 115.0,
    # Hitter-friendly.
    "Cincinnati Reds": 105.0,
    "Boston Red Sox": 104.0,
    "Arizona Diamondbacks": 103.0,
    "Chicago Cubs": 102.0,
    "Philadelphia Phillies": 102.0,
    "Texas Rangers": 102.0,
    # Roughly neutral.
    "Atlanta Braves": 101.0,
    "Baltimore Orioles": 101.0,
    "Washington Nationals": 101.0,
    "Chicago White Sox": 100.0,
    "Kansas City Royals": 100.0,
    "Los Angeles Angels": 100.0,
    "Milwaukee Brewers": 100.0,
    "Minnesota Twins": 100.0,
    "New York Yankees": 100.0,
    "Toronto Blue Jays": 100.0,
    # Pitcher-friendly.
    "Cleveland Guardians": 99.0,
    "Houston Astros": 99.0,
    "Pittsburgh Pirates": 99.0,
    "Detroit Tigers": 98.0,
    "St. Louis Cardinals": 98.0,
    "Los Angeles Dodgers": 97.0,
    "Miami Marlins": 97.0,
    "New York Mets": 97.0,
    "San Diego Padres": 96.0,
    "San Francisco Giants": 96.0,
    "Seattle Mariners": 95.0,
    # Temporary homes -- both clubs are out of their permanent parks, and both
    # replacements are small, warm and open-air, so they play well above the
    # stadium each club is listed against historically. Revisit when they move.
    "Athletics": 105.0,
    "Tampa Bay Rays": 104.0,
}

# Neutral-site games (international series, weather relocations) are played
# somewhere neither club calls home, so the home club's park factor is simply
# wrong. Matched on the venue name we already capture.
_NEUTRAL_SITE_HINTS = (
    "london",
    "tokyo",
    "monterrey",
    "mexico city",
    "san juan",
    "williamsport",
    "dyersville",
    "field of dreams",
    "rickwood",
)


def is_neutral_site(venue_name: str | None) -> bool:
    """True when the venue is not either club's home park."""
    if not venue_name:
        return False
    lowered = str(venue_name).casefold()
    return any(hint in lowered for hint in _NEUTRAL_SITE_HINTS)


def park_factor(home_team: str | None, venue_name: str | None = None) -> float | None:
    """Runs index for the park this game is played in. None when unknown.

    Returns None rather than 100 for an unrecognised club, so "we have no idea"
    stays distinguishable from "we know it is neutral". A feature built on this
    must not treat a missing park as an average one.
    """
    if is_neutral_site(venue_name):
        return None
    if not home_team:
        return None
    return PARK_FACTORS.get(str(home_team).strip())


def park_run_environment(home_team: str | None, venue_name: str | None = None) -> dict[str, Any] | None:
    """Park factor plus a short human explanation, or None if unknown.

    `edge` is centred on zero and expressed in index points (so Coors is +15,
    T-Mobile is -5), which is the form a model wants: a neutral park contributes
    nothing rather than a constant.
    """
    factor = park_factor(home_team, venue_name)
    if factor is None:
        return None

    edge = factor - NEUTRAL_PARK_FACTOR
    if edge >= 8:
        note = "a strong hitters' park"
    elif edge >= 3:
        note = "a mild hitters' park"
    elif edge <= -3:
        note = "a pitchers' park"
    else:
        note = "a neutral park"

    return {
        "factor": factor,
        "edge": round(edge, 1),
        # Multiplier form, for anything scaling an expected run total directly.
        "multiplier": round(factor / NEUTRAL_PARK_FACTOR, 4),
        "note": note,
    }
