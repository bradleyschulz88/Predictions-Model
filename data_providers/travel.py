"""Travel and body-clock context for MLB.

What this measures
------------------
Two things a schedule does to a team that a win-loss record does not capture:

* **Timezone shift** -- playing in a zone your body clock is not in. The effect
  is asymmetric: flying east costs more than flying west, because it shortens
  the day and asks a player to perform earlier in circadian terms than usual.
  This is the better-supported of the two effects.
* **Distance** -- a coast-to-coast trip is more taxing than a division game
  three hours up the road, independent of the clock.

The home side is treated as having neither, which is the common case and the
right default: a club at home is in its own bed and its own zone. That is also
part of what home-field advantage already measures, which is why this is offered
as an *away* penalty rather than a home bonus -- otherwise it would be counting
the same thing twice.

Honest limits
-------------
This uses the visiting club's **home** timezone, not where they actually played
last. A team finishing a west-coast road trip and arriving in New York has no
real eastbound shift left to absorb, and this will still score one. Fixing that
needs each club's previous fixture location, which the schedule context could
supply later. Until then it is a reasonable proxy and is stated as such.

Nothing here moves a published probability. It is logged as an ablation
candidate and ships only if it beats its own absence out of sample.
"""

from __future__ import annotations

from typing import Any

# Home city coordinates and timezone per club. Coordinates are the ballpark's,
# rounded -- a few miles either way cannot matter at this scale.
TEAM_HOME: dict[str, dict[str, Any]] = {
    "Arizona Diamondbacks": {"lat": 33.45, "lon": -112.07, "utc": -7, "dst": False},
    "Atlanta Braves": {"lat": 33.89, "lon": -84.47, "utc": -5, "dst": True},
    "Baltimore Orioles": {"lat": 39.28, "lon": -76.62, "utc": -5, "dst": True},
    "Boston Red Sox": {"lat": 42.35, "lon": -71.10, "utc": -5, "dst": True},
    "Chicago Cubs": {"lat": 41.95, "lon": -87.66, "utc": -6, "dst": True},
    "Chicago White Sox": {"lat": 41.83, "lon": -87.63, "utc": -6, "dst": True},
    "Cincinnati Reds": {"lat": 39.10, "lon": -84.51, "utc": -5, "dst": True},
    "Cleveland Guardians": {"lat": 41.50, "lon": -81.69, "utc": -5, "dst": True},
    "Colorado Rockies": {"lat": 39.76, "lon": -104.99, "utc": -7, "dst": True},
    "Detroit Tigers": {"lat": 42.34, "lon": -83.05, "utc": -5, "dst": True},
    "Houston Astros": {"lat": 29.76, "lon": -95.36, "utc": -6, "dst": True},
    "Kansas City Royals": {"lat": 39.05, "lon": -94.48, "utc": -6, "dst": True},
    "Los Angeles Angels": {"lat": 33.80, "lon": -117.88, "utc": -8, "dst": True},
    "Los Angeles Dodgers": {"lat": 34.07, "lon": -118.24, "utc": -8, "dst": True},
    "Miami Marlins": {"lat": 25.78, "lon": -80.22, "utc": -5, "dst": True},
    "Milwaukee Brewers": {"lat": 43.03, "lon": -87.97, "utc": -6, "dst": True},
    "Minnesota Twins": {"lat": 44.98, "lon": -93.28, "utc": -6, "dst": True},
    "New York Mets": {"lat": 40.76, "lon": -73.85, "utc": -5, "dst": True},
    "New York Yankees": {"lat": 40.83, "lon": -73.93, "utc": -5, "dst": True},
    "Philadelphia Phillies": {"lat": 39.91, "lon": -75.17, "utc": -5, "dst": True},
    "Pittsburgh Pirates": {"lat": 40.45, "lon": -80.01, "utc": -5, "dst": True},
    "San Diego Padres": {"lat": 32.71, "lon": -117.16, "utc": -8, "dst": True},
    "San Francisco Giants": {"lat": 37.78, "lon": -122.39, "utc": -8, "dst": True},
    "Seattle Mariners": {"lat": 47.59, "lon": -122.33, "utc": -8, "dst": True},
    "St. Louis Cardinals": {"lat": 38.62, "lon": -90.19, "utc": -6, "dst": True},
    "Texas Rangers": {"lat": 32.75, "lon": -97.08, "utc": -6, "dst": True},
    "Toronto Blue Jays": {"lat": 43.64, "lon": -79.39, "utc": -5, "dst": True},
    "Washington Nationals": {"lat": 38.87, "lon": -77.01, "utc": -5, "dst": True},
    # Temporary homes, matching where these clubs actually play.
    "Athletics": {"lat": 38.58, "lon": -121.51, "utc": -8, "dst": True},  # Sacramento
    "Tampa Bay Rays": {"lat": 27.98, "lon": -82.51, "utc": -5, "dst": True},  # Tampa
}

# Arizona does not observe daylight saving, so its offset to the eastern clubs
# changes by an hour across the season. Handled by the `dst` flag above rather
# than hard-coding a single number.
EARTH_RADIUS_KM = 6371.0

# Eastbound travel is the harder direction. The multiplier is deliberately
# modest -- published effects are real but small, and this is a candidate
# feature, not a thesis.
EASTBOUND_PENALTY = 1.5


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def _offset(entry: dict[str, Any], *, daylight_saving: bool) -> float:
    """UTC offset in hours, accounting for whether DST is in effect."""
    return entry["utc"] + (1 if daylight_saving and entry.get("dst") else 0)


def travel_context(
    home_team: str | None,
    away_team: str | None,
    *,
    daylight_saving: bool = True,
) -> dict[str, Any] | None:
    """Distance and body-clock shift the visiting club is carrying.

    Returns None when either club is unknown, rather than zero, so "we could not
    work it out" stays distinguishable from "there is no travel".
    """
    home = TEAM_HOME.get(str(home_team or "").strip())
    away = TEAM_HOME.get(str(away_team or "").strip())
    if not home or not away:
        return None

    distance = _haversine_km(home["lat"], home["lon"], away["lat"], away["lon"])
    # Positive means the visitors moved east, which is the costly direction.
    shift = _offset(home, daylight_saving=daylight_saving) - _offset(
        away, daylight_saving=daylight_saving
    )

    # One score, signed so that positive always means "harder on the visitors".
    # Eastbound is weighted more heavily; distance contributes on its own scale.
    weighted_shift = abs(shift) * (EASTBOUND_PENALTY if shift > 0 else 1.0)
    burden = round(weighted_shift + distance / 2000.0, 3)

    return {
        "distanceKm": round(distance),
        "timezoneShift": round(shift, 1),
        "direction": "east" if shift > 0 else "west" if shift < 0 else "same",
        # Negative because it is a cost to the away side, expressed on the same
        # home-minus-away convention every other feature uses.
        "awayBurden": burden,
        "homeEdge": round(burden, 3),
        "note": (
            "Uses the visiting club's home timezone, not where they last played, "
            "so a team already on a road trip is over-penalised."
        ),
    }


def travel_edge(home_team: str | None, away_team: str | None, **kwargs: Any) -> float | None:
    """Single number for the feature log. Positive favours the home side."""
    context = travel_context(home_team, away_team, **kwargs)
    return context["homeEdge"] if context else None
