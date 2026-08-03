"""Derived signals from schedule and enrichment (rest, weather, power index)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def parse_weather_impact(weather: str | None) -> dict[str, Any] | None:
    if not weather:
        return None
    temp_match = re.search(r"(-?\d+)\s*°", weather)
    precip_match = re.search(r"(\d+)%\s*precip", weather, re.I)
    wind_match = re.search(r"(\d+)\s*mph\s*wind", weather, re.I)
    temp_f = float(temp_match.group(1)) if temp_match else None
    precip = float(precip_match.group(1)) if precip_match else None
    wind = float(wind_match.group(1)) if wind_match else None

    run_env = 0.0
    notes: list[str] = []
    if temp_f is not None:
        if temp_f >= 82:
            run_env += 0.04
            notes.append("warm air favors offense")
        elif temp_f <= 50:
            run_env -= 0.03
            notes.append("cold air suppresses scoring")
    if wind is not None and wind >= 12:
        run_env += 0.02
        notes.append("wind can affect fly balls")
    if precip is not None and precip >= 40:
        notes.append("rain risk")

    return {
        "temperatureF": temp_f,
        "precipitationPct": precip,
        "windMph": wind,
        "runEnvironmentAdj": run_env,
        "summary": "; ".join(notes) if notes else None,
    }


def series_win_pct(
    series: dict[str, Any] | None,
    team_name: str | None,
    abbrev: str | None = None,
) -> float | None:
    """Share of the season series this club holds, from ESPN's summary string.

    ESPN writes the summary with the club's ABBREVIATION, not its name:
    "TB wins series 5-1". Matching on the full name and the nickname therefore
    never matched, and a real build reported the result exactly -- of 25 games,
    25 carried a season series, 25 carried a score, and 0 resolved to a win
    share. Passing the abbreviation, which the scoreboard already gives us, is
    what makes this work at all.

    The name and nickname are still tried, because other leagues word it
    differently ("Arsenal lead series 2-1").
    """
    if not series or not (team_name or abbrev):
        return None
    summary = series.get("summary") or ""
    score = series.get("seriesScore") or ""
    if not summary or not score:
        return None

    match = re.search(r"(\d+)\s*-\s*(\d+)", score)
    if not match:
        return None
    left, right = int(match.group(1)), int(match.group(2))
    total = left + right
    if not total:
        return None

    # Longest first, so "Chicago White Sox" is preferred over "Sox".
    candidates = [token for token in (team_name, abbrev) if token]
    if team_name:
        candidates.append(team_name.split()[-1])
    candidates.sort(key=len, reverse=True)

    position = None
    for token in candidates:
        # Word-boundary matched: a bare "TB" must not be found inside "TBR" or
        # some other club's nickname.
        found = re.search(rf"\b{re.escape(token)}\b", summary, re.IGNORECASE)
        if found:
            position = found.start()
            break
    if position is None:
        return None

    # The club named first in the summary holds the first number.
    return left / total if position < len(summary) / 2 else right / total


def compute_rest_days(games: list[dict[str, Any]], team_name: str | None, current_start: str | None) -> int | None:
    if not team_name or not current_start:
        return None
    try:
        current_dt = datetime.fromisoformat(current_start.replace("Z", "+00:00"))
    except ValueError:
        return None

    previous: datetime | None = None
    for game in games:
        start = game.get("startDate")
        if not start:
            continue
        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            continue
        if start_dt >= current_dt:
            continue
        if team_name not in {game.get("homeTeam"), game.get("awayTeam")}:
            continue
        if previous is None or start_dt > previous:
            previous = start_dt

    if previous is None:
        return None
    delta = current_dt - previous
    return max(0, delta.days)


# Typical team points-per-game band per league, used to put the scoring term on
# the same 0-1 scale as win percentage before they are averaged together.
#
# One shared divisor used to serve all four of these leagues, and it was
# basketball-sized. NFL clubs score about 22 a game, so dividing by 130 mapped
# the entire league into 0.115-0.231 -- a 0.115 spread against win percentage's
# full 1.0 -- and the term became a near-constant that diluted the real signal
# rather than adding to it. Measured on the same underlying quality gap (.700
# against .300), powerDiff came out 0.40 for MLB but only ~0.30 for these four,
# so strengthDiff was not comparable across leagues while the fit gives it a
# single shared coefficient.
#
# Bands are the ordinary spread of team scoring averages, not record extremes;
# anything outside clamps, which is the intended behaviour for an outlier.
LEAGUE_SCORING_BAND = {
    "nba": (105.0, 125.0),
    "wnba": (72.0, 92.0),
    "nfl": (15.0, 30.0),
    "afl": (70.0, 100.0),
}


def _scoring_strength(league: str, points_per_game: float) -> float | None:
    """Points per game as a 0-1 score against that league's own scoring band."""
    band = LEAGUE_SCORING_BAND.get(league)
    if band is None:
        return None
    low, high = band
    if high <= low:
        return None
    return max(0.0, min(1.0, (float(points_per_game) - low) / (high - low)))


def compute_power_rating(
    *,
    league: str,
    win_pct: float | None,
    run_diff_per_game: float | None = None,
    goals_for_per_game: float | None = None,
    goals_against_per_game: float | None = None,
    form_pct: float | None = None,
    batting_ops_proxy: float | None = None,
    era: float | None = None,
) -> float | None:
    parts: list[tuple[float, float]] = []
    if win_pct is not None:
        parts.append((win_pct, 0.35))
    if form_pct is not None and form_pct >= 0:
        parts.append((form_pct, 0.20))
    if league == "mlb":
        if run_diff_per_game is not None:
            normalized = max(0.0, min(1.0, (run_diff_per_game + 2.0) / 4.0))
            parts.append((normalized, 0.25))
        if batting_ops_proxy is not None:
            parts.append((max(0.0, min(1.0, (batting_ops_proxy - 0.650) / 0.150)), 0.10))
        if era is not None:
            parts.append((max(0.0, min(1.0, (5.5 - era) / 2.5)), 0.10))
    elif league == "epl":
        if goals_for_per_game is not None and goals_against_per_game is not None:
            attack = max(0.0, min(1.0, goals_for_per_game / 3.0))
            defense = max(0.0, min(1.0, (3.0 - goals_against_per_game) / 2.0))
            parts.append((attack, 0.20))
            parts.append((defense, 0.15))
    elif league in LEAGUE_SCORING_BAND:
        if goals_for_per_game is not None:
            scoring = _scoring_strength(league, goals_for_per_game)
            if scoring is not None:
                parts.append((scoring, 0.15))

    if not parts:
        return None
    weight_total = sum(weight for _, weight in parts)
    return sum(value * weight for value, weight in parts) / weight_total


def merge_team_profile(
    *,
    league: str,
    espn_stats: dict[str, Any] | None,
    espn_standings: dict[str, Any] | None,
    mlb_official: dict[str, Any] | None,
    form_pct: float | None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {"sources": []}

    if espn_standings:
        profile.update(
            {
                "wins": espn_standings.get("wins"),
                "losses": espn_standings.get("losses"),
                "points": espn_standings.get("points"),
                "goalDifference": espn_standings.get("goalDifference"),
                "pointsPerGame": espn_standings.get("pointsPerGame"),
                "goalsAgainstPerGame": espn_standings.get("goalsAgainstPerGame"),
            }
        )
        profile["sources"].append("ESPN standings")

    if mlb_official:
        profile.update(
            {
                "runDifferential": mlb_official.get("runDifferential"),
                "runsPerGame": mlb_official.get("runsPerGame"),
                "runsAllowedPerGame": mlb_official.get("runsAllowedPerGame"),
                "streakCode": mlb_official.get("streakCode"),
                "streakNumber": mlb_official.get("streakNumber"),
                "streakType": mlb_official.get("streakType"),
            }
        )
        profile["sources"].append("MLB.com")

    if espn_stats:
        profile.update(
            {
                "battingAvg": espn_stats.get("battingAvg"),
                "onBasePct": espn_stats.get("onBasePct"),
                "sluggingPct": espn_stats.get("sluggingPct"),
                "era": espn_stats.get("era"),
                "runsScored": espn_stats.get("runsScored"),
                "runsAllowed": espn_stats.get("runsAllowed"),
            }
        )
        profile["sources"].append("ESPN team stats")

    win_pct = None
    if espn_standings and espn_standings.get("winPct") is not None:
        win_pct = espn_standings["winPct"]
    elif mlb_official and mlb_official.get("winPct") is not None:
        win_pct = mlb_official["winPct"]

    run_diff_pg = None
    if mlb_official and mlb_official.get("runDifferential") is not None:
        games = mlb_official.get("gamesPlayed") or 1
        run_diff_pg = mlb_official["runDifferential"] / games

    ops_proxy = None
    if espn_stats and espn_stats.get("onBasePct") is not None and espn_stats.get("sluggingPct") is not None:
        ops_proxy = espn_stats["onBasePct"] + espn_stats["sluggingPct"]

    profile["powerRating"] = compute_power_rating(
        league=league,
        win_pct=win_pct,
        run_diff_per_game=run_diff_pg,
        goals_for_per_game=espn_standings.get("pointsPerGame") if espn_standings else None,
        goals_against_per_game=espn_standings.get("goalsAgainstPerGame") if espn_standings else None,
        form_pct=form_pct,
        batting_ops_proxy=ops_proxy,
        era=espn_stats.get("era") if espn_stats else None,
    )
    profile["opsProxy"] = ops_proxy
    profile["winPct"] = win_pct
    profile["sources"] = sorted(set(profile["sources"]))
    return profile
