"""Bullpen workload -- how much a relief corps has been asked to do lately.

Why this and not bullpen ERA
----------------------------
Bullpen quality is already fetched (`mlb_pitcher._team_bullpen_era`) and is
largely priced by the market, because it is a season-long number anyone can
look up. Recent *workload* is different: a bullpen that threw six innings
yesterday and five the day before has its best arms unavailable tonight, and
that is a same-day fact the closing line often lags.

That lag is the entire reason this is worth trying. A feature the market has
already absorbed cannot help a model that anchors to the market.

What is measured
----------------
Relief innings over the last few days, from the club's game log. Starters'
innings are excluded by subtracting them out -- a nine-inning complete game
rests a bullpen rather than taxing it, and counting total team innings would
score it as the heaviest possible day.

Nothing here moves a published probability. It is logged as an ablation
candidate and ships only if it beats its own absence out of sample.
"""

from __future__ import annotations

from typing import Any

from data_providers.utils import fetch_json, to_float

# Three days is the span over which a modern bullpen actually recovers: a
# reliever who worked back-to-back is usually unavailable on the third day.
WORKLOAD_DAYS = 3

# Roughly a normal share of relief innings per game. Used to centre the score so
# zero means "an ordinary few days" rather than "no innings at all".
TYPICAL_RELIEF_IP_PER_GAME = 3.4


def _fetch(path: str, params: dict[str, str], *, cache_key: str, verify_ssl: bool = True) -> dict:
    query = "?" + "&".join(f"{key}={value}" for key, value in params.items())
    return fetch_json(f"https://statsapi.mlb.com{path}{query}", cache_key=cache_key, verify_ssl=verify_ssl)


def team_relief_innings(
    team_id: int | None, *, days: int = WORKLOAD_DAYS, verify_ssl: bool = True
) -> float | None:
    """Relief innings thrown across the club's last few games. None on failure.

    Never raises. Workload decorates a prediction that is already made, so a
    failure here must cost this feature and nothing else.
    """
    if not team_id:
        return None
    try:
        payload = _fetch(
            f"/api/v1/teams/{team_id}/stats",
            {"stats": "gameLog", "group": "pitching", "season": "2026"},
            cache_key=f"mlb:bullpen:gamelog:{team_id}",
            verify_ssl=verify_ssl,
        )
    except Exception:
        return None

    relief = 0.0
    counted = 0
    for group in payload.get("stats") or []:
        for split in group.get("splits") or []:
            stat = split.get("stat") or {}
            total_ip = to_float(stat.get("inningsPitched"))
            if total_ip is None:
                continue
            # A complete game rests a bullpen. Subtracting the starter's work is
            # what stops that being scored as the heaviest possible day.
            starter_ip = _starter_innings(stat)
            if starter_ip is None:
                # Without the starter's share there is no way to separate relief
                # work from the whole game, so this is genuinely unknown.
                #
                # It previously fell back to "the whole game minus a typical
                # start", which looks harmless and is not: that returns exactly
                # the typical figure every time, so fatigue came out 0.00 for an
                # 18-inning day and a 9-inning day alike. The feature would have
                # logged numbers, passed its tests and never once shown signal.
                _warn_missing_starter_innings()
                return None
            relief += max(0.0, total_ip - starter_ip)
            counted += 1
            if counted >= days:
                break
        if counted >= days:
            break

    if not counted:
        return None
    return round(relief, 2)


# Field names the Stats API has been seen to use for the starters' share. Tried
# in order; if none is present the figure is unknown rather than guessed.
_STARTER_IP_KEYS = ("startersInningsPitched", "startersInnings", "starterInningsPitched")

_warned = False


def _starter_innings(stat: dict[str, Any]) -> float | None:
    for key in _STARTER_IP_KEYS:
        value = to_float(stat.get(key))
        if value is not None:
            return value
    return None


def _warn_missing_starter_innings() -> None:
    """Say so once per run, so a silently dead feature cannot stay silent.

    Printed rather than raised: this is a candidate feature and must not break a
    build. But an ablation candidate that is structurally incapable of producing
    a value needs to be visible, not discovered months later.
    """
    global _warned
    if _warned:
        return
    _warned = True
    print(
        "::warning title=Bullpen workload::MLB Stats API returned no starters' "
        "innings, so relief workload cannot be separated from total innings. "
        "bullpenDiff will stay empty until the field name is corrected."
    )


def bullpen_fatigue(
    team_id: int | None, *, days: int = WORKLOAD_DAYS, verify_ssl: bool = True
) -> float | None:
    """Relief innings above or below a normal few days. None when unknown.

    Positive means the bullpen has been worked harder than usual, which is a
    cost. Centred so an ordinary stretch scores zero rather than a constant.
    """
    innings = team_relief_innings(team_id, days=days, verify_ssl=verify_ssl)
    if innings is None:
        return None
    return round(innings - TYPICAL_RELIEF_IP_PER_GAME * days, 2)


def bullpen_edge(
    home_team_id: int | None,
    away_team_id: int | None,
    *,
    days: int = WORKLOAD_DAYS,
    verify_ssl: bool = True,
) -> float | None:
    """Home minus away fatigue, signed so positive favours the home side.

    Returns None unless both sides are known -- a one-sided figure would read as
    an edge when it is really a gap in the data.
    """
    home = bullpen_fatigue(home_team_id, days=days, verify_ssl=verify_ssl)
    away = bullpen_fatigue(away_team_id, days=days, verify_ssl=verify_ssl)
    if home is None or away is None:
        return None
    # A tired home bullpen is bad for the home side, hence away minus home.
    return round(away - home, 2)
