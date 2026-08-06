"""Bullpen workload -- how much a relief corps has been asked to do lately.

Why this and not bullpen ERA
----------------------------
Bullpen quality is already fetched (`mlb_pitcher._team_bullpen_era`) and is
largely priced by the market, because it is a season-long number anyone can look
up. Recent *workload* is different: a bullpen that threw six innings yesterday
and five the day before has its best arms unavailable tonight, and that is a
same-day fact the closing line often lags.

That lag is the entire reason this is worth trying. A feature the market has
already absorbed cannot help a model that anchors to the market.

Why the boxscore, and what it costs
-----------------------------------
The first version of this read the team game log and tried to subtract the
starter's innings out. That endpoint cannot do it. A production build printed
every field it returns -- 59 of them, including `gamesStarted`, `completeGames`,
`gamesFinished`, `saves` and `holds` -- and **not one separates starters from
relievers**. The feature was unbuildable from there and reported nothing on
every game.

The boxscore does carry the split: it lists a side's pitchers in the order they
appeared, so the first is the starter and the rest are relief. That costs one
schedule call plus one boxscore per game examined, roughly four requests per
club. That is the price of the only endpoint that answers the question.

Nothing here moves a published probability. It is logged as an ablation
candidate and ships only if it beats its own absence out of sample.
"""

from __future__ import annotations


from data_providers.utils import fetch_json, to_float

# Three days is the span over which a modern bullpen actually recovers: a
# reliever who worked back-to-back is usually unavailable on the third day.
WORKLOAD_DAYS = 3

# Roughly a normal share of relief innings per game. Used to centre the score so
# zero means "an ordinary few days" rather than "no innings at all".
TYPICAL_RELIEF_IP_PER_GAME = 3.4

_warned = False


def _fetch(path: str, params: dict[str, str], *, cache_key: str, verify_ssl: bool = True) -> dict:
    query = ("?" + "&".join(f"{key}={value}" for key, value in params.items())) if params else ""
    return fetch_json(
        f"https://statsapi.mlb.com{path}{query}", cache_key=cache_key, verify_ssl=verify_ssl
    )


def _warn_unreadable_boxscore() -> None:
    """Say so once if no boxscore could be read, so a dead feature stays visible.

    Printed rather than raised: this is a candidate feature and must not break a
    build. But a candidate structurally incapable of producing a value needs to
    be seen, not discovered months later in an ablation that reports it as
    "tested and did not help".
    """
    global _warned
    if _warned:
        return
    _warned = True
    print(
        "::warning title=Bullpen workload::no boxscore yielded a pitcher list, so "
        "relief innings could not be measured and bullpenDiff stays empty. The "
        "team game log cannot substitute -- it returns 59 pitching fields and "
        "none of them separates starters from relievers."
    )


def _recent_game_pks(team_id: int, *, days: int, verify_ssl: bool = True) -> list[str]:
    """Completed game ids for a club, most recent first."""
    from datetime import date, timedelta

    today = date.today()
    # A few extra days of slack, because clubs have off-days and a three-game
    # window is three games played, not three calendar days.
    params = {
        "sportId": "1",
        "teamId": str(team_id),
        "startDate": (today - timedelta(days=days + 5)).isoformat(),
        "endDate": today.isoformat(),
    }
    payload = _fetch(
        "/api/v1/schedule",
        params,
        cache_key=f"mlb:bullpen:schedule:{team_id}:{today.isoformat()}",
        verify_ssl=verify_ssl,
    )

    finished: list[tuple[str, str]] = []
    for day in payload.get("dates") or []:
        for game in day.get("games") or []:
            state = ((game.get("status") or {}).get("abstractGameState") or "").lower()
            if state != "final":
                continue
            game_pk = game.get("gamePk")
            if game_pk:
                finished.append((str(day.get("date") or ""), str(game_pk)))
    finished.sort(reverse=True)
    return [game_pk for _, game_pk in finished]


def relief_innings_in_game(
    game_pk: str, team_id: int, *, verify_ssl: bool = True
) -> float | None:
    """Innings the club's bullpen threw in one game. None if unreadable.

    `pitchers` is ordered by appearance, so index 0 is the starter and the rest
    are relief. A single-entry list is a complete game, which rested the bullpen
    -- that is zero innings, not unknown, and the distinction matters because
    zero is data and None is not.
    """
    payload = _fetch(
        f"/api/v1/game/{game_pk}/boxscore",
        {},
        cache_key=f"mlb:bullpen:box:{game_pk}",
        verify_ssl=verify_ssl,
    )
    for side in ("home", "away"):
        block = ((payload.get("teams") or {}).get(side)) or {}
        if ((block.get("team") or {}).get("id")) != team_id:
            continue
        pitchers = block.get("pitchers") or []
        if not pitchers:
            return None
        if len(pitchers) == 1:
            return 0.0
        players = block.get("players") or {}
        relief = 0.0
        for player_id in pitchers[1:]:
            stats = ((players.get(f"ID{player_id}") or {}).get("stats") or {}).get("pitching") or {}
            innings = to_float(stats.get("inningsPitched"))
            if innings:
                relief += innings
        return round(relief, 2)
    return None


def team_relief_innings(
    team_id: int | None, *, days: int = WORKLOAD_DAYS, verify_ssl: bool = True
) -> float | None:
    """Relief innings across the club's last few completed games. None on failure.

    Never raises. Workload decorates a prediction that is already made, so a
    failure here must cost this feature and nothing else.
    """
    if not team_id:
        return None
    try:
        game_pks = _recent_game_pks(team_id, days=days, verify_ssl=verify_ssl)
    except Exception:
        return None
    if not game_pks:
        return None

    total = 0.0
    counted = 0
    for game_pk in game_pks[:days]:
        try:
            innings = relief_innings_in_game(game_pk, team_id, verify_ssl=verify_ssl)
        except Exception:
            continue
        if innings is None:
            continue
        total += innings
        counted += 1

    if not counted:
        _warn_unreadable_boxscore()
        return None
    # Scaled to the window, so a club that has played two games in three days is
    # not scored as though it rested for the third.
    return round(total / counted * days, 2)


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
