"""League definitions for the multi-sport dashboard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueConfig:
    id: str
    label: str
    short_label: str
    espn_path: str
    supports_sbr_odds: bool
    supports_pitchers: bool
    supports_draw: bool
    lineup_label: str
    default_days_ahead: int


LEAGUES: dict[str, LeagueConfig] = {
    "mlb": LeagueConfig(
        id="mlb",
        label="MLB Baseball",
        short_label="MLB",
        espn_path="baseball/mlb",
        supports_sbr_odds=True,
        supports_pitchers=True,
        supports_draw=False,
        lineup_label="Batting lineup",
        default_days_ahead=1,
    ),
    "worldcup": LeagueConfig(
        id="worldcup",
        label="FIFA World Cup",
        short_label="World Cup",
        espn_path="soccer/fifa.world",
        supports_sbr_odds=False,
        supports_pitchers=False,
        supports_draw=True,
        lineup_label="Key players",
        default_days_ahead=0,
    ),
    "afl": LeagueConfig(
        id="afl",
        label="AFL",
        short_label="AFL",
        espn_path="australian-football/afl",
        supports_sbr_odds=False,
        supports_pitchers=False,
        supports_draw=False,
        lineup_label="Key players",
        default_days_ahead=0,
    ),
}


def get_league(league_id: str) -> LeagueConfig:
    league = LEAGUES.get(league_id)
    if league is None:
        raise ValueError(f"Unknown league: {league_id}")
    return league


def list_league_ids() -> list[str]:
    return list(LEAGUES.keys())
