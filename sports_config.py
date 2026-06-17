"""League definitions for the multi-sport dashboard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueConfig:
    id: str
    label: str
    short_label: str
    espn_path: str
    sbr_odds_slug: str | None
    supports_pitchers: bool
    supports_draw: bool
    lineup_label: str
    default_days_ahead: int

    @property
    def supports_sbr_odds(self) -> bool:
        return bool(self.sbr_odds_slug)


LEAGUES: dict[str, LeagueConfig] = {
    "mlb": LeagueConfig(
        id="mlb",
        label="MLB Baseball",
        short_label="MLB",
        espn_path="baseball/mlb",
        sbr_odds_slug="mlb-baseball",
        supports_pitchers=True,
        supports_draw=False,
        lineup_label="Batting lineup",
        default_days_ahead=0,
    ),
    "nfl": LeagueConfig(
        id="nfl",
        label="NFL Football",
        short_label="NFL",
        espn_path="football/nfl",
        sbr_odds_slug="nfl-football",
        supports_pitchers=False,
        supports_draw=False,
        lineup_label="Key players",
        default_days_ahead=0,
    ),
    "nba": LeagueConfig(
        id="nba",
        label="NBA Basketball",
        short_label="NBA",
        espn_path="basketball/nba",
        sbr_odds_slug="nba-basketball",
        supports_pitchers=False,
        supports_draw=False,
        lineup_label="Key players",
        default_days_ahead=0,
    ),
    "wnba": LeagueConfig(
        id="wnba",
        label="WNBA Basketball",
        short_label="WNBA",
        espn_path="basketball/wnba",
        sbr_odds_slug="wnba-basketball",
        supports_pitchers=False,
        supports_draw=False,
        lineup_label="Key players",
        default_days_ahead=0,
    ),
    "worldcup": LeagueConfig(
        id="worldcup",
        label="FIFA World Cup",
        short_label="World Cup",
        espn_path="soccer/fifa.world",
        sbr_odds_slug=None,
        supports_pitchers=False,
        supports_draw=True,
        lineup_label="Key players",
        default_days_ahead=0,
    ),
    "epl": LeagueConfig(
        id="epl",
        label="English Premier League",
        short_label="EPL",
        espn_path="soccer/eng.1",
        sbr_odds_slug="english-premier-league",
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
        sbr_odds_slug=None,
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
