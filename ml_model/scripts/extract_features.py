#!/usr/bin/env python3
"""
Feature extraction from ESPN/SBR fixtures for ML model training.
Converts raw JSON fixtures into structured feature DataFrames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from datetime import datetime


FIXTURE_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class GameFeatures:
    """Flattened features for a single game."""
    # Identifiers
    event_id: str
    date: str
    league: str
    season: int
    
    # Teams
    home_team: str
    away_team: str
    
    # Target (what we're predicting)
    home_win: int | None = None      # 1 if home won, 0 if away won, None if not final
    home_score: int | None = None
    away_score: int | None = None
    
    # Team records (season)
    home_wins: int = 0
    home_losses: int = 0
    home_draws: int = 0
    away_wins: int = 0
    away_losses: int = 0
    away_draws: int = 0
    
    # Home/away splits
    home_home_wins: int = 0
    home_home_losses: int = 0
    away_away_wins: int = 0
    away_away_losses: int = 0
    
    # Last 5 form (W=1, D=0.5, L=0)
    home_form_pts: float = 0.0
    away_form_pts: float = 0.0
    home_form_results: str = ""  # e.g., "W,W,L,W,L"
    away_form_results: str = ""
    
    # Starting pitcher stats (MLB)
    home_pitcher_era: float | None = None
    away_pitcher_era: float | None = None
    home_pitcher_fip: float | None = None
    away_pitcher_fip: float | None = None
    home_pitcher_recent_era: float | None = None
    away_pitcher_recent_era: float | None = None
    
    # Team advanced stats
    home_power_rating: float | None = None
    away_power_rating: float | None = None
    home_ops: float | None = None
    away_ops: float | None = None
    home_era: float | None = None
    away_era: float | None = None
    home_run_diff: float | None = None
    away_run_diff: float | None = None
    
    # Rest days
    home_rest_days: int | None = None
    away_rest_days: int | None = None
    
    # Injuries (count of major injuries)
    home_major_injuries: int = 0
    away_major_injuries: int = 0
    
    # Lineup quality (OPS-weighted)
    home_lineup_ops: float | None = None
    away_lineup_ops: float | None = None
    
    # Market odds (if available)
    market_home_pct: float | None = None
    market_away_pct: float | None = None
    market_draw_pct: float | None = None
    
    # ESPN predictor
    espn_home_pct: float | None = None
    espn_away_pct: float | None = None
    
    # Schedule flags
    home_back_to_back: int = 0
    away_back_to_back: int = 0
    home_long_road_trip: int = 0
    away_long_road_trip: int = 0
    
    # Weather (if available)
    temperature: float | None = None
    wind_speed: float | None = None
    is_dome: int = 0
    
    # Head-to-head
    h2h_home_wins: int = 0
    h2h_away_wins: int = 0
    
    # Derived features (computed)
    home_win_pct: float = 0.0
    away_win_pct: float = 0.0
    home_home_win_pct: float = 0.0
    away_away_win_pct: float = 0.0
    form_diff: float = 0.0
    era_diff: float = 0.0
    power_diff: float = 0.0
    ops_diff: float = 0.0
    run_diff_diff: float = 0.0
    rest_diff: float = 0.0
    injury_diff: float = 0.0
    lineup_ops_diff: float = 0.0
    market_edge_home: float | None = None  # model will learn this
    

def safe_get(d: dict, *keys, default=None):
    """Safely navigate nested dict."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d if d is not None else default


def extract_record(team_data: dict) -> tuple[int, int, int]:
    """Extract W-L-D from team record."""
    record = safe_get(team_data, "record", default={})
    wins = safe_get(record, "wins", default=0)
    losses = safe_get(record, "losses", default=0)
    draws = safe_get(record, "draws", default=0)
    return int(wins), int(losses), int(draws)


def extract_home_away_record(team_data: dict, is_home: bool) -> tuple[int, int]:
    """Extract home/away specific record."""
    record = safe_get(team_data, "record", default={})
    if is_home:
        return int(safe_get(record, "homeWins", default=0)), int(safe_get(record, "homeLosses", default=0))
    return int(safe_get(record, "awayWins", default=0)), int(safe_get(record, "awayLosses", default=0))


def parse_last_five(events: list[dict]) -> tuple[float, str]:
    """Parse last 5 games: returns (form_points, result_string).
    W=1, D=0.5, L=0 per game."""
    if not events:
        return 0.0, ""
    
    results = []
    pts = 0.0
    for e in events[:5]:
        res = safe_get(e, "gameResult")
        if res == "W":
            results.append("W")
            pts += 1.0
        elif res == "D":
            results.append("D")
            pts += 0.5
        elif res == "L":
            results.append("L")
            pts += 0.0
    return pts, ",".join(results)


def extract_pitcher_stats(pitcher: dict) -> dict:
    """Extract pitcher ERA, FIP, recent ERA."""
    era = safe_get(pitcher, "era")
    fip = safe_get(pitcher, "fip")
    recent_era = safe_get(pitcher, "recentEra")
    return {
        "era": float(era) if era is not None else None,
        "fip": float(fip) if fip is not None else None,
        "recent_era": float(recent_era) if recent_era is not None else None,
    }


def extract_advanced(team_adv: dict, prefix: str) -> dict:
    """Extract advanced team metrics."""
    return {
        f"{prefix}power_rating": safe_get(team_adv, "powerRating"),
        f"{prefix}ops": safe_get(team_adv, "ops"),
        f"{prefix}era": safe_get(team_adv, "era"),
        f"{prefix}run_diff": safe_get(team_adv, "runDifferential"),
    }


def extract_lineup_ops(lineup: dict) -> float | None:
    """Compute OPS-weighted lineup quality."""
    batters = safe_get(lineup, "batters", default=[])
    if not batters:
        return None
    ops_vals = []
    for b in batters:
        ops = safe_get(b, "ops")
        if ops is not None:
            try:
                ops_vals.append(float(ops))
            except (ValueError, TypeError):
                pass
    return float(np.mean(ops_vals)) if ops_vals else None


def extract_injury_count(injuries: list[dict], team_name: str) -> int:
    """Count major injuries for a team."""
    if not injuries:
        return 0
    count = 0
    for inj in injuries:
        team = safe_get(inj, "team", "displayName")
        status = safe_get(inj, "status")
        if team == team_name and status in ("Out", "60-Day IL", "10-Day IL", "IL"):
            count += 1
    return count


def extract_rest_days(rest: dict, team_name: str) -> int | None:
    """Extract rest days for a team."""
    if not rest:
        return None
    for block in rest:
        if safe_get(block, "team", "displayName") == team_name:
            return int(safe_get(block, "days", default=0))
    return None


def extract_schedule_flags(flags: dict, team_name: str) -> dict:
    """Extract schedule fatigue flags."""
    if not flags:
        return {"back_to_back": 0, "long_road_trip": 0}
    for block in flags:
        if safe_get(block, "team", "displayName") == team_name:
            return {
                "back_to_back": int(safe_get(block, "backToBack", default=0)),
                "long_road_trip": int(safe_get(block, "longRoadTrip", default=0)),
            }
    return {"back_to_back": 0, "long_road_trip": 0}


def extract_weather(weather: dict) -> dict:
    """Extract weather data."""
    return {
        "temperature": safe_get(weather, "temperature"),
        "wind_speed": safe_get(weather, "windSpeed"),
        "is_dome": int(safe_get(weather, "indoor", default=False)),
    }


def extract_h2h(h2h: dict, home_team: str, away_team: str) -> dict:
    """Extract head-to-head record."""
    if not h2h:
        return {"h2h_home_wins": 0, "h2h_away_wins": 0}
    
    # Look for season series or head-to-head games
    home_wins = 0
    away_wins = 0
    
    for block in safe_get(h2h, "events", default=[]):
        comp = safe_get(block, "competition", default=block)
        competitors = safe_get(comp, "competitors", default=[])
        for c in competitors:
            if safe_get(c, "team", "displayName") == home_team:
                if safe_get(c, "winner", default=False):
                    home_wins += 1
            elif safe_get(c, "team", "displayName") == away_team:
                if safe_get(c, "winner", default=False):
                    away_wins += 1
    
    return {"h2h_home_wins": home_wins, "h2h_away_wins": away_wins}


def extract_odds(odds: dict) -> dict:
    """Extract market implied probabilities from odds."""
    if not odds:
        return {"market_home_pct": None, "market_away_pct": None, "market_draw_pct": None}
    
    # Look for moneyline odds
    home_ml = None
    away_ml = None
    draw_ml = None
    
    # This depends on the odds structure - adapt as needed
    for book, lines in odds.items():
        if isinstance(lines, dict):
            if "home" in lines and home_ml is None:
                home_ml = lines["home"]
            if "away" in lines and away_ml is None:
                away_ml = lines["away"]
            if "draw" in lines and draw_ml is None:
                draw_ml = lines["draw"]
    
    def ml_to_pct(ml):
        if ml is None:
            return None
        try:
            ml = int(ml)
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return abs(ml) / (abs(ml) + 100)
        except (ValueError, TypeError):
            return None
    
    home_pct = ml_to_pct(home_ml)
    away_pct = ml_to_pct(away_ml)
    draw_pct = ml_to_pct(draw_ml)
    
    # Normalize (remove vig)
    total = sum(p for p in [home_pct, away_pct, draw_pct] if p is not None)
    if total and total > 0:
        if home_pct: home_pct = home_pct / total
        if away_pct: away_pct = away_pct / total
        if draw_pct: draw_pct = draw_pct / total
    
    return {
        "market_home_pct": home_pct,
        "market_away_pct": away_pct,
        "market_draw_pct": draw_pct,
    }


def process_game(game: dict, enrichment: dict | None, odds: dict | None) -> GameFeatures:
    """Process a single game into features."""
    event_id = str(safe_get(game, "eventId", default=""))
    date_str = safe_get(game, "startDate", default="")
    league = safe_get(game, "league", "slug", default="mlb")
    
    home = safe_get(game, "homeTeam", default={})
    away = safe_get(game, "awayTeam", default={})
    home_name = safe_get(home, "displayName", default="")
    away_name = safe_get(away, "displayName", default="")
    
    # Scores / outcome
    home_score = safe_get(home, "score", default=None)
    away_score = safe_get(away, "score", default=None)
    home_win = None
    if home_score is not None and away_score is not None:
        try:
            home_win = 1 if int(home_score) > int(away_score) else 0
        except (ValueError, TypeError):
            pass
    
    # Records
    home_w, home_l, home_d = extract_record(home)
    away_w, away_l, away_d = extract_record(away)
    home_hw, home_hl = extract_home_away_record(home, True)
    away_aw, away_al = extract_home_away_record(away, False)
    
    # Form
    home_form = safe_get(enrichment, "homeLastFive", default={})
    away_form = safe_get(enrichment, "awayLastFive", default={})
    home_form_pts, home_form_str = parse_last_five(safe_get(home_form, "games", default=[]))
    away_form_pts, away_form_str = parse_last_five(safe_get(away_form, "games", default=[]))
    
    # Pitchers
    home_pitcher = extract_pitcher_stats(safe_get(game, "homePitcher", default={}))
    away_pitcher = extract_pitcher_stats(safe_get(game, "awayPitcher", default={}))
    
    # Advanced
    home_adv = extract_advanced(safe_get(enrichment, "homeAdvanced", default={}), "home_")
    away_adv = extract_advanced(safe_get(enrichment, "awayAdvanced", default={}), "away_")
    
    # Lineup
    home_lineup_ops = extract_lineup_ops(safe_get(enrichment, "homeLineup", default={}))
    away_lineup_ops = extract_lineup_ops(safe_get(enrichment, "awayLineup", default={}))
    
    # Injuries
    home_inj = extract_injury_count(safe_get(enrichment, "homeMajorInjuries", default=[]), home_name)
    away_inj = extract_injury_count(safe_get(enrichment, "awayMajorInjuries", default=[]), away_name)
    
    # Rest
    home_rest = extract_rest_days(safe_get(enrichment, "restDays", default={}), home_name)
    away_rest = extract_rest_days(safe_get(enrichment, "restDays", default={}), away_name)
    
    # Schedule flags
    home_flags = extract_schedule_flags(safe_get(enrichment, "homeScheduleFlags", default={}), home_name)
    away_flags = extract_schedule_flags(safe_get(enrichment, "awayScheduleFlags", default={}), away_name)
    
    # Weather
    weather = extract_weather(safe_get(enrichment, "weather", default={}))
    
    # H2H
    h2h = extract_h2h(safe_get(enrichment, "headToHead", default={}), home_name, away_name)
    
    # Odds
    odds_data = extract_odds(odds)
    
    # ESPN predictor
    espn_home = safe_get(enrichment, "espnPredictorHome")
    espn_away = safe_get(enrichment, "espnPredictorAway")
    
    # Build feature object
    f = GameFeatures(
        event_id=event_id,
        date=date_str[:10] if date_str else "",
        league=league,
        season=int(date_str[:4]) if date_str else 2024,
        home_team=home_name,
        away_team=away_name,
        home_win=home_win,
        home_score=int(home_score) if home_score else None,
        away_score=int(away_score) if away_score else None,
        home_wins=home_w,
        home_losses=home_l,
        home_draws=home_d,
        away_wins=away_w,
        away_losses=away_l,
        away_draws=away_d,
        home_home_wins=home_hw,
        home_home_losses=home_hl,
        away_away_wins=away_aw,
        away_away_losses=away_al,
        home_form_pts=home_form_pts,
        away_form_pts=away_form_pts,
        home_form_results=home_form_str,
        away_form_results=away_form_str,
        home_pitcher_era=home_pitcher["era"],
        away_pitcher_era=away_pitcher["era"],
        home_pitcher_fip=home_pitcher["fip"],
        away_pitcher_fip=away_pitcher["fip"],
        home_pitcher_recent_era=home_pitcher["recent_era"],
        away_pitcher_recent_era=away_pitcher["recent_era"],
        **home_adv,
        **away_adv,
        home_rest_days=home_rest,
        away_rest_days=away_rest,
        home_major_injuries=home_inj,
        away_major_injuries=away_inj,
        home_lineup_ops=home_lineup_ops,
        away_lineup_ops=away_lineup_ops,
        **odds_data,
        espn_home_pct=espn_home,
        espn_away_pct=espn_away,
        home_back_to_back=home_flags["back_to_back"],
        away_back_to_back=away_flags["back_to_back"],
        home_long_road_trip=home_flags["long_road_trip"],
        away_long_road_trip=away_flags["long_road_trip"],
        temperature=weather["temperature"],
        wind_speed=weather["wind_speed"],
        is_dome=weather["is_dome"],
        **h2h,
    )
    
    # Compute derived features
    f.home_win_pct = home_w / max(1, home_w + home_l + home_d)
    f.away_win_pct = away_w / max(1, away_w + away_l + away_d)
    f.home_home_win_pct = home_hw / max(1, home_hw + home_hl)
    f.away_away_win_pct = away_aw / max(1, away_aw + away_al)
    f.form_diff = home_form_pts - away_form_pts
    f.era_diff = (f.home_pitcher_era or 0) - (f.away_pitcher_era or 0)
    f.power_diff = (f.home_power_rating or 0) - (f.away_power_rating or 0)
    f.ops_diff = (f.home_ops or 0) - (f.away_ops or 0)
    f.run_diff_diff = (f.home_run_diff or 0) - (f.away_run_diff or 0)
    f.rest_diff = (f.home_rest_days or 0) - (f.away_rest_days or 0)
    f.injury_diff = f.home_major_injuries - f.away_major_injuries
    f.lineup_ops_diff = (f.home_lineup_ops or 0) - (f.away_lineup_ops or 0)
    
    if f.market_home_pct is not None:
        f.market_edge_home = f.home_win_pct - f.market_home_pct
    
    return f


def main():
    print("Extracting features from fixtures...")
    
    all_features = []
    
    # Process each fixture file
    for fixture_file in FIXTURE_DIR.glob("*.json"):
        if fixture_file.name.startswith("."):
            continue
            
        print(f"  Processing {fixture_file.name}...")
        
        try:
            with open(fixture_file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"    Error loading {fixture_file}: {e}")
            continue
        
        # Different fixture structures
        games = []
        if "events" in data:  # ESPN scoreboard
            games = data["events"]
        elif "pageProps" in data:  # Next.js page
            page_props = data["pageProps"]
            if "games" in page_props:
                games = page_props["games"]
            elif "matchup" in page_props:
                games = [page_props["matchup"]]
        
        for game in games:
            enrichment = safe_get(game, "enrichment")
            odds = safe_get(game, "odds") or safe_get(game, "sbrOdds")
            
            try:
                features = process_game(game, enrichment, odds)
                # Only keep games with known outcome for training
                if features.home_win is not None:
                    all_features.append(asdict(features))
            except Exception as e:
                print(f"    Error processing game: {e}")
    
    if not all_features:
        print("No labeled games found!")
        return
    
    df = pd.DataFrame(all_features)
    print(f"\nExtracted {len(df)} labeled games")
    print(f"Features: {len(df.columns)}")
    print(f"Leagues: {df['league'].unique()}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Home win rate: {df['home_win'].mean():.3f}")
    
    # Save
    output_file = OUTPUT_DIR / "training_features.parquet"
    df.to_parquet(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    # Also save as CSV for inspection
    csv_file = OUTPUT_DIR / "training_features.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved to {csv_file}")


if __name__ == "__main__":
    main()