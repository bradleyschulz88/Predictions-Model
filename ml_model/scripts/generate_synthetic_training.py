#!/usr/bin/env python3
"""
Generate synthetic training data using the existing simulation engine.
This leverages the existing match simulation to create labeled games with features.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matchEngine import (
    simMatchEvents, teamRating, competitiveOppRating
)
from playerGen import generateSquad
from leagueEngine import PYRAMID, findClub, competitionClubsForCareer, generateFixtures
from defaults import DEFAULT_FACILITIES, DEFAULT_TRAINING, generateStaff, defaultKits
from finance.engine import makeStartingFinance
from difficulty import getDifficultyConfig
from community import generateCommittee, generateJournalist
from lineupHelpers import LINEUP_CAP

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

N_GAMES = 5000  # Number of synthetic games to generate
SEASON = 2024


def create_synthetic_career(league_key: str = "AFL", club_id: str = "ade") -> dict:
    """Create a synthetic career object for simulation."""
    club = findClub(club_id)
    league = PYRAMID[league_key]
    _ = getDifficultyConfig("balanced")
    
    # Generate squads for all clubs in league
    comp_clubs = competitionClubsForCareer(
        type("Career", (), {"leagueKey": league_key, "clubId": club_id, "regionState": club.state, "season": SEASON})()
    )
    fixtures = generateFixtures(comp_clubs)
    
    squad_raw = generateSquad(club_id, league.tier, 32, SEASON)
    # Scale squad to fit cap
    tuned_finance = makeStartingFinance(league.tier, "balanced", 55)
    from lib.finance.engine import scaledSquadToFitCap
    squad = scaledSquadToFitCap({
        "clubId": club_id, "leagueKey": league_key, "difficulty": "balanced",
        "finance": tuned_finance, "squad": squad_raw
    })
    
    lineup = squad[:].sort(key=lambda p: p.get("overall", 0), reverse=True)[:LINEUP_CAP]
    lineup_ids = [p["id"] for p in lineup]
    
    career = {
        "managerName": "Synthetic",
        "clubId": club_id,
        "leagueKey": league_key,
        "regionState": club.state,
        "season": SEASON,
        "week": 0,
        "currentDate": f"{SEASON}-04-01",
        "phase": "season",
        "eventQueue": [],
        "lastEvent": None,
        "inMatchDay": False,
        "currentMatchResult": None,
        "squad": squad,
        "lineup": lineup_ids,
        "training": DEFAULT_TRAINING(),
        "facilities": DEFAULT_FACILITIES(),
        "finance": tuned_finance,
        "sponsors": [],
        "staff": generateStaff(league.tier),
        "staffTasks": {"recruitPriorityState": None, "matchPrepTier": 0, "trainingLeadId": None},
        "kits": defaultKits(club.colors),
        "ladder": [],
        "fixtures": fixtures,
        "tradePool": [],
        "draftPool": [],
        "youth": {"recruits": [], "zone": club.state, "programLevel": 1, "scoutFocus": "All-rounders"},
        "news": [],
        "weeklyHistory": [],
        "inFinals": False,
        "finalsRound": 0,
        "finalsFixtures": [],
        "finalsResults": [],
        "premiership": None,
        "tacticChoice": "balanced",
        "seasonHistory": [],
        "aiSquads": {},
        "draftOrder": [],
        "history": [],
        "brownlow": {},
        "boardWarning": 0,
        "gameOver": None,
        "themeMode": "A",
        "options": {"autosave": True, "confirmBeforeNewCareer": True, "confirmBeforeDeleteSlot": True, "uiDensity": "comfortable", "reduceMotion": False},
        "pendingTradeOffers": [],
        "inbox": [],
        "retiredThisSeason": [],
        "difficulty": "balanced",
        "gameMode": "standard",
        "challengeId": None,
        "challengeGoal": None,
        "tutorialStep": 6,
        "tutorialComplete": True,
        "isFirstCareer": False,
        "committee": generateCommittee(league.tier),
        "footyTripAvailable": False,
        "footyTripUsed": False,
        "groundCondition": 85,
        "clubGround": {"id": f"{club_id}_home", "name": f"{club.name} Stadium", "shortName": f"{club.name} Stadium", "capacity": 30000, "surface": "grass", "condition": 85},
        "groundName": f"{club.name} Stadium",
        "weeklyWeather": {},
        "winStreak": 0,
        "homeWinStreak": 0,
        "coachReputation": 30,
        "coachTier": "Journeyman",
        "coachStats": {"totalWins": 0, "totalLosses": 0, "totalDraws": 0, "premierships": 0, "promotions": 0, "relegations": 0, "clubsManaged": 1, "seasonsManaged": 1},
        "previousClubs": [],
        "isSacked": False,
        "jobMarketOpen": False,
        "sackingStep": None,
        "jobOffers": [],
        "boardVotePrepBonus": 0,
        "jobMarketRerolls": 0,
        "arrivalBriefing": None,
        "journalist": generateJournalist(),
        "lastBoardConfidenceDelta": 0,
        "lastMatchSummary": None,
        "lastFinanceTickWeek": None,
        "lastFinanceTickDay": None,
        "cashCrisisStartWeek": None,
        "cashCrisisLevel": 0,
        "bankLoan": None,
        "sponsorRenewalProposals": [],
        "sponsorOffers": [],
        "expiredSponsorsLastSeason": [],
        "pendingRenewals": [],
        "renewalsClosed": False,
        "pendingStaffRenewals": [],
        "fundraisersUsed": {},
        "communityGrantUsed": False,
        "lastEosFinance": None,
        "postSeasonPhase": "none",
        "inTradePeriod": False,
        "tradePeriodDay": 0,
        "freeAgencyOpen": False,
        "postSeasonDraftCountdown": None,
        "freeAgentBalance": {"gained": 0, "lost": 0},
        "tradeHistory": [],
        "draftPickBank": None,
        "offSeasonFreeAgents": [],
        "clubCulture": {"identity": "balanced", "fanExpectation": "finals", "loyalty": 50, "stability": 50, "ruthlessness": 50},
        "headToHead": {},
        "finalsRivalryLog": [],
        "captainId": None,
        "viceCaptainId": None,
        "captainHistory": [],
        "bogeyTeamId": None,
        "dominatedTeamId": None,
        "crucialFive": [],
        "crisisFiredThisSeason": False,
        "teamStats": None,
    }
    
    return career, comp_clubs, fixtures


def generate_synthetic_game(home_club_id: str, away_club_id: str, career: dict, league_key: str) -> dict:
    """Generate a single synthetic game with features and outcome."""
    league = PYRAMID[league_key]
    home_club = findClub(home_club_id)
    away_club = findClub(away_club_id)
    
    # Get or create AI squads
    home_squad = career.get("aiSquads", {}).get(home_club_id)
    away_squad = career.get("aiSquads", {}).get(away_club_id)
    
    if not home_squad:
        home_squad = generateSquad(home_club_id, league.tier, 32, SEASON)
        career.setdefault("aiSquads", {})[home_club_id] = home_squad
    if not away_squad:
        away_squad = generateSquad(away_club_id, league.tier, 32, SEASON)
        career.setdefault("aiSquads", {})[away_club_id] = away_squad
    
    # Get home/away ratings
    home_rating = teamRating(home_squad, career.get("lineup", []), career.get("training", DEFAULT_TRAINING()), 
                             career.get("facilities", DEFAULT_FACILITIES()), career.get("staff", []))
    away_rating = teamRating(away_squad, [], career.get("training", DEFAULT_TRAINING()), 
                             career.get("facilities", DEFAULT_FACILITIES()), career.get("staff", []))
    
    # Apply difficulty competitive adjustment
    away_rating = competitiveOppRating(away_rating, home_rating)
    
    # Simulate match
    result = simMatchEvents(
        {"rating": home_rating}, 
        {"rating": away_rating}, 
        True,  # home is player
        home_rating,
        {"homeFixtureAdvantage": 4}
    )
    
    # Determine outcome
    home_win = 1 if result["winner"] == "home" else 0
    
    # Extract features from simulation
    features = {
        "event_id": f"synth_{home_club_id}_vs_{away_club_id}_{datetime.now().timestamp()}",
        "date": f"2024-{(np.random.randint(1, 12)):02d}-{(np.random.randint(1, 28)):02d}",
        "league": league_key,
        "season": SEASON,
        "home_team": home_club.name,
        "away_team": away_club.name,
        "home_win": home_win,
        "home_score": result["homeGoals"],
        "away_score": result["awayGoals"],
        
        # Team ratings (core features)
        "home_rating": home_rating,
        "away_rating": away_rating,
        "rating_diff": home_rating - away_rating,
        
        # Score details
        "home_goals": result["homeGoals"],
        "away_goals": result["awayGoals"],
        "home_behinds": result["homeBehinds"],
        "away_behinds": result["awayBehinds"],
        
        # Quarter scores
        "q1_home": result["quarters"][0]["homeGoals"] if result["quarters"] else 0,
        "q1_away": result["quarters"][0]["awayGoals"] if result["quarters"] else 0,
        "q2_home": result["quarters"][1]["homeGoals"] if len(result["quarters"]) > 1 else 0,
        "q2_away": result["quarters"][1]["awayGoals"] if len(result["quarters"]) > 1 else 0,
        "q3_home": result["quarters"][2]["homeGoals"] if len(result["quarters"]) > 2 else 0,
        "q3_away": result["quarters"][2]["awayGoals"] if len(result["quarters"]) > 2 else 0,
        "q4_home": result["quarters"][3]["homeGoals"] if len(result["quarters"]) > 3 else 0,
        "q4_away": result["quarters"][3]["awayGoals"] if len(result["quarters"]) > 3 else 0,
        
        # Match dynamics
        "momentum": result.get("momentum", 0),
        "total_shots": len(result.get("events", [])),
        "home_shots": len([e for e in result.get("events", []) if e.get("side") == "home"]),
        "away_shots": len([e for e in result.get("events", []) if e.get("side") == "away"]),
        
        # Key moments
        "key_moments": len(result.get("keyMoments", [])),
        "injuries": len(result.get("injuredPlayerIds", [])),
        
        # Squad composition features
        "home_squad_avg": np.mean([p.get("overall", 60) for p in career.get("aiSquads", {}).get(home_club_id, [])]) if career.get("aiSquads", {}).get(home_club_id) else 60,
        "away_squad_avg": np.mean([p.get("overall", 60) for p in career.get("aiSquads", {}).get(away_club_id, [])]) if career.get("aiSquads", {}).get(away_club_id) else 60,
    }
    
    return features


def main():
    print(f"Generating {N_GAMES} synthetic games...")
    
    # Create careers for different leagues
    careers = {}
    leagues_to_use = ["AFL", "VFL", "SANFL", "WAFL"]  # leagues with clubs
    
    for league_key in leagues_to_use:
        if league_key not in PYRAMID:
            continue
        league = PYRAMID[league_key]
        if not league.clubs:
            continue
        club_id = league.clubs[0].id
        print(f"  Creating career for {league_key}/{club_id}...")
        career, clubs, fixtures = create_synthetic_career(league_key, club_id)
        careers[league_key] = (career, clubs)
    
    # Generate games
    all_features = []
    
    for i in range(N_GAMES):
        if i % 500 == 0:
            print(f"  Generated {i}/{N_GAMES} games...")
        
        # Pick random league and clubs
        league_key = np.random.choice(list(careers.keys()))
        career, clubs = careers[league_key]
        
        if len(clubs) < 2:
            continue
            
        home_club = np.random.choice(clubs)
        away_club = np.random.choice([c for c in clubs if c.id != home_club.id])
        
        try:
            features = generate_synthetic_game(home_club.id, away_club.id, career, league_key)
            all_features.append(features)
        except Exception:
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Add some noise and additional features
    print(f"\nGenerated {len(df)} games")
    print(f"Home win rate: {df['home_win'].mean():.3f}")
    
    # Save
    output_file = OUTPUT_DIR / "synthetic_training_features.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Saved to {output_file}")
    
    csv_file = OUTPUT_DIR / "synthetic_training_features.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved to {csv_file}")
    
    # Print stats
    print(f"\nFeature columns: {len(df.columns)}")
    print(f"Features: {list(df.columns)}")
    
    # Correlation with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if "home_win" in df.columns:
        corr = df[numeric_cols].corr()["home_win"].sort_values(ascending=False)
        print("\nTop correlations with home_win:")
        print(corr.head(15))


if __name__ == "__main__":
    main()