"""Measure a league's final-margin standard deviation from played games.

READ THIS BEFORE USING THE NUMBER IT PRINTS. What this measures -- the spread
of final margins across all games -- is NOT what MARGIN_STD_DEV holds, and
substituting it there makes the model worse.

MARGIN_STD_DEV needs the spread of a game's margin around THAT GAME's expected
margin, a residual. The across-all-games figure is larger because it also
carries the game-to-game variation in team strength:

    Var(margin over all games) = Var(expected margin) + Var(residual)

This was learned the hard way: NBA measures 16.21 here, and putting that in
MARGIN_STD_DEV turns an 80% favourite into a 13.6-point favourite where the
market says about 9. The existing 11.5 is the residual and is right.

So this script is for the first term, and is useful for sanity-checking a league
or spotting a value that is wildly off -- not for setting MARGIN_STD_DEV. Getting
the residual needs closing spreads alongside results, which ESPN's historical
scoreboard does not carry.

Seasons that have already been played are available from ESPN's scoreboard, so
there is no need to wait for tip-off to measure this. The dev sandbox's egress
policy blocks espn.com, so run it from a GitHub runner rather than locally:

    python scripts/measure_margin_sd.py nba 2025-11-01 2026-03-31

MEASURED 2026-08-05, most recent complete season of each league. The question
was whether NBA and NFL -- which have no graded games and therefore no league
intercept of their own -- are being handed a home-field figure fitted on
baseball, whose home edge is the weakest of the four.

    league   n     home win rate   implied logit
    wnba      288  55.6% +/-2.9    +0.2231
    nba      1059  55.0% +/-1.5    +0.1990
    nfl       271  53.9% +/-3.0    +0.1553
    mlb      1603  52.8% +/-1.2    +0.1137

They are not, and the premise was wrong twice over.

First, the gaps are not measurable at these sample sizes. NBA minus MLB is
2.2 points against a standard error of sqrt(1.5^2 + 1.2^2) = 1.9 -- 1.15 sigma.
NFL minus MLB is 1.1 against 3.2. Nothing here separates the leagues.

Second, and larger: the fitted intercept is not a home-field term. The market
anchor already carries home advantage, so model_fit's intercept only holds the
residual bias the market misses -- which is why every fitted league intercept
is tiny (mlb -0.0362, wnba +0.0992) next to the raw logits above. A league at
zero games inherits the global intercept, +0.0367, worth 0.9 points at even
money, and that is *larger* than what MLB ends up with (+0.0367 - 0.0362 =
+0.0005). Whatever the cold start costs NBA, it does not cost it home-field.

So this needs no fix, and re-running the four leagues on every build would burn
~500 requests to re-derive the table above. Run it by hand once a season has
finished if you want to check the figures have not moved.
"""

from __future__ import annotations

import datetime
import json
import math
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sports_config import get_league  # noqa: E402

USER_AGENT = "EdgeBoard/1.0 (+https://github.com/bradleyschulz88/Predictions-Model)"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard?dates={date}"


def _iter_dates(start: str, end: str):
    day = datetime.date.fromisoformat(start)
    last = datetime.date.fromisoformat(end)
    while day <= last:
        yield day
        day += datetime.timedelta(days=1)


def _final_margins(payload: dict) -> list[tuple[float, float | None]]:
    """Return (home margin, home spread) for each completed game on a slate."""
    out: list[tuple[float, float | None]] = []
    for event in payload.get("events") or []:
        for competition in event.get("competitions") or []:
            status = ((competition.get("status") or {}).get("type") or {})
            if not status.get("completed"):
                continue
            home = away = None
            for competitor in competition.get("competitors") or []:
                score = competitor.get("score")
                try:
                    value = float(score)
                except (TypeError, ValueError):
                    continue
                if competitor.get("homeAway") == "home":
                    home = value
                elif competitor.get("homeAway") == "away":
                    away = value
            if home is None or away is None:
                continue

            # ESPN quotes the spread from the favourite's side, so read the
            # home team's own handicap rather than the headline number.
            spread = None
            for odds in competition.get("odds") or []:
                home_odds = odds.get("homeTeamOdds") or {}
                for key in ("spreadOdds", "pointSpread", "spread"):
                    raw = home_odds.get(key)
                    if isinstance(raw, dict):
                        raw = raw.get("value") or raw.get("displayValue")
                    try:
                        spread = float(raw)
                        break
                    except (TypeError, ValueError):
                        continue
                if spread is None:
                    try:
                        spread = float(odds.get("spread"))
                    except (TypeError, ValueError):
                        spread = None
                if spread is not None:
                    break
            out.append((home - away, spread))
    return out


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    league, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    path = get_league(league).espn_path

    margins: list[float] = []
    residuals: list[float] = []
    days = failures = 0
    for day in _iter_dates(start, end):
        url = SCOREBOARD.format(path=path, date=day.strftime("%Y%m%d"))
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, json.JSONDecodeError):
            failures += 1
            continue
        days += 1
        for margin, spread in _final_margins(payload):
            margins.append(margin)
            if spread is not None:
                # spread is the home handicap, so margin + spread is the
                # amount by which the home side beat its own line.
                residuals.append(margin + spread)

    print(f"{league}: {start} to {end}")
    print(f"  slates fetched {days}, failed {failures}")
    if len(margins) < 2:
        print("  not enough completed games to measure")
        return 1

    # Home win rate, which is the other thing a league with no graded games
    # cannot supply. fit_from_observations shrinks a league's own home-field
    # estimate by count/(count+50), so a league at zero games gets exactly 0.0
    # and silently inherits an intercept fitted on the leagues that do have
    # data -- almost all of it baseball, which has the weakest home edge of
    # any of them.
    home_wins = sum(1 for margin in margins if margin > 0)
    decided = sum(1 for margin in margins if margin != 0)
    if decided:
        rate = home_wins / decided * 100
        std_err = math.sqrt(0.25 / decided) * 100
        # Intercept in logit space, which is the unit model_fit stores.
        implied = math.log((rate / 100) / (1 - rate / 100))
        print(f"  home win rate      n={decided:5}  {rate:5.1f}% +/-{std_err:.1f}"
              f"   implied intercept {implied:+.4f}")

    print(f"  raw final margin   n={len(margins):5}  "
          f"mean {statistics.fmean(margins):+6.2f}  SD {statistics.stdev(margins):6.2f}")
    if len(residuals) >= 2:
        print(f"  margin vs spread   n={len(residuals):5}  "
              f"mean {statistics.fmean(residuals):+6.2f}  SD {statistics.stdev(residuals):6.2f}")
    else:
        print("  margin vs spread   no spreads on these slates")
    print()
    print("  NOT the MARGIN_STD_DEV value. That needs the residual -- the second")
    print("  line where a priced source supplies spreads -- not the first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
