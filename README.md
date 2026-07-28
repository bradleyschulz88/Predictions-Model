# Sports Predictions Dashboard

Multi-sport win predictions for **MLB**, **NFL**, **NBA**, **World Cup**, **EPL**, and **AFL** using ESPN schedule data.

Features include model vs market edge, totals picks, injury-adjusted predictions, accuracy tracking, mobile-friendly UI, and PWA support.

No third-party dependencies are required. Uses Python 3.10+.

## Live site (GitHub Pages)

**URL:** https://bradleyschulz88.github.io/Predictions-Model/

Data is refreshed **hourly** by GitHub Actions. To rebuild immediately:

1. Open [Actions → Publish GitHub Pages dashboard](https://github.com/bradleyschulz88/Predictions-Model/actions/workflows/pages.yml)
2. Click **Run workflow**

### One-time setup (if Pages is not live yet)

1. Repo **Settings → Pages**
2. **Build and deployment → Source:** GitHub Actions
3. Push to `main` or run the workflow manually

## Setup

```bash
python -m unittest discover -s tests -q
```

Tests are hermetic — they never reach the network and run in about a second.

## Dashboard

The dashboard supports **MLB**, **World Cup**, and **AFL** with win predictions, reasoning, lineups, and odds.

### GitHub Pages (phone / anywhere)

Use the live URL above — no PC required.

### Local (Windows)

Double-click:

- `start-dashboard.bat` — live ESPN schedule for tomorrow (recommended)
- `start-dashboard-offline.bat` — offline sample with 15 games for 2026-06-16

Your browser should open automatically. If it does not, copy the URL printed in the terminal (usually `http://127.0.0.1:8765/`).

**Important:** keep the terminal/command window open. Closing it stops the dashboard.

### Manual start

```bash
cd c:\Users\bradl\Desktop\MLB

# Tomorrow's games from ESPN
python mlb_sbr.py dashboard --source espn --insecure

# Offline sample schedule
python mlb_sbr.py dashboard --fixture tests/fixtures/espn_scoreboard_20260616.json --no-odds

# Schedule-only, no odds merge
python mlb_sbr.py dashboard --source espn --no-odds --insecure
```

The dashboard defaults to **tomorrow's date** and lists all scheduled MLB games with time, venue, broadcast, and records. Each game includes a **win prediction** ranked from most likely to least likely.

Each game shows a **Why they win** section with bullet-point reasoning and data sources.

## The model

Win probabilities come from a logistic model whose coefficients are **fitted from
graded outcomes**, not hand-tuned. Two inputs survive walk-forward feature
selection:

| feature | what it is |
|---|---|
| `strengthDiff` | one team-quality score collapsed from season record, home/road splits and power rating |
| `marketLogit` | de-vigged consensus moneyline, when odds exist |

Out-of-sample walk-forward on 675 graded games:

| forecaster | log loss | Brier |
|---|---|---|
| **fitted model** | **0.6428** | **0.2272** |
| market (de-vigged) | 0.6563 | 0.2336 |
| constant home base rate | 0.6931 | 0.2500 |

Everything else the pipeline gathers — starting pitching, bullpen ERA, rest,
injuries, back-to-back, schedule fatigue, weather, head-to-head — is still shown
as reasoning but is **not allowed to move the probability**, because none of it
improved out-of-sample accuracy at this sample size. Starting pitching matters in
baseball, but the market has already priced it. Re-run the ablation as the log
grows; a feature ships when it beats its own absence, not before.

Raw accuracy is ~60%, and that is close to the ceiling: published MLB
binary-prediction accuracy tops out around 55-60%. The value is in probabilities
that mean what they say, not in a higher hit rate.

### Model commands

```bash
# Fit coefficients from graded outcomes and write docs/data/model_weights.json
python model_fit.py

# Walk-forward score of each nested feature set
python model_fit.py --ablate

# Score the model against market and naive baselines
python scripts/backtest_model.py --evaluate

# Fail if the model regressed against docs/data/model_baseline.json
python scripts/check_regression.py
```

Predictions fall back to a hand-tuned heuristic when no fitted weights exist;
`prediction.probabilityMethod` records which path produced each number.

## CLI

All commands live in `mlb_sbr.py`:

```bash
# List today's games and sportsbooks
python mlb_sbr.py list-odds --date 2026-06-15

# JSON output
python mlb_sbr.py list-odds --date 2026-06-15 --format json

# Sample spread/moneyline lines for first 2 games
python mlb_sbr.py inspect-odds --date 2026-06-15

# Unique viewType values per game
python mlb_sbr.py viewtypes --date 2026-06-15

# Matchups listing page
python mlb_sbr.py matchups

# Single matchup detail
python mlb_sbr.py matchup --id 368835
python mlb_sbr.py matchup-keys --id 368835
python mlb_sbr.py spreads --id 368835

# Save live __NEXT_DATA__ for offline work
python mlb_sbr.py dump --url "https://www.sportsbookreview.com/betting-odds/mlb-baseball/?date=2026-06-15" --output tests/fixtures/live_odds.json
```

### Offline fixtures

Pass `--fixture path/to.json` to any command that reads page data. Fixtures can be full `__NEXT_DATA__` objects or bare `pageProps` objects.

### Network options

- `--retries 3` — HTTP retry attempts (default: 3)
- `--delay 1.0` — base delay between retries in seconds
- `--insecure` — skip TLS certificate verification if your network blocks SBR

## Library

`sbr_client.py` exposes reusable helpers:

- `get_text(url)` — fetch HTML with retries
- `fetch_next_data(url)` — parse `__NEXT_DATA__`
- `get_page_props(url)` — return `pageProps`
- `get_game_rows(page_props)` — odds listing rows
- `get_matchup(page_props)` — single matchup object
- `build_odds_url(date)`, `build_matchup_url(id)`, `build_matchups_url()`

`espn_client.py` fetches tomorrow's MLB schedule from ESPN's public scoreboard API.

`mlb_data.py` builds dashboard-ready payloads and optionally merges SportsBookReview odds.

## Notes

- Respect SportsBookReview's terms of service when fetching live pages.
- Saved fixtures under `tests/fixtures/` support offline development and unit tests.
