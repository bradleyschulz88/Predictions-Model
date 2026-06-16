# Sports Predictions Dashboard

Multi-sport win predictions for **MLB**, **NFL**, **NBA**, **World Cup**, **EPL**, and **AFL** using ESPN schedule data.

Features include model vs market edge, totals picks, injury-adjusted predictions, accuracy tracking, mobile-friendly UI, and PWA support.

No third-party dependencies are required for the core Python pipeline. Uses Python 3.10+.

## Cloud accounts & bet sync (Phase 1)

Deploy on **Vercel** with **Supabase** to let users sign in and sync **My Bets** across devices.

See **[docs/SAAS_SETUP.md](docs/SAAS_SETUP.md)** for Supabase migration, Vercel env vars, and testing steps.

GitHub Pages continues to work without accounts (bets stay in the browser).

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
cd MLB
python -m unittest discover -s tests -v
```

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

Predictions combine:
- Season and home/road records (ESPN)
- Probable starting pitchers and ERA (ESPN)
- ESPN Matchup Predictor, last-five form, head-to-head series, injuries, and weather (ESPN summary API)
- Home-field advantage
- Moneyline odds when available (SportsBookReview)

Each game shows a **Why they win** section with bullet-point reasoning and data sources.

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
