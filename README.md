# Edge Board

Model-vs-market edge for **MLB**, **NFL**, **NBA**, **WNBA**, **EPL** and **AFL**.

A daily board of every game, each priced against the market, ranked by expected
value. Built as a **benchmark to start from**, not a tipster: the probabilities
are calibrated and the record is public, including the parts that lose.

The repository is still named `Predictions-Model` so the existing Pages URL and
any installed PWAs keep working. Renaming it would change both.

No third-party dependencies. Python 3.10+, standard library only.

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

The dashboard supports **MLB**, **WNBA**, and **AFL** with win predictions, reasoning, lineups, and odds.

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

Scored on the **same slate** for every forecaster — the 505 graded games that
have market odds, so the market is not being judged on a different population:

| forecaster | log loss | Brier | AUC | accuracy |
|---|---|---|---|---|
| model (published) | 0.6878 | 0.2448 | 0.6470 | 59.0% |
| **market (de-vigged)** | **0.6532** | **0.2322** | 0.6460 | 57.8% |
| constant home base rate | 0.6931 | 0.2500 | 0.5000 | 50.5% |

**The market is better calibrated than the model, and that is the honest
reading.** The model ranks marginally better — a shade more accuracy, a shade
more AUC — but its probabilities are worse. Any earlier claim that this model
beats the market on log loss came from comparing a walk-forward score against a
market score computed on a different set of games, which is not a fair test.

Where the model does earn its place: on the 103 picks that fade the market it
wins 55.3%, against a 52.4% break-even. That is a thin, unproven edge on a small
sample, not a system.

Reliability is still the weak point. The 55-65% band predicts ~11 points high,
and the thin 85-90% bucket predicts 34.9 points high on n=17:

| bucket | predicted | actual | n |
|---|---|---|---|
| 55-60% | 57.9% | 46.7% | 120 |
| 60-65% | 62.6% | 50.8% | 130 |
| 65-70% | 67.5% | 62.3% | 122 |
| 70-75% | 72.6% | 68.3% | 139 |
| 75-80% | 76.4% | 72.4% | 134 |

The model also still picks home too often: 56.7% of picks against a 51.4% actual
home win rate, and +7.2 points of that bias is MLB.

Everything else the pipeline gathers — starting pitching, bullpen ERA, rest,
injuries, back-to-back, schedule fatigue, weather, head-to-head — is still shown
as reasoning but is **not allowed to move the probability**, because none of it
improved out-of-sample accuracy at this sample size. Starting pitching matters in
baseball, but the market has already priced it. Re-run the ablation as the log
grows; a feature ships when it beats its own absence, not before.

Raw accuracy is ~60%, and that is close to the ceiling: published MLB
binary-prediction accuracy tops out around 55-60%. The value is in probabilities
that mean what they say, not in a higher hit rate.

### Games that never happened get voided, not left pending

A pick can only be graded against a result. If the game is called off there will
never be one, so the pick is marked **`voided`** — never a win, never a loss, and
no longer pending.

Without that terminal state 12 picks sat at "pending" indefinitely, the oldest
from 18 June. Ten were rain-outs replayed later, **six of them as the second game
of a doubleheader the next day**, which is what a postponed MLB game normally
becomes. Grading was already refusing to score them — correctly — but then
treated "called off" exactly like "not finished yet".

Two routes to that dead end, so both are handled:

| what ESPN does | how it resolves |
|---|---|
| still lists the game on its original date, flagged postponed | voided on sight, with the reason recorded |
| drops the game from that date once it is rescheduled | aged out after `VOID_UNRESOLVED_AFTER_DAYS` (3) |

Two guards keep this from firing wrongly:

- **A failed fetch never voids anything.** Only dates actually read this run can
  age a pick out — ESPN being down is not evidence about a game.
- **A delay is not a cancellation.** `isDelayed` stays pending; only postponed,
  cancelled, voided or washed-out games are closed.

The makeup game is predicted and graded in its own right, so voiding the
original is also what stops one postponement being counted twice.

### Dates are league-local, everywhere

A slate is filed under the **league's own calendar day**, which is how ESPN and
the leagues label it. `schedule_dates.py` holds the timezone per league and the
dashboard mirrors it exactly, including the rule that a US slate before 10am
local still shows the previous day's games.

This matters more than it sounds, because **roughly half a normal MLB night
starts after midnight UTC** — a 10:10pm game in New York is 02:10 the next day
in UTC. Filing by UTC would scatter one night's slate across two dates.

Game times are therefore **rendered in the league's timezone**, with the
viewer's own time appended when it differs:

```
Tue, 16 June, 10:10 pm EDT · Wed 12:10 pm your time
```

Rendering only in the viewer's timezone — which the dashboard used to do,
weekday and day-of-month included — made every card contradict the header above
it. A viewer in Australia opening the "Today · Tue 16 June" slate saw all
fifteen cards stamped "Wed, Jun 17". Nothing was mis-scheduled; the times were
being quoted in a calendar the page never used.

Tests assert both halves: that every game in the MLB and AFL fixtures lands on
its filed date when read in league-local time, and that the MLB fixture really
does straddle UTC midnight — otherwise the first assertion would be vacuous.

### Doubleheaders are two games, not a duplicate

MLB doubleheaders appear on the board twice, and that is correct: ESPN issues
each game its own event id, both are predicted, and both grade separately. Nine
pairs are in the logged history and every one has two different final scores.

The board numbers them **Gm 1 / Gm 2**, ordered by start time. Before that they
collapsed to the same matchup line with the start time hidden inside the
collapsed card, so a real doubleheader read as a duplicate row.

ESPN issues the second game an event id from a much later block
(`401902545` alongside `401816308`), so id order is a sound tiebreak when a
start time is missing.

### Team abbreviations come from ESPN

`awayAbbr` / `homeAbbr` are captured from the scoreboard payload. They are never
derived from the team name — initials render "Atlanta Braves" as `AB` and
"Boston Red Sox" as `BRS`, and no rule gets `NYM`, `NYY`, `CHC` and `CWS` right
from the words alone. The dashboard keeps a fallback for records stored before
the field was captured, and it is word-count aware: two-word names take the
first three letters of the city (`ATL`, `PIT`, `CLE`), longer names keep
initials so `NYM` and `LAD` stay correct.

### Publish thresholds

A pick is shown once it clears **55%** confidence. No league currently carries
its own bar — and the story of why is worth keeping, because the obvious version
of that idea is wrong.

MLB briefly carried a 65% bar. The evidence looked strong: 164 priced graded
picks in the 55‑65 band hit 45.1% into prices implying 58‑62%, for −16.4% ROI,
stable across both halves of the history and on both home and away sides.

It was still wrong, because **a fixed cutoff was pinned to a distribution that
then moved underneath it.** Platt calibration was correcting the model's
overconfidence at the same time:

| graded MLB | n | mean stated | actual | gap |
|---|---|---|---|---|
| before 2026‑07‑24 | 432 | 67.6% | 55.1% | **+12.5 pts** |
| after 2026‑07‑24 | 71 | 59.8% | 59.2% | **+0.6 pts** |

MLB's median stated confidence fell from 65‑73 to 54‑61 — 9.6 points at the
median. A bar measured to exclude the bottom quartile ended up excluding the
middle: it withheld **90% of MLB games, and 100% on several days**, leaving an
empty board.

The band was never the problem. Overconfidence was, and calibration fixed it. A
stated 60 used to mean roughly 45, which loses money at any price; it now means
roughly 60, which clears the 52.4% break‑even at −110. Withholding that band
today would discard exactly the picks calibration had just repaired.

`MIN_PICK_CONFIDENCE_BY_LEAGUE` remains, empty, with tests proving it still
works. **If a league ever looks like it needs its own bar, first check whether
the distribution has moved** — the two are very hard to tell apart from band
statistics alone. Derive any new bar from the current distribution, and
re-derive it whenever the model is refit.

A coverage warning now fires when a league publishes under 25% of a slate of 8+
games, because nothing caught this the first time: every build stayed green, as
an empty board is not an error.

> **Measuring this correctly.** Join the current model's confidence from
> `predictions_log.json` to the graded outcome. Do **not** band on
> `accuracy.json`'s own `confidence` field — that is frozen at the value shown
> when the pick was graded, and 537 rows now disagree with the live model, so
> bands drawn on it group the wrong games.

**Withheld picks are still logged.** Publishing and logging are different
questions: the 55-65% band is exactly where the fit is most wrong, so censoring
it from training would entrench the very error the threshold exists to hide.
Every pick is written to `predictions_log.json` with a `published` flag.

Which side of that line each consumer sits on is deliberate:

| reads the record | filters on `published`? | why |
|---|---|---|
| `accuracy.json` summary / ROI / streak | **yes** | a record nobody could have bet is not a record |
| dashboard board, day tally, hydration | **yes** | it reads `picksByEventId` directly, around the board filter |
| `model_fit.py` | no | the withheld band is where the fit most needs correcting |
| `scripts/backtest_model.py` | no | calibration must see the errors it exists to fix |
| `scripts/evaluation.py` | no | emits log loss / Brier / AUC; filtering would bias them |
| `elo.py` | no | replays game outcomes, not picks |

Do not add a `published` filter to the bottom four — that would recreate the
censoring this split exists to prevent. Currently the model trains on 706 graded
games while the board reports 687.

`published` is reconciled against the log on every run, because graded rows in
`accuracy.json` are carried forward and never rebuilt. Without that pass, a
threshold change would apply only to picks made after it.

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

### Odds sources

Moneylines reach a game through three sources, in order, each only trying what
the previous one left unpriced:

| order | source | covers | notes |
|---|---|---|---|
| 1 | SportsBookReview | MLB, NFL, NBA, EPL | full board where it has one |
| 2 | ESPN game summary | opportunistic | already part of enrichment |
| 3 | **ESPN core API** (`espn_odds.py`) | **WNBA** (not AFL) | keyless, no quota |

The third exists because the first two leave a third of the model unpriced.
SBR returns WNBA games with spread and total but **never a moneyline**, which
the model cannot use -- `marketLogit` is built from moneyline alone. AFL has no
SBR board at all.

Measured on the live build of 2026-07-29:

- **WNBA is fixed.** 8 games priced via DraftKings that had never had a price.
  Coverage grows as games approach, because books post lines a day or two out --
  a game four days away legitimately has no market yet.
- **AFL is not.** ESPN returns no moneyline for any AFL game, so it stays
  unpriced and honestly reports "ROI is not measurable". A circuit breaker
  stops us re-asking after three consecutive empties, which self-heals if ESPN
  ever adds coverage.

ESPN needs no key and publishes no quota, and -- the part that matters most --
is the same origin as the schedule, so team names and event ids match by
construction. None of the fuzzy matching SBR needs applies.

It is called **only for games still lacking a moneyline**, so a normal MLB day
makes zero extra requests. Prediction sites (Consensus, TeamRankings,
numberFire) are skipped: folding a model's output back in as "the market" would
make the anchor circular.

### YouTube team news (`youtube_intel.py`)

Collects pre-game team news from the channels you subscribe to, extracts it into
one number per team, and offers it to the ablation as `videoIntelDiff`. It is
**not** in the live model and will not move a published probability unless it
beats its own absence out of sample.

**This runs on your machine, not in CI.** YouTube's official `captions.download`
only works for videos you own, and the unofficial transcript endpoint blocks
datacenter IP ranges — GitHub Actions runners are Azure, so in CI every
transcript comes back empty. The script says so explicitly if that happens.

```bash
# One-off: put credentials in a gitignored .env, never in the repo
export YOUTUBE_CLIENT_ID=...        # Google Cloud OAuth client (Desktop app)
export YOUTUBE_CLIENT_SECRET=...
export YOUTUBE_REFRESH_TOKEN=...    # one-time consent, scope youtube.readonly
export NVIDIA_API_KEY=nvapi-...     # optional; without it no news is extracted

python youtube_intel.py             # writes docs/data/video_intel.json
git add docs/data/video_intel.json && git commit -m "Refresh video intel"
```

Quota is a non-issue: `subscriptions.list`, `channels.list` and
`playlistItems.list` cost 1 unit each against a 10,000/day allowance, so a
hundred channels costs a few hundred units. `search.list` costs 100 a call and
is deliberately never used.

**Leakage.** A recap video published after the final whistle knows who won.
Every record stores the video's `publishedAt`, and `intel_edge()` refuses any
video published at or after the game start — the same discipline as Elo's
pre-game edge. That boundary is exclusive and has its own tests.

**Expect it to fail the ablation for priced leagues.** The model anchors to
`marketLogit`, and a de-vigged closing line already contains public information;
a preview show is public. AFL is the case worth watching, because it has no odds
source and so no market to anchor to.

### What is committed under `docs/data/`

Split by whether it can be recomputed:

| file | committed | why |
|---|---|---|
| `predictions_log.json` | yes | every pick ever published — irreplaceable |
| `accuracy.json` | yes | every graded result — irreplaceable |
| `model_baseline.json` | yes | the regression ratchet; resets if lost |
| `video_intel.json` | yes | CI cannot rebuild it — YouTube blocks datacenter IPs, so `youtube_intel.py` produces it locally |
| `model_weights.json` | **no** | rebuilt by `model_fit.py` each run |
| `evaluation.json` | **no** | rebuilt by `backtest_model.py --evaluate` |
| `calibration.json` | **no** | rebuilt by `backtest_model.py --write` |
| `elo_ratings.json` | **no** | replayed from `accuracy.json` by `elo.py` |
| `manifest.json` | **no** | rebuilt by `build_pages_data.py` each run |
| `{league}_{date}.json` | **no** | per-day snapshots, rebuilt for the current window |

The derivatives still reach the site — the Pages artifact uploads `docs/` from
disk, which gitignore does not affect. They are not committed because CI
rewrites them every 30 minutes, so any branch that also regenerates them
collided on generated output at every rebase.

A fresh clone has no weights and will fall back to the heuristic until you run
`python model_fit.py`. CI runs it before predicting and fails the build if it
produces nothing.

### Candidate features, and why none of them ship yet

Beyond `strengthDiff` and `marketLogit`, everything the pipeline gathers is a
**candidate**: logged on every game, offered to the walk-forward ablation, and
allowed to move a probability only once it beats its own absence out of sample.

| candidate | what it is | status |
|---|---|---|
| `pitchingDiff` | starter + bullpen ERA edge | tested, does not beat its absence |
| `restDiff` · `b2bDiff` | days off, back-to-backs | tested, does not beat its absence |
| `injuryDiff` · `injurySeverityDiff` | who is unavailable, and what it costs | tested, does not beat its absence |
| `eloDiff` | pre-game Elo gap | tested, does not beat its absence |
| `videoIntelDiff` | team news from subscribed YouTube channels | tested, does not beat its absence |
| `h2hDiff` | season series between the two clubs | no coverage yet |
| `parkDiff` | ballpark run index | no coverage yet |
| `travelDiff` | distance + body-clock shift on the visitors | no coverage yet |
| `handednessDiff` | southpaw asymmetry between the starters | no coverage yet |
| `bullpenDiff` | relief innings absorbed in the last three days | no coverage yet — see note |

> **`bullpenDiff` has an unverified dependency.** Separating relief innings from
> total innings needs the starters' share, and the exact field name in the MLB
> Stats API team game log could not be checked offline. If it is absent the
> feature returns nothing and prints a CI warning, rather than fabricating a
> figure — an earlier version fell back to "whole game minus a typical start",
> which returns the typical figure every time and scored an 18-inning day and a
> 9-inning day identically at 0.00. Check the build log for
> `::warning title=Bullpen workload::` after the first run.

"No coverage yet" is the expected state for anything recently added: the fit
reads features out of `predictions_log.json`, so a new one starts at zero
history and contributes nothing until games grade with it present. An absent
feature is standardised to the mean, so it cannot distort the fit while it
waits.

**The sample size is the binding constraint.** There are ~700 graded games, and
roughly 10-20 games per predictor is the honest limit, so this list is a queue
to be tested one at a time rather than a model to be fitted all at once. Re-run
`python model_fit.py --ablate` every few weeks and let it arbitrate.

Why so many candidates fail: the model anchors to `marketLogit`, and a de-vigged
closing line has already priced the starter, the rest and the injuries. A
feature only helps if it carries information the market has not absorbed, which
is why recent bullpen workload is the most promising of the new ones and
head-to-head the least.

### Optional: LLM injury weighting

`data_providers/injury_severity.py` scores how much an injury list costs a team
from availability (`60-Day-IL` vs `Day-To-Day`) and seriousness (`Surgery` vs
`Soreness`). The feed carries no position or role, so player importance is the
one thing it cannot infer. Setting an API key adds that step:

```bash
export NVIDIA_API_KEY=nvapi-...          # https://build.nvidia.com (free tier)
export NVIDIA_INJURY_MODEL=meta/llama-3.1-8b-instruct   # optional
```

In CI, add it at **Settings > Secrets and variables > Actions** as
`NVIDIA_API_KEY`. The workflow passes it to the build step only. Never commit a
key — put it in `.env` locally, which is gitignored.

It is off by default and degrades to the deterministic score on any failure, so
a missing key, a timeout, a 429 or an unparseable reply cannot break a build.
Requests are cached per team and sent at temperature 0, so the same slate scores
the same way across builds.

**On model choice.** The default is a small instruct model on purpose. The task
is "rate this player 0-3", not reasoning — a large reasoning model answers no
better, takes far longer, and exhausts the free tier in a fraction of the calls.

**On rate limits.** The free tier allows ~40 requests/minute across the key. A
full seven-league build scores ~80 teams, so calls are spaced 1.6s apart to stay
under it, adding roughly two minutes to the build. Without that throttle every
team after the first 40 would quietly fall back to the deterministic score,
which looks identical in the output.

**This is an open experiment, not a shipped feature.** `injurySeverityDiff` is a
candidate in `CANDIDATE_FEATURES`; it does not move any probability unless
`python model_fit.py --ablate` shows it beating its own absence out of sample.
It cannot be judged until a few weeks of games have graded with the data
present.

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
