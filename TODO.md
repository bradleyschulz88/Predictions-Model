# Open items — 2026-08-06

Nothing is broken. CI is green, 972 tests pass, the regression gate is inside
tolerance, and the last nine builds all succeeded. What follows is what is
waiting on a decision or on more data, not a defect list.

Figures are from the 00:16Z `accuracy.json` and a freshly regenerated
`evaluation.json`. Sample sizes are quoted because most of them are still small
enough to change the reading.

---

## Blocked on you: two secrets

Both live at *Settings → Secrets and variables → Actions*. Nothing else in this
file needs anything from you.

**`ODDS_API_KEY`** — not set. This is the one with real consequences now that
the runline work has landed:

- AFL: **0 of 33** graded picks priced, so its 78.8% hit rate still carries
  `units: 0.0` and cannot be ranked against anything.
- MLB runlines: every one still unpriced. ESPN publishes the handicap with no
  juice attached, so The Odds API is the only source that closes this.

Free tier at the-odds-api.com is 500 credits a month. The estimate is ~320 for
both leagues — MLB asks for `spreads` alone at 1 credit a call, AFL needs all
three at 3, and the cache now persists across builds. Worth checking the quota
line the build prints after the first full day: that number is reasoned from
the call pattern, not yet measured against a live key.

**`NVIDIA_API_KEY`** — set, but with a space or line break inside it, so it
fails to authenticate and injury severity falls back to its deterministic
score. Re-paste it as a single line.

## Watch, do not act yet

**Totals are the weakest market on the board.** 43-37-4 overall, but **48.6%
±6.0 on the 70 priced picks**, which is the figure the 52.4% break-even applies
to, and −6.8% priced ROI. Below break-even, not conclusively so. The blended
53.8% is not evidence of anything: it mixes priced and unpriced picks, which is
what made this market look like it was winning and losing money at once.

**Spreads read well and rest on almost nothing.** 69.2% priced — on **13
picks**, ±12.8. The lower bound is under 45%. Once the Odds API key lands and
runlines start pricing, this sample grows quickly and the number will move.

**CLV has just started reporting.** 3 confirmed picks, +1.05% average, against
97 still provisional. Too few to read. This is the metric that will actually
say whether the model is profitable long-run, so it is the one worth waiting
for — a fortnight of games should make it meaningful.

## Done since the last list

Items 2 through 6 are closed: the totals reporting split, the ESPN side-market
pass (totals went 71/81 → 85/86 priced), the divergence fix, `TEAM_HOME` for
NBA/NFL/WNBA, and the probe promoted to a real check. Plus MLB runline pricing
via The Odds API, and three silent-corruption bugs found by a full audit.

Model, current pipeline: median gap to market **3.3pts** with **0.0%** of games
over 15, against an original target of under 8. All-time 61.2% on 802 graded
picks, +2.0% ROI.

## Not worth doing yet

- **NFL backfill depth.** `MIN_GAMES_BEFORE_USABLE = 10` is a game count, so a
  weekly sport forfeits eleven weeks a season to it: NFL kept 122 rows of ~280.
  Only worth scaling to season length if NFL ever needs to carry a conclusion
  on its own. It is 122 of 3407 rows.
- **The 30-minute cadence.** GitHub's scheduler delivers roughly one build
  every two hours regardless of the cron, and moving off the congested minutes
  changed nothing measurable. The board now reports its own age rather than
  promising a cadence. A real 30 minutes needs an external trigger calling
  `workflow_dispatch`, which is infrastructure this project does not have.
