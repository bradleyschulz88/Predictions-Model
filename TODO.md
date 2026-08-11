# Open items — 2026-08-11

Nothing is broken. CI is green, 1,004 tests pass, the regression gate is inside
tolerance, and the 02:06Z build deployed in 4m25s. What follows is what is
waiting on a decision or on more data, not a defect list.

Figures are from the 02:10Z `evaluation.json`. Sample sizes are quoted because
most of them are still small enough to change the reading.

---

## The Odds API key works. Measured, not assumed.

Confirmed on the 02:06Z build of 2026-08-11, the first with the key in place:

- **MLB runlines price.** 15/15 on 11 Aug, 15/15 on 12 Aug, 10/10 on 10 Aug,
  7/9 on 13 Aug, via DraftKings, FanDuel and BetMGM. This is the market ESPN
  publishes with no juice attached, so all 62 graded runlines had been
  unvaluable; new ones are not.
- **AFL prices.** 1/1 on 14 Aug via SportsBet, TAB and Betfair — the first AFL
  price this project has ever had from any source. Games further out returned
  nothing, which is books not having posted yet rather than a failure.

One caveat on the older slates: 1/15 priced on 7–9 August. Those are games
already played, so the API no longer carries them. Nothing to fix.

## Blocked on you: one secret

At *Settings → Secrets and variables → Actions*.

**`NVIDIA_API_KEY`** — set, but with a space or line break inside it, so it
fails to authenticate and injury severity falls back to its deterministic
score. The build says so explicitly now: *"the key is set but the LLM step
scored 0 of 24 teams"*. Re-paste it as a single line.

## Two things I got wrong, now fixed

**The quota line never printed.** `odds_api` declared a `_QUOTA` dict, exposed
`quota_status()`, and the build printed `Odds API quota: {...}` whenever it came
back non-empty — but nothing ever wrote to it, because `get_text` returns a body
and drops the response that carries the headers. So the credit balance was
invisible from the day the provider shipped, and the 11 Aug build spent credits
on 7 slates with no record of what was left. `get_text_with_headers` now exists
and the balance is read off `x-requests-remaining`. **The next build is the
first that will print a real number** — that is the one to check the ~320/month
estimate against.

**The deploy timeout fix did nothing.** `timeout: 1200000` shipped as "twenty
minutes of headroom" and `actions/deploy-pages` clamps the input at 600000ms:
every run since has logged *"timeout value is greater than the allowed maximum"*
and waited exactly the ten minutes that failed on 6 Aug. The value is now the
real maximum, with a second attempt behind it — the only headroom actually
available — and `tests/test_deploy_timeout.py` fails the build if anyone raises
it again.

## Watch, do not act yet

**Totals are the weakest market on the board.** 43-37-4 overall, but **48.6%
±6.0 on the 70 priced picks**, which is the figure the 52.4% break-even applies
to, and −6.8% priced ROI. Below break-even, not conclusively so. The blended
53.8% is not evidence of anything: it mixes priced and unpriced picks, which is
what made this market look like it was winning and losing money at once.

**Spreads read well and rest on almost nothing.** 69.2% priced — on **13
picks**, ±12.8. The lower bound is under 45%. Now that the key is in and
runlines price, this sample grows by roughly a slate a day and the number will
move; do not read it until it has.

**CLV has just started reporting.** 3 confirmed picks, +1.05% average, against
97 still provisional. Too few to read. This is the metric that will actually
say whether the model is profitable long-run, so it is the one worth waiting
for — a fortnight of games should make it meaningful.

## Done since the last list

Items 2 through 6 are closed: the totals reporting split, the ESPN side-market
pass (totals went 71/81 → 85/86 priced), the divergence fix, `TEAM_HOME` for
NBA/NFL/WNBA, and the probe promoted to a real check. Plus MLB runline pricing
via The Odds API, and three silent-corruption bugs found by a full audit.

Model, current pipeline (02:10Z, n=261 against the market): median gap **3.7pts**
with **1.9%** of games over 15, against an original target of under 8. Walk-forward
logloss 0.6395, ahead of the de-vigged market by 0.0087 on the 580 priced games.
Reliability on the last 282 graded picks: predicted 63.6% against 64.2% actual.

The one significant home bias left is **World Cup**: picks home 77.1% where home
wins 54.3%, +22.9 ±8.5pts on n=35. Every other league is inside noise. Worth a
look, but the sample is small and the competition is idle.

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
