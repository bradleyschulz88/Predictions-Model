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

## Nothing is blocked on a secret any more

**`NVIDIA_API_KEY` is gone.** The injury-importance scorer that needed it has
been removed, so no key is required and the secret can be deleted from the
repository settings. The ablation is the reason: `injuryDiff` and
`injurySeverityDiff` both made walk-forward log loss worse at every sample size
measured — 0.6438 rising to 0.6459 as they went in. Carrying a metered external
dependency for a feature the data kept declining was not a trade worth making.

The deterministic injury score stays. It costs nothing and is what the board
has in practice always shown, since the key was never successfully configured.

One reference remains, in `youtube_intel.py` — an offline tool run by hand,
never in CI, whose `videoIntelDiff` feature has zero coverage in the graded log.
It is orphaned rather than active. Worth deleting; say the word.

## Two things I got wrong, now fixed

**The quota line never printed.** `odds_api` declared a `_QUOTA` dict, exposed
`quota_status()`, and the build printed `Odds API quota: {...}` whenever it came
back non-empty — but nothing ever wrote to it, because `get_text` returns a body
and drops the response that carries the headers. So the credit balance was
invisible from the day the provider shipped, and the 11 Aug build spent credits
on 7 slates with no record of what was left. `get_text_with_headers` now exists
and the balance is read off `x-requests-remaining`.

Reading the headers turned out to be necessary and not sufficient. Six builds
sampled across 11 Aug — 03:01, 05:19, 07:50, 09:11 and 23:53Z — were **all pure
cache hits**, each opening with *"restored 2 cached Odds API slates"*. A cache
hit makes no call, so it reads no headers, so it can report nothing. Correct
behaviour that added up to exactly the invisibility the reading was meant to
end.

The balance now rides in the cache file next to the slates and is stamped with
when it was taken, so every build prints it:

```
Odds API quota: {"remaining": 463, "used": 37, "lastCallCost": 3} (read 4.2h ago)
```

Unknown is stated as unknown rather than shown as zero. **Still no measured
number** — the first build after this lands is the one to read, and any figure
it shows will be carried rather than live until a fetch happens.

That the cache absorbs most builds is the budget mechanism working, not a
problem: it is the difference between ~320 credits a month and the entire free
tier gone inside a week.

**The deploy timeout fix did nothing.** `timeout: 1200000` shipped as "twenty
minutes of headroom" and `actions/deploy-pages` clamps the input at 600000ms:
every run since had logged *"timeout value is greater than the allowed maximum"*
and waited exactly the ten minutes that failed on 6 Aug. The value is now the
real maximum, with a second attempt behind it — the only headroom actually
available — and `tests/test_deploy_timeout.py` fails the build if anyone raises
it again. Confirmed on the 03:01Z build: no warning, deploy in 6 seconds, retry
step correctly skipped.

## NFL preseason: a one-game sample wearing a power rating

Found 13 Aug when the preseason started. The model was publishing **92.3%** on
Colts @ Patriots where the market implied **41.2%**, and **95.0%** — the
`MAX_PROB` clamp — on Panthers @ Cardinals against 45.5%.

| league | n | median gap | >15pts |
|---|---:|---:|---:|
| **NFL** | 10 | **27.2 pts** | **70%** |
| MLB | 658 | 8.2 pts | 25% |
| WNBA | 50 | 4.5 pts | 6% |
| AFL | 6 | 5.6 pts | 0% |

The original review opened on median 25 pts with 77% over 15. That was fixed —
*in baseball*. `recordDiff` is None in August, so strength fell through to
`powerRating` built on one or two preseason games: `homePower 0.80` against
`awayPower 0.00` is one club that won its opener against one that lost.
`MIN_GAMES_BEFORE_USABLE` guards the fit; nothing guarded the feature.

Three fixes, all general rather than NFL-specific, because the same defect
arrives in MLB every April and the NBA every October:

- **`strengthGames`** — how many games the estimate rests on, from the thinner
  of the two records. Preseason reports zero, so exhibitions fall out of the
  same arithmetic with no second code path.
- **Shrinkage** at `n/(n+10)`, the same rule already used for league
  intercepts. One game carries 9%, twenty carries two thirds, a full season is
  untouched. Rows logged before the fix have no `strengthGames` and are left
  exactly as they were rather than having a sample size guessed for them.
- **A per-league divergence gate** in `check_regression.py`, failing the build
  above 15 points on 10+ picks. Every working league sits at 4.5–8.2.

Walk-forward after the change: 0.6382 against a 0.6359 baseline — within
tolerance and near-identical, which is what a correct shrinkage does to mature
records.

**Why it went unseen:** the published divergence figure is pooled and baseball
dominates it. It read 3.7 points while NFL sat at 27.2. That is the third time
this week a pooled statistic concealed a subgroup contradicting it — after CLV
hiding MLB behind WNBA, and the staking gate comparing a blended hit rate to a
priced break-even. The gate is the answer to the pattern, not just to this
instance.

## The leak, and what it actually cost

Features are now frozen at first pitch, exactly as the closing price is. Before
that, `accuracy_tracker` overwrote a row's `features` on every build against a
build that re-enriches dates already played, so anything read from a source
that updates on a result encoded it.

**I said last turn that published picks were unaffected. That was half right.**
The features the board predicts on were always clean — at pick time the game
has not happened. The *weights* were not. A fit trained on rows where
`strengthDiff` scored an AUC of 0.682 will over-weight it against the ~0.62 it
is really worth pre-game, and that weight is then applied to live games. The
leak degraded predictions, not only the reports.

Freezing stops new contamination. It cannot repair the **947 rows already
logged**, and no later care will — `ablation.json` now carries `frozenSamples`
as the countdown to a re-baseline. It reads 0 today.

This is also the most likely explanation of the contradiction that has been
running through everything: the backtest claims +0.0075 log loss over the
market while CLV says the model takes worse prices than the close 61% of the
time. A contaminated fit and an honest CLV look exactly like that.

## Phase 1 results

**B1 — coverage. Answered, and it corrects my own earlier diagnosis.** No
provider is broken. The 25–33% figures I reported were an artefact of averaging
across a log that starts 18 June while every one of those features was added
24–26 July. Measured since each first appeared:

| feature | since added | last 200 rows |
|---|---:|---:|
| eloEdge | 94% | 96% |
| travelDiff | 89% | 92% |
| h2hDiff | 89% | 86% |
| parkEdge | 75% | **99% of MLB rows** |
| bullpenDiff | 75% | **100% of MLB rows** |
| handednessDiff | 74% | **97% of MLB rows** |

The last three are baseball-only, and MLB is 72% of the recent slate — so their
"70%" was the league mix, not a gap. **This workstream was waiting, not work,
and the wait is already over.** It also means the corrected ablation is judging
these on near-full coverage, so its verdicts can be trusted.

**B3 — lineup-conditioned edge. Cannot be answered, and the reason matters.**
`hasLineup` is True on 1,023 of 1,042 rows. There is no variation to split on,
because the build re-enriches played dates and by then a lineup always exists —
the same overwrite that produced the h2h leak. Blocked on pinning features at
pick time, not on data.

**A3 — line shopping. Not measurable retroactively; now instrumented.** The
build already takes the best quote across books, but never recorded the
alternatives, so the value of shopping existed for microseconds inside a build
and was discarded. `priceSpread` now pins books, best, median, worst and
`gainPct` at pick time. The unit is implied-probability points, the same scale
as CLV — which runs at a median of −0.4, so a shopping gain of one point would
more than cover what the model loses to the close, mechanically. Needs a week
of games.

## Side markets are publish-only

Both are now gated out of staking, and the gate had two bugs pointing the same
way.

It compared `pct` — the hit rate over **every** graded pick — against a
break-even that only applies to the **priced** ones. Live on 13 Aug: spreads
read 57.1% blended and passed, while the priced record it would actually be
staked at was 52.3% against a 53.4% break-even. It was backing a market on the
strength of picks that carried no price.

It also compared on the point estimate alone. Totals cleared by **a tenth of a
point** — 52.3 against 52.2, on a standard error near four.

Now: priced hit rate, and it must clear the bar by at least one standard error.
Not a significance test, which would want roughly two — the weaker claim that
the record has room to spare rather than sitting on the line.

| market | priced | break-even | stakeable |
|---|---|---|---|
| totals | 52.3% ±4.1 | 52.2% | no |
| spreads | 52.3% ±5.3 | 53.4% | no |

Both stay ranked, priced and visible on the card with their EV — publish-only,
not hidden — and the gate reopens on its own if a record earns it.

## Odds API: run it to empty, deliberately

Decision taken to keep AFL on all three markets and spend the remaining budget
rather than cut back. At ~16 credits a day the 303 remaining run out around
**1 September**, roughly ten days before the period resets.

That degrades safely: calls fail, the failure is cached for six hours, and
prices stop appearing rather than anything breaking. The quota line prints
`remaining` every build, and the build now warns once it drops under 60 so the
last week is visible in advance rather than in hindsight.

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

**CLV is now readable, and it is negative.** This has moved out of "watch" and
is the most important number on the project.

Over all **90 confirmed closes**: the model beats the closing line **38.9%
±5.3** of the time, 95% interval **28.5–49.3**, which excludes 50. Median CLV
**−0.61%**. It concentrates in the league with the sample and the coverage:

| League | n | Beat close | Median CLV |
|---|---:|---:|---:|
| MLB | 65 | **35.4% ±6.2** | −0.82% |
| WNBA | 16 | 56.2% ±12.5 | +1.52% |
| AFL | 8 | 37.5% ±17.7 | −0.52% |

The measurement was audited before this was believed: sign, side-guarding,
freeze-at-first-pitch and provisional exclusion all check out, and no-movement
ties account for 3 picks in 90. The finding is real.

Two reporting defects were hiding it. The headline used the **mean** (−0.16%,
dragged toward zero by a few large favourable moves) rather than the median,
and over a **71-pick window** whose interval spans 50 rather than the full
record whose interval does not. Both fixed. `beatsCoinFlip` also had no
negative counterpart, so "provably bad" and "too thin to say" both rendered as
the same shrug — there is now a `worseThanCoinFlip` flag and the board says it.

A negative CLV means the closing implied probability is lower than the one
paid: the side drifts out after the model takes it. That is what buying early
into a price the market later corrects looks like, so the next question is
timing, not features. `openingOddsAt` and `priceHistory` now record when each
price was taken and how it moved, which is what makes that testable.

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
