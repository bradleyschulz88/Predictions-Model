# Tomorrow — 2026-08-06

State at close of 2026-08-05: build 783 green and deployed, 894 tests passing,
zero coverage warnings, 3407 rows in the screening set.

Every number below is from `docs/data/accuracy.json` and `docs/data/evaluation.json`
as of the 14:21Z build. Sample sizes are quoted because most of them are small
enough to matter.

---

## 1. Two API keys — yours, five minutes, unblocks two things

**`NVIDIA_API_KEY`** has a space or line break inside it, so it fails to
authenticate and injury severity silently falls back to its deterministic
score. Re-paste it as a single line at *Settings → Secrets and variables →
Actions*.

**`ODDS_API_KEY`** appears to be unset. AFL has 33 graded picks and **0 priced**,
so its 78.8% hit rate carries `units: 0.0` — no price behind it, no EV, no
ranking against the rest of the board. The Odds API integration is wired and
tested; it just needs the key. Free tier is enough for one league.

## 2. Totals are winning and losing money at the same time

41-36-4, **53.2% ±5.7** — above the 52.4% break-even — yet **ROI −7.2%**
(−8.3% on the priced subset). A win rate over break-even with negative ROI
means the wins are coming at short prices and the losses at long ones.

n=77 decided, so this is not conclusive on its own. But the *sign* disagreement
between win rate and ROI is the kind of thing that does not fix itself, and it
is worth an hour: pull the 77 graded totals, sort by price taken, and see
whether the losses cluster at the long end. If they do, the totals model is
mis-ranking within its own market rather than being wrong about direction.

Compare against spreads, which do not have this problem: 52-29, 64.2% ±5.3,
+31.6% ROI on the priced subset.

## 3. Spreads are the best market on the board and the least priced

**13 of 81** spread picks carry a price. The other 68 cannot be valued, EV'd or
ranked. Given spreads are the only side market currently beating break-even
(64.2%, +5.1% ROI overall), closing that coverage gap is probably the highest-
value data work available.

Where the gap is: check whether the ESPN core odds pass is finding spread
markets it is currently dropping, before reaching for another source.

## 4. The model still diverges from the market far more than intended

Median gap **19.2 points**; **57.5%** of games diverge by more than 15. The
original target when this work started was a median under 8.

The fade-the-market subset is now 116 picks at **52.6%** against a 52.4%
break-even — statistically indistinguishable from zero edge, on a sample large
enough to say so. That is 116 picks a year of pure variance. Either the anchor
weight needs to go up, or the divergent picks need a gate that keeps them off
the board.

Live walk-forward, for reference: log loss 0.6394, Brier 0.2256, accuracy
61.6% over 665 games. The market's own de-vigged log loss on the overlapping
590 is 0.6521, the published model's is 0.6798 — so the model is still behind
the market out of sample on log loss while ahead on accuracy.

## 5. Fill in `TEAM_HOME` for the non-baseball leagues

`TEAM_HOME` holds 30 MLB clubs, which is why the travel tile is flagged
baseball-only and hidden on every other sport. Travel matters *more* in
basketball than in baseball — back-to-backs across time zones are a real NBA
effect — so this is a feature being withheld from the league it would help
most. NBA tips off in late October; there is time.

## 6. Housekeeping, when convenient

- **Delete or promote `.github/workflows/espn-probe.yml`.** Its header says
  "temporary diagnostic, delete once the cause is known." The cause is known
  (Akamai rejects browser-claiming UAs; the honest project UA works, 5/5 today).
  Either delete it or rewrite the header to say it is now a standing check.
- **`MIN_GAMES_BEFORE_USABLE`** is a game count, so NFL forfeits eleven weeks
  of every season to it where NBA forfeits three — 122 rows kept of ~280. Only
  worth scaling to season length if NFL ever needs to carry a conclusion alone.

## 7. Nothing to do, just check

**CLV** reads 0 confirmed / 97 provisional. It needs about a week of games that
start with a frozen closing price before it says anything. First real reading
should land around 2026-08-12. This is the metric that will actually tell you
whether the model is profitable long-run, so it is worth the wait.
