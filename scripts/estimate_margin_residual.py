"""Estimate the residual margin SD that MARGIN_STD_DEV actually needs.

MARGIN_STD_DEV holds the spread of a game's final margin around THAT GAME's
expected margin. Two entries hold the wrong thing: wnba 13.49 and afl 40.05
were measured as the spread of margins across all games, which is larger
because it also carries the variation in team strength between matchups.

    Var(margin over all games) = Var(expected margin) + Var(residual)

The direct fix needs closing spreads alongside results, and neither ESPN's
historical scoreboard nor the graded log has enough of them. This takes the
other route, which needs no spreads at all. If margin ~ N(mu, sigma_r) and the
market's win probability p satisfies p = Phi(mu / sigma_r), then
mu = sigma_r * z with z = Phi^-1(p), so

    Var(margin) = sigma_r^2 * Var(z) + sigma_r^2
    sigma_r     = SD(margin) / sqrt(1 + Var(z))

Both inputs are measurable: SD(margin) from played seasons via
scripts/measure_margin_sd.py, and Var(z) from the market prices already in the
graded log.

Sanity check on a league where the answer is known: NBA measures SD 16.21
across 1059 games, and the residual that makes an 80% favourite a 9-to-10
point favourite is about 11.5. That implies Var(z) = (16.21/11.5)^2 - 1 = 0.99,
i.e. market probabilities spanning roughly 16%-84% at one standard deviation,
which is what NBA lines look like. The method is consistent with the entry
already known to be right.

    python scripts/estimate_margin_residual.py
"""

from __future__ import annotations

import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model_fit  # noqa: E402

# Raw final-margin SD measured over completed seasons by
# scripts/measure_margin_sd.py. Not the value MARGIN_STD_DEV holds -- the
# first term of the decomposition above.
RAW_MARGIN_SD = {
    "nba": (16.21, "2025-11-01..2026-03-31", 1059),
    "nfl": (14.18, "2025-09-04..2026-01-05", 272),
    "wnba": (15.16, "2025-05-16..2025-09-11", 288),
}

# Var(z) is a variance estimate, so it needs a real sample before it means
# anything. Below this the answer is reported as unavailable rather than
# guessed -- the whole point of this exercise was that a plausible number
# measured the wrong thing.
MIN_PRICED_GAMES = 100


def market_z_scores(data_dir: Path) -> dict[str, list[float]]:
    """Inverse-normal of the market's win probability, per league."""
    normal = NormalDist()
    by_league: dict[str, list[float]] = defaultdict(list)
    samples, _ = model_fit.samples_from_log(data_dir)
    for sample in samples:
        market = sample.values.get("marketLogit")
        if market in (None, 0.0):
            continue
        prob = min(max(model_fit.sigmoid(market), 0.001), 0.999)
        by_league[sample.league].append(normal.inv_cdf(prob))
    return dict(by_league)


def residual_sd(raw_margin_sd: float, z_scores: list[float]) -> float | None:
    """sigma_r from the raw margin SD and the spread of market probabilities."""
    if len(z_scores) < MIN_PRICED_GAMES:
        return None
    return raw_margin_sd / math.sqrt(1.0 + statistics.pvariance(z_scores))


def main() -> int:
    data_dir = ROOT / "docs" / "data"
    z_by_league = market_z_scores(data_dir)

    print("Residual margin SD, the quantity MARGIN_STD_DEV holds")
    print(f"  needs at least {MIN_PRICED_GAMES} priced graded games per league\n")
    print(f"  {'league':<8} {'priced':>7} {'Var(z)':>8} {'raw SD':>8} {'sigma_r':>9}")

    ready = False
    for league, (raw, window, n) in sorted(RAW_MARGIN_SD.items()):
        zs = z_by_league.get(league, [])
        sigma = residual_sd(raw, zs)
        if sigma is None:
            print(f"  {league:<8} {len(zs):7} {'--':>8} {raw:8.2f} {'not yet':>9}"
                  f"   (raw SD from {window}, n={n})")
            continue
        ready = True
        var_z = statistics.pvariance(zs)
        print(f"  {league:<8} {len(zs):7} {var_z:8.3f} {raw:8.2f} {sigma:9.2f}")

    print()
    for league in ("afl",):
        zs = z_by_league.get(league, [])
        print(f"  {league}: {len(zs)} priced graded games. AFL only gained a price source "
              f"recently, so this cannot be estimated until the log fills.")
    if not ready:
        print("\n  Nothing to promote yet. MARGIN_STD_DEV is unchanged and its comment"
              "\n  records which entries are known to hold the wrong statistic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
