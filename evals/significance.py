"""The statistics every eval script reports a difference with.

A difference is claimed only when these support it: an exact sign test over the
cases the arms disagree on, and a paired bootstrap whose share of resamples at or
below zero says the same thing on every seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb
import random


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test. Ties carry no information and are dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(max(wins, losses), n + 1))
    return min(2 * tail / 2**n, 1.0)


def paired_sign_test(diffs: Sequence[float]) -> tuple[int, int, float]:
    """Wins, losses, and the sign test's p-value for per-case differences.

    Reported beside the bootstrap because it forces the honest denominator into
    view: a difference of +0.03 over 88 questions can mean seven disagreed.
    """
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    return wins, losses, sign_test(wins, losses)


def bootstrap_ci(
    diffs: Sequence[float], rounds: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """A 95% percentile interval for the mean difference, and the share at or below 0.

    The share is returned because the interval alone is treacherous here: the
    per-case difference takes few distinct values, so the 2.5th percentile lands
    on zero for some seeds and just above it for others. The share moves smoothly.
    """
    if not diffs:
        return (0.0, 0.0, 1.0)
    rng = random.Random(seed)
    values = list(diffs)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(rounds))
    at_or_below = sum(1 for mean in means if mean <= 0) / rounds
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)], at_or_below
