"""Edge quantification and bet sizing.

This module is the fix for weakness #1. The old sheet said "lean under on Indiana
10.5" and every pick was implicitly the same size. Here a pick carries a model
probability, a de-vigged market probability, an expected value, and a Kelly
fraction — so picks can be ranked against each other and sized differently.

Vig removal matters more than people expect. A -110/-110 market implies 52.4%
on both sides, summing to 104.8%. Comparing a model probability against the raw
52.4% overstates your edge by roughly 2.4 points on every bet.
"""
from __future__ import annotations

from dataclasses import dataclass

# Never stake more than this fraction of bankroll on one bet regardless of what
# Kelly says. Kelly assumes your probability estimate is correct; ours rests on
# unfitted coefficients, so the cap is doing real work.
MAX_STAKE_FRACTION = 0.05

# Fraction of full Kelly to actually bet. Quarter Kelly is the common choice —
# it gives up a little growth for a large reduction in variance and in sensitivity
# to probability error.
KELLY_FRACTION = 0.25

# Below this edge, the pick is noise relative to our estimation error.
MIN_EDGE_TO_BET = 0.03


def american_to_decimal(odds: int) -> float:
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def american_to_implied(odds: int) -> float:
    """Raw implied probability, vig included."""
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def devig_two_way(over_odds: int, under_odds: int) -> tuple[float, float]:
    """Remove vig proportionally from a two-sided market.

    Proportional (a.k.a. multiplicative) de-vigging. It slightly overstates the
    favorite's true probability versus Shin or power methods, but it is
    transparent and standard.
    """
    po, pu = american_to_implied(over_odds), american_to_implied(under_odds)
    booksum = po + pu
    if booksum <= 0:
        raise ValueError("Degenerate market prices")
    return po / booksum, pu / booksum


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Full Kelly stake as a fraction of bankroll. Negative means no bet."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return (p * b - (1.0 - p)) / b


def expected_value(p: float, decimal_odds: float) -> float:
    """EV per 1 unit staked."""
    return p * (decimal_odds - 1.0) - (1.0 - p)


@dataclass
class EdgeResult:
    subject: str
    market: str
    side: str
    line: float
    price_american: int | None
    model_prob: float
    market_prob: float | None
    edge: float | None
    ev_per_unit: float | None
    kelly_full: float | None
    stake_fraction: float | None
    bettable: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "subject": self.subject, "market": self.market, "side": self.side,
            "line": self.line, "price_american": self.price_american,
            "model_prob": round(self.model_prob, 4),
            "market_prob": round(self.market_prob, 4) if self.market_prob is not None else None,
            "edge": round(self.edge, 4) if self.edge is not None else None,
            "ev_per_unit": round(self.ev_per_unit, 4) if self.ev_per_unit is not None else None,
            "kelly_full": round(self.kelly_full, 4) if self.kelly_full is not None else None,
            "stake_fraction": round(self.stake_fraction, 4) if self.stake_fraction is not None else None,
            "bettable": self.bettable, "reason": self.reason,
        }


def evaluate(
    subject: str,
    market: str,
    side: str,
    line: float,
    model_prob: float,
    price_american: int | None,
    opposite_price_american: int | None = None,
    kelly_fraction_used: float = KELLY_FRACTION,
    min_edge: float = MIN_EDGE_TO_BET,
) -> EdgeResult:
    """Score one candidate bet.

    When both sides are priced, the market probability is de-vigged. When only one
    side is priced, vig cannot be separated and the comparison is skipped rather
    than done wrong — the result comes back not bettable with the reason recorded.
    """
    if price_american is None:
        return EdgeResult(
            subject, market, side, line, None, model_prob, None, None, None, None, None,
            bettable=False, reason="no price sourced — edge cannot be computed",
        )

    dec = american_to_decimal(price_american)

    if opposite_price_american is not None:
        p_side, p_other = devig_two_way(price_american, opposite_price_american)
        market_prob = p_side
        devig_note = ""
    else:
        market_prob = american_to_implied(price_american)
        devig_note = "one-sided price: vig NOT removed, edge is overstated — "

    edge = model_prob - market_prob
    ev = expected_value(model_prob, dec)
    kf = kelly_fraction(model_prob, dec)
    stake = max(0.0, min(kf * kelly_fraction_used, MAX_STAKE_FRACTION))

    if opposite_price_american is None:
        bettable, reason = False, devig_note + "treat as indicative only"
    elif edge < min_edge:
        bettable, reason = False, f"edge {edge:.1%} below {min_edge:.0%} threshold"
    elif kf <= 0:
        bettable, reason = False, "negative Kelly — price does not compensate"
    else:
        bettable, reason = True, f"edge {edge:.1%}, quarter-Kelly stake {stake:.2%} of bankroll"

    return EdgeResult(
        subject, market, side, line, price_american, model_prob, market_prob,
        edge, ev, kf, stake, bettable, reason,
    )
