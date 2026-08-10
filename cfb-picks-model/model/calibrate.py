"""Calibration: is the model's confidence worth anything?

The other half of the fix for weakness #3. A model that says "70%" should be right
about 70% of the time. Until that is measured, "high confidence" is a label, not a
claim.

Three questions this answers:
  1. Are the probabilities calibrated?  (Brier score, calibration curve)
  2. Do the tiers mean anything?        (do "high" picks beat "medium"?)
  3. Are we beating the closing line?   (CLV — the fastest-converging signal)

It refuses to report conclusions on small samples. With 12 settled picks the
honest output is "not enough data", and saying so is the entire point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ledger import LedgerEntry

# Rough thresholds for how much a sample can support.
MIN_FOR_ANY_SIGNAL = 30
MIN_FOR_TIER_COMPARISON = 60
MIN_FOR_COEFFICIENT_REFIT = 100


@dataclass
class CalibrationReport:
    n_settled: int
    n_open: int
    brier: float | None = None
    brier_baseline: float | None = None      # always predicting the base rate
    skill_score: float | None = None         # 1 - brier/baseline; >0 beats naive
    hit_rate: float | None = None
    mean_predicted: float | None = None
    buckets: list[dict] = field(default_factory=list)
    by_tier: list[dict] = field(default_factory=list)
    clv: dict = field(default_factory=dict)
    roi: float | None = None
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_settled": self.n_settled, "n_open": self.n_open,
            "brier": _r(self.brier), "brier_baseline": _r(self.brier_baseline),
            "skill_score": _r(self.skill_score), "hit_rate": _r(self.hit_rate),
            "mean_predicted": _r(self.mean_predicted), "roi": _r(self.roi),
            "buckets": self.buckets, "by_tier": self.by_tier, "clv": self.clv,
            "verdict": self.verdict, "warnings": self.warnings,
        }


def _r(v, n=4):
    return round(v, n) if isinstance(v, (int, float)) else None


def _outcome(e: LedgerEntry) -> int:
    return 1 if e.result == "win" else 0


def _profit(e: LedgerEntry) -> float:
    """Profit per 1 unit staked."""
    if e.price_american is None:
        return 0.0
    from .edge import american_to_decimal
    if e.result == "win":
        return american_to_decimal(e.price_american) - 1.0
    if e.result == "loss":
        return -1.0
    return 0.0


def build_report(settled: list[LedgerEntry], n_open: int = 0,
                 bucket_edges=(0.5, 0.6, 0.7, 0.8, 0.9, 1.01)) -> CalibrationReport:
    rep = CalibrationReport(n_settled=len(settled), n_open=n_open)

    if not settled:
        rep.verdict = (
            "No settled picks yet. Nothing about this model's accuracy is known — "
            "treat every confidence label as an untested assertion."
        )
        rep.warnings.append(f"Need ~{MIN_FOR_ANY_SIGNAL} settled picks before any signal emerges")
        return rep

    probs = [e.model_prob for e in settled]
    outs = [_outcome(e) for e in settled]
    n = len(settled)

    rep.brier = sum((p - o) ** 2 for p, o in zip(probs, outs)) / n
    base = sum(outs) / n
    rep.brier_baseline = sum((base - o) ** 2 for o in outs) / n
    rep.skill_score = (1 - rep.brier / rep.brier_baseline) if rep.brier_baseline > 0 else None
    rep.hit_rate = base
    rep.mean_predicted = sum(probs) / n

    staked = [e for e in settled if e.price_american is not None]
    if staked:
        rep.roi = sum(_profit(e) for e in staked) / len(staked)

    # Calibration curve
    lo = 0.0
    for hi in bucket_edges:
        members = [(p, o) for p, o in zip(probs, outs) if lo <= p < hi]
        if members:
            rep.buckets.append({
                "range": f"{lo:.0%}-{hi:.0%}",
                "n": len(members),
                "mean_predicted": round(sum(p for p, _ in members) / len(members), 3),
                "actual_rate": round(sum(o for _, o in members) / len(members), 3),
            })
        lo = hi

    # Do the tiers separate?
    for tier in ("high", "medium", "speculative"):
        members = [e for e in settled if e.tier == tier]
        if members:
            rep.by_tier.append({
                "tier": tier, "n": len(members),
                "hit_rate": round(sum(_outcome(e) for e in members) / len(members), 3),
                "mean_predicted": round(sum(e.model_prob for e in members) / len(members), 3),
                "roi": round(
                    sum(_profit(e) for e in members if e.price_american is not None)
                    / max(1, len([e for e in members if e.price_american is not None])), 3),
            })

    clv_vals = [e.clv_prob for e in settled if e.clv_prob is not None]
    if clv_vals:
        beat = sum(1 for v in clv_vals if v > 0)
        rep.clv = {
            "n": len(clv_vals),
            "mean_clv_prob": round(sum(clv_vals) / len(clv_vals), 4),
            "pct_beating_close": round(beat / len(clv_vals), 3),
            "note": "Positive mean CLV is the strongest early evidence of genuine edge.",
        }
    else:
        rep.clv = {"n": 0, "note": "No closing lines recorded — settle picks with closing prices to enable CLV."}

    # Verdict, gated hard on sample size
    if n < MIN_FOR_ANY_SIGNAL:
        rep.verdict = (
            f"{n} settled picks is too few to conclude anything. Brier and hit rate are "
            f"reported for completeness but are dominated by variance at this sample size."
        )
        rep.warnings.append(f"Need ~{MIN_FOR_ANY_SIGNAL} settled picks for a first read")
    else:
        direction = "better than" if (rep.skill_score or 0) > 0 else "no better than"
        rep.verdict = (
            f"{n} settled picks. Brier {rep.brier:.3f} vs baseline {rep.brier_baseline:.3f} "
            f"— the model's probabilities are {direction} naively predicting the base rate."
        )
        if n < MIN_FOR_TIER_COMPARISON:
            rep.warnings.append(
                f"Tier comparison needs ~{MIN_FOR_TIER_COMPARISON} picks; treat by_tier as provisional"
            )
        if n < MIN_FOR_COEFFICIENT_REFIT:
            rep.warnings.append(
                f"Adjustment coefficients in ratings.py stay at their priors until "
                f"~{MIN_FOR_COEFFICIENT_REFIT} settled picks exist to fit them against"
            )

    if rep.mean_predicted is not None and rep.hit_rate is not None and n >= MIN_FOR_ANY_SIGNAL:
        gap = rep.mean_predicted - rep.hit_rate
        if abs(gap) > 0.08:
            rep.warnings.append(
                f"Mean predicted {rep.mean_predicted:.1%} vs actual {rep.hit_rate:.1%} — "
                f"the model is systematically {'over' if gap > 0 else 'under'}confident"
            )
    return rep
