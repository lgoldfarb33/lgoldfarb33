"""Power ratings.

Ratings are solved from posted spreads rather than from game results, because
posted spreads are the sharpest publicly available estimate of team strength and
because historical result data is unreachable from this environment.

The consequence is worth stating plainly: a market-derived rating cannot, by
itself, disagree with the market. All disagreement — and therefore all edge —
comes from the adjustment layer below, where the qualitative thread scores are
converted into explicit points of spread. That makes every qualitative judgment
falsifiable: it now has a number attached, and `calibrate.py` can eventually tell
you whether that number was any good.

Model
-----
For each game, with r_h and r_a the home and away ratings:

    home_margin = r_h - r_a + HFA * (0 if neutral site else 1)

Stacked over games this is a linear system X·beta = y, solved by ridge-regularized
least squares. The ridge penalty applies to team ratings only, never to HFA, and a
sum-to-zero row anchors the overall level (ratings are only identified up to a
constant otherwise).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sources import Game, HFA_DEFAULT

# --------------------------------------------------------------------------
# Adjustment coefficients: points of spread per unit of thread score.
#
# THESE ARE PRIORS, NOT FITTED VALUES. Nobody has ever measured what a "+2
# coaching fit" is worth in points, and the historical thread could not source a
# single new-coach base rate. They are documented guesses, chosen conservatively,
# and they exist so the qualitative layer produces falsifiable numbers instead of
# vibes. `calibrate.py` reports whether they are earning their keep; refit them
# once ~100 settled picks exist.
# --------------------------------------------------------------------------
ADJUSTMENT_COEFFICIENTS = {
    # personnel.net_personnel is already a composite of coaching fit and roster
    # continuity, so coaching_fit is NOT applied separately — that would double count.
    "net_personnel": 1.00,
    "sentiment": 0.75,
}

# Below this many observed games, a team's rating is not meaningfully identified
# and no pick should lean on it. With one game a team's rating is just that game's
# spread reflected off its opponent, shrunk toward zero by the ridge penalty.
MIN_GAMES_FOR_CONFIDENCE = 3

# Ridge exists for numerical conditioning, NOT to paper over sparse data — that is
# the games-observed gate's job. Validation against synthetic data with known true
# ratings showed ridge=3.0 shrinking the rating scale by 16% (fitted-vs-true slope
# 0.84), which compresses every projected margin and systematically understates
# edges on strong teams. At 0.1 the slope is 0.994 and the recovered residual SD
# matches the injected pricing noise. See tests/test_model.py::test_recovers_truth.
HFA_PLAUSIBLE_RANGE = (0.0, 6.0)   # home advantage in CFB is ~2-3 pts, never 18
RIDGE_DEFAULT = 0.1
RIDGE_SPARSE = 2.0            # when teams average < 3 games, stability beats fidelity
SPARSE_GAMES_PER_TEAM = 3.0


def suggest_ridge(games: list[Game]) -> float:
    """Pick a ridge penalty from how much data there is.

    Dense schedules are well conditioned and want almost no penalty. A handful of
    posted openers is nearly unidentified and needs stabilizing — though those
    teams get gated out of picks regardless.
    """
    usable = [g for g in games if g.home_margin is not None]
    if not usable:
        return RIDGE_DEFAULT
    teams = {t for g in usable for t in (g.home, g.away)}
    per_team = (2.0 * len(usable)) / max(1, len(teams))
    return RIDGE_SPARSE if per_team < SPARSE_GAMES_PER_TEAM else RIDGE_DEFAULT


@dataclass
class RatingSolution:
    ratings: dict[str, float]
    hfa: float
    games_observed: dict[str, int]
    std_errors: dict[str, float]
    n_games: int
    n_teams: int
    residual_sd: float
    ridge: float
    hfa_fitted: bool
    effective_dof: float = 0.0   # trace of the hat matrix; near n_teams = little shrinkage
    warnings: list[str] = field(default_factory=list)

    def is_identified(self, team: str) -> bool:
        """Whether this team has enough games to trust its rating."""
        return self.games_observed.get(team, 0) >= MIN_GAMES_FOR_CONFIDENCE

    def rating(self, team: str) -> float | None:
        return self.ratings.get(team)

    def expected_margin(self, home: str, away: str, neutral: bool = False) -> float | None:
        """Model's fair spread: points the home team should be favored by."""
        rh, ra = self.ratings.get(home), self.ratings.get(away)
        if rh is None or ra is None:
            return None
        return rh - ra + (0.0 if neutral else self.hfa)


def solve_ratings(
    games: list[Game],
    ridge: float | None = None,
    fit_hfa: bool = True,
    anchor_weight: float = 10.0,
) -> RatingSolution:
    """Solve team ratings from posted spreads.

    ridge=None    -> chosen from data density by suggest_ridge()
    anchor_weight -> strength of the sum-to-zero level constraint
    """
    if ridge is None:
        ridge = suggest_ridge(games)
    usable = [g for g in games if g.home_margin is not None]
    warnings: list[str] = []
    if not usable:
        raise ValueError("No games with a posted spread — cannot solve ratings")

    teams = sorted({t for g in usable for t in (g.home, g.away)})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # Fit HFA only if at least a few non-neutral games exist, otherwise it is
    # indistinguishable from the ratings themselves.
    non_neutral = sum(1 for g in usable if not g.neutral)
    if fit_hfa and non_neutral < 3:
        fit_hfa = False
        warnings.append(
            f"Only {non_neutral} non-neutral games; HFA held at {HFA_DEFAULT} rather than fitted"
        )
    ncols = n + (1 if fit_hfa else 0)

    X = np.zeros((len(usable), ncols))
    y = np.zeros(len(usable))
    games_observed = {t: 0 for t in teams}

    for row, g in enumerate(usable):
        X[row, idx[g.home]] = 1.0
        X[row, idx[g.away]] = -1.0
        if fit_hfa and not g.neutral:
            X[row, n] = 1.0
        margin = g.home_margin
        if not fit_hfa and not g.neutral:
            margin = margin - HFA_DEFAULT  # remove fixed HFA from the target
        y[row] = margin
        games_observed[g.home] += 1
        games_observed[g.away] += 1

    # Ridge penalty on ratings only; HFA is a real physical quantity, not noise
    # to be shrunk toward zero.
    P = np.eye(ncols)
    if fit_hfa:
        P[n, n] = 0.0

    # Sum-to-zero anchor: without it the whole rating vector can slide by any
    # constant and fit identically.
    anchor = np.zeros((1, ncols))
    anchor[0, :n] = 1.0
    Xa = np.vstack([X, anchor_weight * anchor])
    ya = np.concatenate([y, [0.0]])

    A = Xa.T @ Xa + ridge * P
    beta = np.linalg.solve(A, Xa.T @ ya)

    ratings = {t: float(beta[idx[t]]) for t in teams}
    hfa = float(beta[n]) if fit_hfa else HFA_DEFAULT

    # Sanity-check the fitted HFA. Home advantage in CFB is worth roughly 2-3
    # points and has never plausibly been outside this band. When the game set is
    # small and lopsided — a slate of home-favorite non-conference openers, say —
    # HFA absorbs the mismatch and fits to absurd values (18.7 on the 11 real
    # sourced games), which then contaminates every projected margin.
    if fit_hfa and not (HFA_PLAUSIBLE_RANGE[0] <= hfa <= HFA_PLAUSIBLE_RANGE[1]):
        warnings.append(
            f"Fitted HFA {hfa:.1f} is outside the plausible range "
            f"{HFA_PLAUSIBLE_RANGE} — the game set is too small or too lopsided to "
            f"identify it. Falling back to {HFA_DEFAULT}."
        )
        hfa = HFA_DEFAULT
        hfa_was_overridden = True
    else:
        hfa_was_overridden = False

    resid = X @ beta - y
    dof = max(len(usable) - 1, 1)
    residual_sd = float(np.sqrt(float(resid @ resid) / dof))

    # Approximate standard errors from the ridge covariance. With one game per
    # team these come out large, which is the honest answer.
    effective_dof = 0.0
    try:
        Ainv = np.linalg.inv(A)
        cov = Ainv * (residual_sd ** 2)
        std_errors = {t: float(np.sqrt(max(cov[idx[t], idx[t]], 0.0))) for t in teams}
        # Effective degrees of freedom = trace(X (X'X + λP)^-1 X'). Well below the
        # column count means the ridge is doing heavy shrinking.
        effective_dof = float(np.trace(X @ Ainv @ X.T))
    except np.linalg.LinAlgError:
        std_errors = {t: float("nan") for t in teams}
        warnings.append("Covariance inversion failed; standard errors unavailable")

    if effective_dof and effective_dof < 0.75 * ncols:
        warnings.append(
            f"Effective dof {effective_dof:.1f} vs {ncols} parameters — ridge={ridge} is "
            f"shrinking ratings substantially; margins and edges will be understated"
        )

    thin = [t for t in teams if games_observed[t] < MIN_GAMES_FOR_CONFIDENCE]
    if thin:
        warnings.append(
            f"{len(thin)}/{n} teams have fewer than {MIN_GAMES_FOR_CONFIDENCE} observed games; "
            f"their ratings are weakly identified and are gated out of picks"
        )

    return RatingSolution(
        ratings=ratings, hfa=hfa, games_observed=games_observed, std_errors=std_errors,
        n_games=len(usable), n_teams=n, residual_sd=residual_sd, ridge=ridge,
        hfa_fitted=fit_hfa and not hfa_was_overridden,
        effective_dof=effective_dof, warnings=warnings,
    )


@dataclass
class Adjustment:
    team: str
    source_thread: str
    raw_score: float
    coefficient: float
    points: float
    rationale: str = ""


def build_adjustments(
    personnel: dict | None = None,
    sentiment: dict | None = None,
    coefficients: dict[str, float] | None = None,
) -> dict[str, list[Adjustment]]:
    """Translate thread JSON into points-of-spread adjustments per team."""
    coef = coefficients or ADJUSTMENT_COEFFICIENTS
    out: dict[str, list[Adjustment]] = {}

    for row in (personnel or {}).get("team_scores", []):
        team, score = row.get("team"), row.get("net_personnel")
        if not team or score is None:
            continue
        c = coef["net_personnel"]
        out.setdefault(team, []).append(Adjustment(
            team=team, source_thread="personnel", raw_score=float(score),
            coefficient=c, points=float(score) * c,
            rationale=row.get("one_line_rationale", ""),
        ))

    for row in (sentiment or {}).get("teams", []):
        team, score = row.get("team"), row.get("score")
        if not team or score is None:
            continue
        c = coef["sentiment"]
        out.setdefault(team, []).append(Adjustment(
            team=team, source_thread="sentiment", raw_score=float(score),
            coefficient=c, points=float(score) * c,
            rationale=row.get("justification", ""),
        ))

    return out


def apply_adjustments(
    solution: RatingSolution,
    adjustments: dict[str, list[Adjustment]],
    cap: float = 4.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (adjusted_ratings, total_points_applied_per_team).

    `cap` bounds how far the qualitative layer may move a rating. Uncapped, a
    team scoring +2 on both threads would move 3.5 points off unfitted priors —
    more confidence than this evidence supports.
    """
    adjusted = dict(solution.ratings)
    applied: dict[str, float] = {}
    for team, adjs in adjustments.items():
        if team not in adjusted:
            continue  # no market rating for this team; nothing to adjust
        delta = sum(a.points for a in adjs)
        delta = max(-cap, min(cap, delta))
        adjusted[team] = adjusted[team] + delta
        applied[team] = delta
    return adjusted, applied
