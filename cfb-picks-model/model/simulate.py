"""Monte Carlo season simulation.

Turns ratings into a win distribution per team, which is the thing a posted win
total can actually be compared against. "Lean under on 10.5" becomes "projected
9.1 wins, P(under) = 0.68" — a number you can size a bet with.

Each game is simulated once and its outcome applied to both participants, so a
team's wins and its opponents' losses stay consistent within a simulated season.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .sources import MARGIN_SD_DEFAULT


@dataclass
class ScheduledGame:
    home: str
    away: str
    neutral: bool = False
    date: str = ""
    week: str = ""


@dataclass
class TeamProjection:
    team: str
    games: int
    mean_wins: float
    median_wins: float
    p10: float
    p90: float
    win_distribution: dict[int, float]  # wins -> probability
    unidentified_opponents: int = 0

    def prob_over(self, total: float) -> float:
        """P(season wins > total). Assumes a half-point total, so no push."""
        return float(sum(p for w, p in self.win_distribution.items() if w > total))

    def prob_under(self, total: float) -> float:
        return float(sum(p for w, p in self.win_distribution.items() if w < total))


@dataclass
class SimulationResult:
    projections: dict[str, TeamProjection]
    n_sims: int
    margin_sd: float
    n_games: int
    warnings: list[str] = field(default_factory=list)


def win_probability(margin: float, margin_sd: float = MARGIN_SD_DEFAULT) -> float:
    """P(win) given an expected point margin.

    Normal CDF of the margin over its standard deviation. CFB margins are not
    perfectly normal — there is mass at key numbers (3, 7) — but for season-level
    win totals that texture washes out.
    """
    return float(norm.cdf(margin / margin_sd))


def win_probability_vec(margins, margin_sd: float = MARGIN_SD_DEFAULT):
    """Vectorized win_probability over an array of margins."""
    return norm.cdf(np.asarray(margins) / margin_sd)


def simulate_season(
    ratings: dict[str, float],
    schedule: list[ScheduledGame],
    hfa: float,
    n_sims: int = 20000,
    margin_sd: float = MARGIN_SD_DEFAULT,
    seed: int = 20260807,
    rating_se: dict[str, float] | None = None,
) -> SimulationResult:
    """Simulate the season n_sims times and return per-team win distributions.

    `rating_se` propagates uncertainty in the ratings themselves. Without it the
    simulation treats each rating as exactly known, which understates the width of
    the win distribution and inflates edges — on synthetic data that alone
    produced 8 spurious "bettable" edges of 6-10% on correctly-priced teams.
    Ratings are redrawn once per simulated season from N(rating, se), so a season
    is internally consistent while the ensemble reflects what we do not know.
    """
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    teams = sorted({t for g in schedule for t in (g.home, g.away)})
    tidx = {t: i for i, t in enumerate(teams)}

    playable, missing = [], set()
    for g in schedule:
        if g.home in ratings and g.away in ratings:
            playable.append(g)
        else:
            missing.update(t for t in (g.home, g.away) if t not in ratings)
    if missing:
        warnings.append(
            f"{len(missing)} teams in the schedule have no rating and their games were "
            f"dropped: {', '.join(sorted(missing)[:8])}"
            + (" ..." if len(missing) > 8 else "")
        )
    if not playable:
        raise ValueError("No schedule games have ratings for both teams")

    home_i = np.array([tidx[g.home] for g in playable])
    away_i = np.array([tidx[g.away] for g in playable])
    hfa_vec = np.array([0.0 if g.neutral else hfa for g in playable])

    if rating_se:
        # Two sources of randomness: which team is actually better (rating
        # uncertainty, redrawn per season) and who wins on the day (margin noise).
        base = np.array([ratings[t] for t in teams])
        se = np.array([max(0.0, rating_se.get(t, 0.0)) for t in teams])
        drawn = base + rng.normal(0.0, 1.0, (n_sims, len(teams))) * se
        margins = drawn[:, home_i] - drawn[:, away_i] + hfa_vec
        p_home = norm.cdf(margins / margin_sd)          # (n_sims, n_games)
        if float(se.max()) > 0:
            warnings.append(
                f"Rating uncertainty propagated (max SE {se.max():.2f} pts); win "
                f"distributions are wider and edges smaller than a point-estimate run"
            )
    else:
        margins = np.array([ratings[g.home] - ratings[g.away] for g in playable]) + hfa_vec
        p_home = win_probability_vec(margins, margin_sd)  # (n_games,)
        warnings.append(
            "Ratings treated as exact — no standard errors supplied. Edges are "
            "optimistic; pass rating_se to propagate estimation uncertainty."
        )

    # One Bernoulli draw per game per simulation; the same draw settles both sides.
    draws = rng.random((n_sims, len(playable)))
    home_wins = draws < p_home  # (n_sims, n_games) bool

    wins = np.zeros((n_sims, len(teams)), dtype=np.int16)
    for i, g in enumerate(playable):
        hi, ai = tidx[g.home], tidx[g.away]
        col = home_wins[:, i]
        wins[:, hi] += col
        wins[:, ai] += ~col

    games_per_team = {t: 0 for t in teams}
    unident = {t: 0 for t in teams}
    for g in playable:
        games_per_team[g.home] += 1
        games_per_team[g.away] += 1
    for g in schedule:
        for t, opp in ((g.home, g.away), (g.away, g.home)):
            if t in games_per_team and opp not in ratings:
                unident[t] += 1

    projections = {}
    for t in teams:
        col = wins[:, tidx[t]]
        n_games_t = games_per_team[t]
        if n_games_t == 0:
            continue
        counts = np.bincount(col, minlength=n_games_t + 1).astype(float)
        dist = {int(w): float(c / n_sims) for w, c in enumerate(counts) if c > 0}
        projections[t] = TeamProjection(
            team=t,
            games=n_games_t,
            mean_wins=float(col.mean()),
            median_wins=float(np.median(col)),
            p10=float(np.percentile(col, 10)),
            p90=float(np.percentile(col, 90)),
            win_distribution=dist,
            unidentified_opponents=unident[t],
        )

    return SimulationResult(
        projections=projections, n_sims=n_sims, margin_sd=margin_sd,
        n_games=len(playable), warnings=warnings,
    )
