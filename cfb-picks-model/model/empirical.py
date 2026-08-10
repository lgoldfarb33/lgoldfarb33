"""Opponent-adjusted ratings fit from actual game results.

This is the module that lets the model disagree with the market.

`ratings.py` solves ratings from posted spreads, which makes them sharp but
circular — a market-derived rating can never, on its own, say the market is wrong.
Here ratings are fit from realized scoring margins instead. The two are built from
genuinely different information, so the *gap between them* is a signal in its own
right: it is precisely where the market has repriced a team away from what it
actually did on the field.

Data comes from sportsdataverse/cfbfastR-data over raw.githubusercontent.com — no
API key, and reachable from this environment when the odds APIs are not.

Constants measured here (2022-2025, FBS vs FBS only, 3,132 games) replace the
values the engine previously assumed:

    home advantage    3.91 pts measured   (was assumed 2.5)
    residual margin SD  16.94 measured    (was assumed 16.5 — close)

The residual SD is the honest one to use: raw margin SD is ~24.9, but most of that
is teams being unequal, not game-to-game randomness. Using raw SD would flatten
every win probability toward 50% and erase real edges.
"""
from __future__ import annotations

import csv
import json
import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np

DATA_BASE = ("https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/"
             "main/schedules/csv/cfb_schedules_{year}.csv")

# The eleven FBS conferences. Filtering on this matters more than it looks: without
# it the fit picks up ~530 FCS/DII teams, HFA inflates and residual SD jumps to 18.4
# on the strength of FBS-over-FCS blowouts that no one can bet.
FBS_CONFERENCES = {
    "SEC", "Big Ten", "Big 12", "ACC", "American Athletic", "Conference USA",
    "Mid-American", "Mountain West", "Sun Belt", "Pac-12", "FBS Independents",
}

# Measured, not assumed. See module docstring.
MEASURED_HFA = 3.91
MEASURED_MARGIN_SD = 16.94

# Ridge for the results fit. Heavier than the spread fit because a 12-game season
# is a small sample per team and unregularized margins overfit blowouts.
RESULTS_RIDGE = 25.0

# How much a season's results count toward the prior. Rosters turn over hard in the
# portal era, so last season dominates and older seasons decay fast.
SEASON_DECAY = 0.55


@dataclass
class ResultGame:
    season: int
    home: str
    away: str
    margin: int
    neutral: bool
    week: str = ""


@dataclass
class EmpiricalFit:
    ratings: dict[str, float]
    hfa: float
    residual_sd: float
    n_games: int
    n_teams: int
    games_observed: dict[str, int]
    seasons: list[int]
    std_errors: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "seasons": self.seasons, "n_games": self.n_games, "n_teams": self.n_teams,
            "hfa": round(self.hfa, 3), "residual_sd": round(self.residual_sd, 3),
            "ratings": {t: round(v, 3) for t, v in
                        sorted(self.ratings.items(), key=lambda kv: -kv[1])},
            "games_observed": self.games_observed,
            "std_errors": {t: round(v, 3) for t, v in self.std_errors.items()},
        }


def download_seasons(years, out_dir: str, timeout: int = 40) -> dict[int, str]:
    """Fetch schedule+result CSVs. Returns {year: path} for what actually landed."""
    os.makedirs(out_dir, exist_ok=True)
    got = {}
    for y in years:
        path = os.path.join(out_dir, f"sched_{y}.csv")
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            got[y] = path
            continue
        try:
            with urllib.request.urlopen(DATA_BASE.format(year=y), timeout=timeout) as resp:
                body = resp.read()
            if len(body) > 10_000:
                with open(path, "wb") as fh:
                    fh.write(body)
                got[y] = path
        except Exception:  # noqa: BLE001 — a missing season is normal, not fatal
            continue
    return got


def load_results(paths, fbs_only: bool = True, completed_only: bool = True) -> list[ResultGame]:
    games: list[ResultGame] = []
    for path in paths:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if completed_only and r.get("completed") != "TRUE":
                    continue
                try:
                    hp, ap = int(r["home_points"]), int(r["away_points"])
                except (ValueError, TypeError, KeyError):
                    continue
                if fbs_only and (r.get("home_conference") not in FBS_CONFERENCES
                                 or r.get("away_conference") not in FBS_CONFERENCES):
                    continue
                games.append(ResultGame(
                    season=int(r["season"]), home=r["home_team"], away=r["away_team"],
                    margin=hp - ap, neutral=r.get("neutral_site") == "TRUE",
                    week=str(r.get("week", "")),
                ))
    return games


def load_schedule(paths, season: int, fbs_only: bool = True) -> list[ResultGame]:
    """Upcoming (or any) games for one season, played or not — the simulator's input."""
    out = []
    for path in paths:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if int(r.get("season", 0)) != season:
                    continue
                if fbs_only and (r.get("home_conference") not in FBS_CONFERENCES
                                 or r.get("away_conference") not in FBS_CONFERENCES):
                    continue
                out.append(ResultGame(
                    season=season, home=r["home_team"], away=r["away_team"], margin=0,
                    neutral=r.get("neutral_site") == "TRUE", week=str(r.get("week", "")),
                ))
    return out


def fit_from_results(
    games: list[ResultGame],
    ridge: float = RESULTS_RIDGE,
    decay: float = SEASON_DECAY,
) -> EmpiricalFit:
    """Opponent-adjusted ratings by weighted ridge regression on scoring margin.

    Recent seasons carry more weight (`decay` per season back). Weighting rather
    than truncating keeps older games contributing signal where a team has little
    recent data, without letting a 2022 roster dictate a 2026 rating.
    """
    if not games:
        raise ValueError("No result games supplied")

    teams = sorted({t for g in games for t in (g.home, g.away)})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    newest = max(g.season for g in games)

    X = np.zeros((len(games), n + 1))
    y = np.zeros(len(games))
    w = np.zeros(len(games))
    observed = {t: 0 for t in teams}

    for i, g in enumerate(games):
        X[i, idx[g.home]] = 1.0
        X[i, idx[g.away]] = -1.0
        if not g.neutral:
            X[i, n] = 1.0
        y[i] = g.margin
        w[i] = decay ** (newest - g.season)
        observed[g.home] += 1
        observed[g.away] += 1

    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw

    P = np.eye(n + 1)
    P[n, n] = 0.0                      # never shrink home advantage
    anchor = np.zeros((1, n + 1))
    anchor[0, :n] = 1.0                # ratings sum to zero
    Xa = np.vstack([Xw, 10.0 * anchor])
    ya = np.concatenate([yw, [0.0]])

    A = Xa.T @ Xa + ridge * P
    beta = np.linalg.solve(A, Xa.T @ ya)

    resid = y - X @ beta
    try:
        Ainv = np.linalg.inv(A)
        # Weighted dof must use the SUM OF WEIGHTS, not the raw game count. Using
        # len(games) against a weighted residual sum inflates the denominator by
        # 1/mean(w) and deflates the SD by its square root — it reported 12.40
        # against a true ~17, which would have made every win probability far too
        # confident and manufactured edges across the board.
        dof = max(float(w.sum()) - float(np.trace(Xw @ Ainv @ Xw.T)), 1.0)
        residual_sd = float(np.sqrt(float((resid * w) @ resid) / dof))
        cov = Ainv * (residual_sd ** 2)
        std_errors = {t: float(np.sqrt(max(cov[idx[t], idx[t]], 0.0))) for t in teams}
    except np.linalg.LinAlgError:
        residual_sd = float(resid.std())
        std_errors = {t: float("nan") for t in teams}

    return EmpiricalFit(
        ratings={t: float(beta[idx[t]]) for t in teams},
        hfa=float(beta[n]), residual_sd=residual_sd,
        n_games=len(games), n_teams=n, games_observed=observed,
        seasons=sorted({g.season for g in games}), std_errors=std_errors,
    )


def blend(
    market: dict[str, float],
    prior: dict[str, float],
    market_weight: float = 0.75,
) -> tuple[dict[str, float], dict[str, float]]:
    """Combine market-implied and results-based ratings.

    Returns (blended, divergence) where divergence = market - prior.

    The market gets most of the weight because it is sharper and current; the prior
    supplies an independent check. **Divergence is the interesting output.** A large
    positive divergence means the market rates a team well above what it did on the
    field — offseason optimism about a new coach, a portal haul, a returning QB. That
    is exactly the kind of belief that is sometimes right and sometimes a bubble, and
    it is where the qualitative threads have something to say that the numbers cannot.
    """
    blended, divergence = {}, {}
    for t, m in market.items():
        p = prior.get(t)
        if p is None:
            blended[t] = m
            continue
        blended[t] = market_weight * m + (1.0 - market_weight) * p
        divergence[t] = m - p
    return blended, divergence


def build_prior(data_dir: str, years=(2022, 2023, 2024, 2025), cache: bool = True) -> EmpiricalFit | None:
    """Download if needed, fit, and cache the results-based prior."""
    raw = os.path.join(data_dir, "raw")
    paths = download_seasons(years, raw)
    if not paths:
        return None
    fit = fit_from_results(load_results(list(paths.values())))
    if cache:
        with open(os.path.join(data_dir, "empirical_prior.json"), "w") as fh:
            json.dump(fit.as_dict(), fh, indent=2)
    return fit


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    f = build_prior(os.path.join(here, "data"))
    if f is None:
        print("No seasons downloadable — check egress to raw.githubusercontent.com")
    else:
        print(f"seasons={f.seasons} games={f.n_games} teams={f.n_teams} "
              f"HFA={f.hfa:.2f} residual_SD={f.residual_sd:.2f}")
        top = sorted(f.ratings.items(), key=lambda kv: -kv[1])[:10]
        for t, v in top:
            print(f"   {t:<22}{v:+7.2f}")
