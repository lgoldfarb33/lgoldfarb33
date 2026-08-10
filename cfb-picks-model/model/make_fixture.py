"""Generate a synthetic league with KNOWN true ratings.

This exists to validate the engine, not to produce picks. Because the true
ratings are known by construction, we can check that `solve_ratings` recovers
them and that `simulate_season` finds the win totals we deliberately mispriced.
A model that cannot recover truth from clean synthetic data has no business
being pointed at real money.

Every artifact it writes is stamped synthetic=True so it can never be confused
with a real market snapshot.
"""
from __future__ import annotations

import json
import os

import numpy as np

TEAMS = [
    "Anchor State", "Basalt Tech", "Cedar A&M", "Dune Valley", "Ember City",
    "Foxglove U", "Granite Bay", "Harbor Point", "Ironwood", "Juniper State",
    "Kestrel College", "Lantern Hill", "Marrow University", "Nettle State",
    "Obsidian A&M", "Pinnacle West",
]

TRUE_HFA = 2.6
MARGIN_NOISE_SD = 0.8   # bookmaker pricing noise around the true rating gap
SEED = 424242


VIG = 0.045  # ~4.5% hold, typical for a two-way total


def _game_probs(team, schedule, ratings):
    """Per-game win probabilities for `team` under the supplied ratings."""
    from scipy.stats import norm
    from .sources import MARGIN_SD_DEFAULT
    ps = []
    for g in schedule:
        if team not in (g["home"], g["away"]):
            continue
        ish = g["home"] == team
        opp = g["away"] if ish else g["home"]
        m = (ratings[team] - ratings[opp]) + (
            0.0 if g["neutral"] else (TRUE_HFA if ish else -TRUE_HFA))
        ps.append(float(norm.cdf(m / MARGIN_SD_DEFAULT)))
    return np.array(ps)


def _expected_wins(team, schedule, ratings):
    return float(_game_probs(team, schedule, ratings).sum())


def _prob_over(team, posted, schedule, ratings, n=40000, seed=99):
    """P(season wins > posted) under the supplied ratings."""
    rng = np.random.default_rng(seed)
    ps = _game_probs(team, schedule, ratings)
    wins = (rng.random((n, len(ps))) < ps).sum(axis=1)
    return float((wins > posted).mean())


def _to_american(p: float) -> int:
    """Convert a probability to American odds."""
    p = min(max(p, 0.01), 0.99)
    dec = 1.0 / p
    return int(round(-100.0 / (dec - 1.0))) if dec < 2.0 else int(round((dec - 1.0) * 100))


def _fair_prices(p_over: float) -> tuple[int, int]:
    """Two-way prices carrying VIG, centered on the true probability."""
    scale = 1.0 + VIG
    return _to_american(p_over * scale), _to_american((1.0 - p_over) * scale)


def build(out_dir: str, n_games_per_team: int = 11) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(TEAMS)

    # True ratings, centered at zero, spread like a real FBS conference.
    true_ratings = {t: float(r) for t, r in zip(TEAMS, rng.normal(0, 9.0, n).round(2))}
    mean_r = sum(true_ratings.values()) / n
    true_ratings = {t: round(r - mean_r, 2) for t, r in true_ratings.items()}

    # Round-robin-ish schedule: each team plays n_games_per_team distinct opponents.
    schedule, seen = [], set()
    for i, home in enumerate(TEAMS):
        for k in range(1, n_games_per_team + 1):
            away = TEAMS[(i + k) % n]
            pair = tuple(sorted((home, away)))
            if pair in seen or home == away:
                continue
            seen.add(pair)
            neutral = (len(schedule) % 17 == 0)
            schedule.append({"home": home, "away": away, "neutral": neutral,
                             "week": str(1 + len(schedule) % 12)})

    # The book's BELIEF about each team, which is what its lines encode. For most
    # teams belief equals reality; for a few the book is wrong by a few points.
    #
    # This is the causal chain the real product depends on. Ratings are solved from
    # the book's own spreads, so the model inherits the book's errors and cannot
    # find them on its own. The only way it beats the market is if the qualitative
    # threads supply a correction. Planting the error in the book's belief — rather
    # than in the posted number alone — is what makes that testable.
    book_error = {TEAMS[2]: +3.0, TEAMS[7]: -3.0, TEAMS[11]: +3.0}
    book_ratings = {t: true_ratings[t] + book_error.get(t, 0.0) for t in TEAMS}

    # Posted spreads = the book's believed gap + HFA + small pricing noise.
    games = []
    for g in schedule:
        gap = book_ratings[g["home"]] - book_ratings[g["away"]]
        margin = gap + (0.0 if g["neutral"] else TRUE_HFA) + rng.normal(0, MARGIN_NOISE_SD)
        games.append({
            "home": g["home"], "away": g["away"],
            "home_margin": round(float(margin) * 2) / 2,   # half-point grid
            "total": None, "neutral": g["neutral"], "date": "2026-09-05",
            "book": "SyntheticBook", "source_url": "fixture://synthetic", "week": g["week"],
        })

    # Win totals reflect the book's belief, and the juice is priced fairly against
    # that belief — a fully self-consistent, professionally-priced market. Its only
    # flaw is that the belief is wrong for three teams.
    #
    # Pricing MUST go through the exact same pipeline run() uses to evaluate picks
    # — solve_ratings() then simulate_season() with rating_se propagation — not a
    # simplified stand-in. Two earlier versions of this file learned that lesson
    # the hard way:
    #   1. A logistic approximation instead of the simulator's normal CDF
    #      manufactured apparent edges on 11 of 16 teams.
    #   2. Pricing off the exact true book_ratings with a deterministic CDF sum,
    #      while the live model prices off *fitted* ratings plus their standard
    #      errors (simulate_season(..., rating_se=...)), left a structural gap
    #      between "fair" and what the model actually computes for a fair team —
    #      deterministic, reproducible false positives on 3 of the 13 non-mispriced
    #      teams, every run, at the same 3 teams. Zero information gained from
    #      more simulations; the two computations were just answering slightly
    #      different questions.
    # Routing both through solve_ratings + simulate_season closes that gap: a team
    # with no book_error now prices to (approximately) zero edge under the model's
    # own methodology, not under an idealized stand-in for it.
    from .ratings import solve_ratings
    from .simulate import simulate_season, ScheduledGame
    from .sources import Game

    game_objs = [Game(home=g["home"], away=g["away"], home_margin=g["home_margin"],
                      neutral=g["neutral"]) for g in games]
    sched_objs = [ScheduledGame(home=g["home"], away=g["away"], neutral=g["neutral"],
                                week=g["week"]) for g in schedule]
    fit = solve_ratings(game_objs)
    price_sim = simulate_season(fit.ratings, sched_objs, fit.hfa, n_sims=60000,
                                rating_se=fit.std_errors, seed=13)

    win_totals = []
    for t in TEAMS:
        proj = price_sim.projections[t]
        # Snap to the nearest half-integer so no total can push. Naively rounding
        # then bumping integers up by 0.5 biases every total upward and hands the
        # UNDER a free edge on every team — an earlier version did exactly that.
        posted = round(proj.mean_wins - 0.5) + 0.5
        p_over = proj.prob_over(posted)
        over_price, under_price = _fair_prices(p_over)
        win_totals.append({
            "team": t, "total": posted,
            "over_price": over_price, "under_price": under_price,
            "book": "SyntheticBook", "source_url": "fixture://synthetic",
            "il_retail_only": False,
        })

    # Synthetic thread output that correctly flags the three teams the book has
    # wrong. Scores are chosen so score x coefficient roughly cancels book_error:
    # net_personnel is worth 1.0 pt/unit and sentiment 0.75 pt/unit, so -2/-2
    # removes 3.5 points. Sources differ per thread, so the independence checker
    # sees genuine corroboration rather than one article counted twice.
    def _rows(sign):
        return int(-2 * sign), int(-2 * sign)
    personnel_rows, sentiment_rows = [], []
    for t in TEAMS:
        err = book_error.get(t, 0.0)
        if err == 0:
            continue
        s = 1 if err > 0 else -1
        p_score, s_score = _rows(s)
        personnel_rows.append({
            "team": t, "net_personnel": p_score, "coaching_fit": p_score,
            "roster_continuity": p_score,
            "one_line_rationale": f"Synthetic: book overrates by {err:+.1f} pts",
        })
        sentiment_rows.append({
            "team": t, "score": s_score, "sources": [f"fixture://sentiment/{t}"],
            "justification": f"Synthetic: independent corroboration of a {err:+.1f} pt book error",
        })
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "synthetic_personnel.json"), "w") as fh:
        json.dump({"thread": "personnel", "synthetic": True,
                   "team_scores": personnel_rows,
                   "coaching_changes": [
                       {"team": r["team"], "source_url": f"fixture://personnel/{r['team']}"}
                       for r in personnel_rows]}, fh, indent=2)
    with open(os.path.join(out_dir, "synthetic_sentiment.json"), "w") as fh:
        json.dump({"thread": "sentiment", "synthetic": True,
                   "teams": sentiment_rows}, fh, indent=2)

    from .sources import MARGIN_SD_DEFAULT as _SD
    snapshot = {
        "as_of": "2026-08-10", "synthetic": True, "source": "fixture",
        # Stamped so a stale fixture is detectable. The fixture's win totals are
        # priced with the simulator's margin SD; if that constant changes and the
        # fixture is not regenerated, validation silently reports false positives.
        "generated_with": {"margin_sd": _SD, "true_hfa": TRUE_HFA},
        "notes": [
            "SYNTHETIC DATA — generated by model/make_fixture.py, not a real market.",
            "Exists to validate the engine against known truth. Never a basis for a bet.",
        ],
        "games": games, "win_totals": win_totals,
    }
    with open(os.path.join(out_dir, "synthetic_market.json"), "w") as fh:
        json.dump(snapshot, fh, indent=2)

    truth = {
        "synthetic": True,
        "true_ratings": true_ratings, "book_ratings": book_ratings,
        "true_hfa": TRUE_HFA, "book_error": book_error,
        "note": ("Ground truth for validation; the engine never reads this file. "
                 "book_error is what the market has wrong — recoverable only via "
                 "the thread adjustment layer, since ratings are solved from the "
                 "book's own spreads."),
    }
    with open(os.path.join(out_dir, "synthetic_truth.json"), "w") as fh:
        json.dump(truth, fh, indent=2)

    with open(os.path.join(out_dir, "synthetic_schedule.json"), "w") as fh:
        json.dump({"synthetic": True, "games": schedule}, fh, indent=2)

    return {"teams": n, "games": len(games), "win_totals": len(win_totals)}


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(build(os.path.join(here, "data", "fixtures")))
