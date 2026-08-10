"""Test suite. Run: python3 -m pytest tests/ -q   (or python3 tests/test_model.py)"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import make_fixture
from model.calibrate import build_report
from model.edge import (american_to_decimal, american_to_implied, devig_two_way,
                        evaluate, kelly_fraction)
from model.independence import assess, confidence_tier
from model.ledger import Ledger, LedgerEntry, make_pick_id
from model.empirical import blend, RESULTS_RIDGE, FCS_FALLBACK_RATING
from model.ratings import solve_ratings, suggest_ridge, build_adjustments, apply_adjustments
from model.simulate import ScheduledGame, simulate_season, win_probability
from model.sources import FixtureSource, Game

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "data", "fixtures")


def _fixture():
    """Load the fixture, regenerating it if the engine's constants have moved.

    The fixture prices its win totals with the simulator's margin SD. When that
    constant changed (16.5 -> a measured 17.5) a stale fixture reported two false
    positives — the engine was correctly finding the gap between the old curve and
    the new one. Staleness must fail loudly or not at all.
    """
    from model.sources import MARGIN_SD_DEFAULT
    path = os.path.join(FIX, "synthetic_market.json")
    stale = True
    if os.path.exists(path):
        stamp = json.load(open(path)).get("generated_with", {})
        stale = stamp.get("margin_sd") != MARGIN_SD_DEFAULT
    if stale:
        make_fixture.build(FIX)
    snap = FixtureSource(os.path.join(FIX, "synthetic_market.json")).fetch()
    truth = json.load(open(os.path.join(FIX, "synthetic_truth.json")))
    sched = [ScheduledGame(**g) for g in
             json.load(open(os.path.join(FIX, "synthetic_schedule.json")))["games"]]
    return snap, truth, sched


# ----------------------------------------------------------------- ratings

def test_recovers_truth():
    """Ratings solved from spreads must reproduce the ratings that generated them."""
    snap, truth, _ = _fixture()
    sol = solve_ratings(snap.games)
    book = truth["book_ratings"]
    est = np.array([sol.ratings[t] for t in book])
    act = np.array([book[t] for t in book])
    assert np.corrcoef(est, act)[0, 1] > 0.995
    assert np.abs(est - act).mean() < 0.5
    assert abs(sol.hfa - truth["true_hfa"]) < 0.5


def test_ridge_does_not_over_shrink():
    """Default ridge must not compress the rating scale (regression: ridge=3.0 -> slope 0.84)."""
    snap, truth, _ = _fixture()
    sol = solve_ratings(snap.games)
    book = truth["book_ratings"]
    slope = float(np.polyfit([book[t] for t in book], [sol.ratings[t] for t in book], 1)[0])
    assert 0.95 < slope < 1.05, f"rating scale distorted, slope={slope}"


def test_suggest_ridge_scales_with_density():
    # home must cycle within the same 12 teams, or every "dense" game invents a
    # new team and the schedule is actually sparse.
    dense = [Game(home=f"T{i%12}", away=f"T{(i+1)%12}", home_margin=3.0) for i in range(60)]
    sparse = [Game(home="A", away="B", home_margin=3.0)]
    assert suggest_ridge(sparse) > suggest_ridge(dense)


def test_sparse_teams_flagged_unidentified():
    sol = solve_ratings([Game(home="A", away="B", home_margin=7.0)])
    assert not sol.is_identified("A")
    assert sol.warnings


# -------------------------------------------------------------- simulation

def test_simulation_is_unbiased():
    """Projected wins must match the expected wins implied by the same ratings."""
    snap, truth, sched = _fixture()
    book = truth["book_ratings"]
    sim = simulate_season(book, sched, truth["true_hfa"], n_sims=20000)
    errs = []
    for t in book:
        exp = make_fixture._expected_wins(
            t, [{"home": g.home, "away": g.away, "neutral": g.neutral} for g in sched], book)
        errs.append(sim.projections[t].mean_wins - exp)
    assert abs(float(np.mean(errs))) < 0.05
    assert float(np.mean(np.abs(errs))) < 0.15


def test_uncertainty_widens_distribution():
    """Propagating rating SE must widen win distributions, not narrow them."""
    _, truth, sched = _fixture()
    book = truth["book_ratings"]
    tight = simulate_season(book, sched, truth["true_hfa"], n_sims=8000)
    loose = simulate_season(book, sched, truth["true_hfa"], n_sims=8000,
                            rating_se={t: 3.0 for t in book})
    t0 = next(iter(book))
    spread_tight = tight.projections[t0].p90 - tight.projections[t0].p10
    spread_loose = loose.projections[t0].p90 - loose.projections[t0].p10
    assert spread_loose >= spread_tight


def test_win_probability_monotone():
    assert win_probability(-14) < win_probability(0) < win_probability(14)
    assert abs(win_probability(0) - 0.5) < 1e-9


# -------------------------------------------------------------------- edge

def test_devig_sums_to_one():
    o, u = devig_two_way(-110, -110)
    assert abs(o + u - 1.0) < 1e-9
    assert abs(o - 0.5) < 1e-9


def test_devig_actually_removes_vig():
    """Raw implied probs overstate; de-vigged must be strictly smaller."""
    assert american_to_implied(-110) > devig_two_way(-110, -110)[0]


def test_kelly_zero_at_fair_odds():
    assert abs(kelly_fraction(0.5, american_to_decimal(100))) < 1e-9
    assert kelly_fraction(0.6, american_to_decimal(100)) > 0
    assert kelly_fraction(0.4, american_to_decimal(100)) < 0


def test_one_sided_price_is_not_bettable():
    """Without both sides the vig cannot be separated, so no bet is offered."""
    r = evaluate("X", "win_total", "over", 9.5, 0.75, -110, None)
    assert not r.bettable and "vig NOT removed" in r.reason


def test_small_edge_rejected():
    """Edges under the threshold are noise relative to our estimation error."""
    r = evaluate("X", "win_total", "over", 9.5, 0.52, -110, -110)   # 2% edge
    assert not r.bettable and "below" in r.reason
    assert evaluate("X", "win_total", "over", 9.5, 0.60, -110, -110).bettable


# ------------------------------------------------------------ independence

def test_shared_source_discounts_corroboration():
    """The real failure this module exists for: one ESPN article counted twice."""
    espn = "https://www.espn.com/college-football/story/_/id/49000177/brendan-sorsby"
    rep = assess({"personnel": [espn], "sentiment": [espn]})
    assert rep.raw_threads == 2
    assert rep.effective_threads < 1.5, "identical sources must not count as two threads"
    assert espn in rep.shared_urls


def test_distinct_sources_keep_full_credit():
    rep = assess({"personnel": ["https://on3.com/a"], "sentiment": ["https://247sports.com/b"]})
    assert rep.effective_threads >= 1.9


def test_same_domain_partially_discounted():
    """Two different ESPN pieces: more independent than one, less than two outlets."""
    rep = assess({"personnel": ["https://espn.com/a"], "sentiment": ["https://espn.com/b"]})
    assert 1.0 < rep.effective_threads < 2.0


def test_tier_rules():
    assert confidence_tier(2.0, price_verified=True)[0] == "high"
    assert confidence_tier(2.0, price_verified=False)[0] == "medium"
    assert confidence_tier(2.5, contradicted=True)[0] == "speculative"
    assert confidence_tier(2.5, weakest_confidence="rumor")[0] == "speculative"
    assert confidence_tier(1.2)[0] == "speculative"


# ------------------------------------------------------ adjustments layer

def test_adjustments_apply_and_cap():
    personnel = {"team_scores": [{"team": "A", "net_personnel": 2}]}
    sentiment = {"teams": [{"team": "A", "score": 2, "sources": ["u"]}]}
    adj = build_adjustments(personnel, sentiment)
    assert len(adj["A"]) == 2
    sol = solve_ratings([Game(home="A", away="B", home_margin=0.0)])
    _, applied = apply_adjustments(sol, adj, cap=1.0)
    assert abs(applied["A"]) <= 1.0, "cap must bound the qualitative layer"


# ------------------------------------------------------------------ ledger

def test_ledger_is_append_only():
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(d)
        e = LedgerEntry(
            pick_id=make_pick_id("2026-08-10", "A", "win_total", "over", 9.5),
            placed_at="2026-08-10", subject="A", market="win_total", side="over",
            line=9.5, price_american=-110, book="B", model_prob=0.6,
            market_prob=0.5, edge=0.1, stake_fraction=0.02, tier="high",
            effective_threads=2.0)
        led.record(e)
        led.record(e)                       # duplicate must not double-write
        assert len(led.all()) == 1
        led.settle(e.pick_id, "win", closing_price_american=-130)
        assert len(led.all()) == 1          # collapses to latest revision
        assert led.all()[0].settled
        assert len(open(led.path).readlines()) == 2   # both revisions retained on disk


def test_clv_requires_consistent_vig_basis():
    """Regression for a bug a review caught: comparing de-vigged open prob against
    a RAW closing price inflated CLV by ~2.2pp at -110/-110 on every settled pick —
    exactly the "genuine edge" signal calibrate.py watches for. Passing only one
    closing price must not silently compute a biased number."""
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(d)
        e = LedgerEntry(
            pick_id="clv1", placed_at="2026-08-10", subject="A", market="win_total",
            side="over", line=9.5, price_american=-110, book="B", model_prob=0.60,
            market_prob=0.50, edge=0.10, stake_fraction=0.02, tier="high",
            effective_threads=2.0)
        led.record(e)
        one_sided = led.settle("clv1", "win", closing_price_american=-130)
        assert one_sided.clv_prob is None, "one closing price must not fabricate a CLV number"

    with tempfile.TemporaryDirectory() as d:
        led = Ledger(d)
        led.record(e)
        both_sided = led.settle("clv1", "win", closing_price_american=-130,
                                closing_opposite_price_american=110)
        assert both_sided.clv_prob is not None
        # de-vigged close for a -130/+110 market is well below the raw -130 implied
        # probability (56.5%) — the bug this guards against would have used the raw
        # number and overstated CLV by several points.
        from model.edge import american_to_implied
        assert both_sided.clv_prob < american_to_implied(-130) - both_sided.market_prob


def test_side_chosen_by_edge_not_probability():
    """Regression: picking the higher-PROBABILITY side instead of the higher-EDGE
    side silently rejects real bets. If the model gives the over 45% against a
    de-vigged market of 35% (a genuine +10pt edge), comparing raw probabilities
    instead evaluates the under (55% model vs 65% market, negative edge) and skips
    a bet that should have been taken."""
    # Market favors under (-180) over over (+150) -> devigged p(over)~0.38,
    # p(under)~0.62. Model thinks over is more likely than the market does (0.45
    # vs 0.38, positive edge) while still rating under as the more probable
    # OUTCOME in isolation (0.55 vs 0.45) -- the scenario where picking by raw
    # probability and picking by edge disagree.
    over = evaluate("X", "win_total", "over", 9.5, 0.45, 150, -180)
    under = evaluate("X", "win_total", "under", 9.5, 0.55, -180, 150)
    assert over.edge is not None and over.edge > 0
    assert under.edge is not None and under.edge < 0
    chosen = max([over, under], key=lambda c: c.edge)
    assert chosen.side == "over", "picking by max probability would wrongly select 'under' here"


def test_synthetic_excluded_from_calibration():
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(d)
        e = LedgerEntry(
            pick_id="synth1", placed_at="2026-08-10", subject="A", market="win_total",
            side="over", line=9.5, price_american=-110, book="B", model_prob=0.9,
            market_prob=0.5, edge=0.4, stake_fraction=0.05, tier="high",
            effective_threads=3.0, synthetic=True)
        led.record(e)
        led.settle("synth1", "win")
        assert led.settled() == []


# ------------------------------------------------------------- calibration

def test_calibration_refuses_small_samples():
    rep = build_report([], 0)
    assert rep.n_settled == 0 and "Nothing about this model" in rep.verdict
    entries = [LedgerEntry(
        pick_id=f"p{i}", placed_at="2026-08-10", subject=f"T{i}", market="win_total",
        side="over", line=9.5, price_american=-110, book="B", model_prob=0.6,
        market_prob=0.5, edge=0.1, stake_fraction=0.02, tier="high",
        effective_threads=2.0, settled=True, result="win" if i % 2 else "loss")
        for i in range(10)]
    rep = build_report(entries, 0)
    assert "too few" in rep.verdict


def test_calibration_detects_overconfidence():
    entries = [LedgerEntry(
        pick_id=f"p{i}", placed_at="2026-08-10", subject=f"T{i}", market="win_total",
        side="over", line=9.5, price_american=-110, book="B", model_prob=0.90,
        market_prob=0.5, edge=0.4, stake_fraction=0.05, tier="high",
        effective_threads=2.0, settled=True, result="win" if i < 15 else "loss")
        for i in range(40)]   # claims 90%, delivers 37.5%
    rep = build_report(entries, 0)
    assert any("overconfident" in w for w in rep.warnings)


# -------------------------------------------------------- empirical prior

def test_results_ridge_is_cv_selected_not_a_guess():
    """Regression: RESULTS_RIDGE was 25.0, an unvalidated guess. 5-fold CV on the
    real dataset selected 1.0 (8% better out-of-sample RMSE than 25). Guard
    against silently drifting back toward heavy, unjustified shrinkage."""
    assert RESULTS_RIDGE <= 3.0, (
        f"RESULTS_RIDGE={RESULTS_RIDGE} is far from the CV-selected optimum (~1.0) — "
        f"if this was changed deliberately, re-run the CV sweep in empirical.py's "
        f"module docstring and update the comment, don't just bump the number")


def test_blend_covers_prior_only_teams():
    """Regression: a team present only in the results prior (the normal case for
    any schedule opponent the current market snapshot has no line on) used to be
    dropped from the blended ratings entirely, which forced the simulator to skip
    every game that opponent played in and silently understated real teams' win
    totals by ~1 win whenever their schedule included one."""
    market = {"A": 5.0, "B": -2.0}
    prior = {"A": 4.0, "B": -1.0, "C": -10.0}   # C: no market line this week
    blended, divergence = blend(market, prior)
    assert "C" in blended and blended["C"] == -10.0
    assert "C" not in divergence, "divergence is only meaningful for teams in both sets"


def test_blend_divergence_is_recentered():
    """Regression: market is anchored sum-to-zero over the snapshot's teams; prior
    is anchored over the full ~136-team FBS population. Differencing them without
    recentering on their shared teams bakes in a constant offset with no game-level
    meaning whenever the snapshot is a non-representative (e.g. ranked-team-heavy)
    subset."""
    # Every team is uniformly 10 pts "higher" in market than prior — pure offset,
    # zero real disagreement. Recentered divergence must wash this out to ~0.
    market = {"A": 15.0, "B": 5.0, "C": -5.0}
    prior = {"A": 5.0, "B": -5.0, "C": -15.0}
    _, divergence = blend(market, prior)
    for t, d in divergence.items():
        assert abs(d) < 1e-9, f"uniform offset should recenter to ~0 divergence, got {t}={d}"


def test_fallback_rating_fills_unrated_schedule_opponents():
    """Regression: a schedule opponent with no rating anywhere (typically FCS —
    load_results excludes non-FBS teams by construction) caused simulate_season to
    drop every game that opponent appeared in rather than use a documented
    placeholder. Exercises the same fill logic run() applies before simulating."""
    ratings = {"RealTeam": 10.0}
    schedule = [ScheduledGame(home="RealTeam", away="FCS Opponent", neutral=False)]
    missing = {t for g in schedule for t in (g.home, g.away)} - set(ratings)
    assert missing == {"FCS Opponent"}
    filled = dict(ratings)
    for t in missing:
        filled[t] = FCS_FALLBACK_RATING
    sim = simulate_season(filled, schedule, hfa=3.2, n_sims=4000)
    assert sim.projections["RealTeam"].unidentified_opponents == 0
    assert sim.projections["RealTeam"].mean_wins > 0.9, "heavy favorite over an FCS-level fallback"


# ------------------------------------------------------------ end-to-end

def test_pipeline_recovers_planted_errors():
    """The whole point: find book errors the adjustment layer flags, clearly.

    The fixture's "fair" prices are generated by routing through the exact same
    solve_ratings + simulate_season pipeline run() uses (see make_fixture.py), so a
    non-mispriced team prices to ~zero edge under the model's own methodology.
    "~zero" is not "exactly zero": fixture generation and the run under test use
    different Monte Carlo seeds/sample sizes, so a team sitting near a half-integer
    win-total boundary can occasionally land a percent or two on either side of
    the 3% MIN_EDGE_TO_BET floor by chance — confirmed by rerunning at several
    n_sims and finding a different, small, near-floor "extra" team each time. That
    is honest estimation noise, not a bug, and demanding exact set equality here
    would be testing for a lucky seed alignment rather than real correctness.
    The planted errors (deliberately 3-5x the floor) must clear it cleanly; any
    other pick must not.
    """
    from model.run import run
    r = run(source="fixture", n_sims=20000)
    truth = json.load(open(os.path.join(FIX, "synthetic_truth.json")))
    picks_by_team = {p["subject"]: p for p in r["picks"]}
    planted = set(truth["book_error"])

    for team in planted:
        edge = picks_by_team[team]["edge"]["edge"]
        assert edge is not None and edge > 0.10, (
            f"planted error at {team} should clear the floor with a wide margin, got edge={edge}")

    NOISE_BAND = 0.06   # observed near-floor noise tops out ~3.4% across reruns; 6% is a safety margin
    for p in r["picks"]:
        if p["subject"] in planted:
            continue
        edge = p["edge"]["edge"] or 0.0
        assert edge < NOISE_BAND, (
            f"{p['subject']} shows edge={edge:.3f} well above the estimation-noise band — "
            f"likely a real false positive, not sampling noise")

    assert r["SYNTHETIC"] is True


def test_synthetic_run_is_labelled():
    from model.run import render_markdown, run
    md = render_markdown(run(source="fixture", n_sims=4000))
    assert "SYNTHETIC" in md and "Nothing here is a bet" in md


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
