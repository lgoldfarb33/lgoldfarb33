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
from model.ratings import solve_ratings, suggest_ridge, build_adjustments, apply_adjustments
from model.simulate import ScheduledGame, simulate_season, win_probability
from model.sources import FixtureSource, Game

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "data", "fixtures")


def _fixture():
    if not os.path.exists(os.path.join(FIX, "synthetic_market.json")):
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


# ------------------------------------------------------------ end-to-end

def test_pipeline_recovers_planted_errors():
    """The whole point: find book errors the adjustment layer flags, and nothing else."""
    from model.run import run
    r = run(source="fixture", n_sims=12000)
    truth = json.load(open(os.path.join(FIX, "synthetic_truth.json")))
    bettable = {p["subject"] for p in r["picks"] if p["edge"]["bettable"]}
    planted = set(truth["book_error"])
    assert planted <= bettable, f"missed planted errors: {planted - bettable}"
    assert not (bettable - planted), f"false positives: {bettable - planted}"
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
