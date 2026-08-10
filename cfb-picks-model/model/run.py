"""Orchestrator: market snapshot -> ratings -> simulation -> sized bets -> ledger.

Usage
-----
  python3 -m model.run --source fixture              # synthetic validation run
  python3 -m model.run --source odds_api             # live (needs egress + ODDS_API_KEY)
  python3 -m model.run --source real-market          # ratings only from data/market.json
  python3 -m model.run --calibrate                   # calibration report only

Runs against synthetic data are stamped SYNTHETIC end to end and are excluded from
calibration, so a validation run can never contaminate the record of how the model
actually performs.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from .calibrate import build_report
from .empirical import blend, build_prior
from .edge import evaluate
from .independence import assess, confidence_tier
from .ledger import Ledger, LedgerEntry, make_pick_id
from .ratings import (apply_adjustments, build_adjustments, solve_ratings,
                      MIN_GAMES_FOR_CONFIDENCE, ADJUSTMENT_COEFFICIENTS)
from .simulate import ScheduledGame, simulate_season
from .sources import FixtureSource, Game, MarketSnapshot, OddsAPISource, SourceError, WinTotal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
FIXTURES = os.path.join(DATA, "fixtures")

DISCLAIMER = "Not financial advice. Research output for entertainment and analysis."


def _load_env() -> None:
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _read_json(path: str):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def load_snapshot(kind: str) -> tuple[MarketSnapshot, list[ScheduledGame]]:
    """Return (snapshot, schedule). Schedule may be empty when none is available."""
    if kind == "fixture":
        snap = FixtureSource(os.path.join(FIXTURES, "synthetic_market.json")).fetch()
        raw = _read_json(os.path.join(FIXTURES, "synthetic_schedule.json")) or {"games": []}
        return snap, [ScheduledGame(**g) for g in raw["games"]]

    if kind == "odds_api":
        _load_env()
        src = OddsAPISource()
        snap = src.fetch(as_of=datetime.now(timezone.utc).date().isoformat())
        # The Odds API serves upcoming games, not full-season schedules. Until a
        # schedule source is wired in, season simulation is unavailable.
        return snap, []

    if kind == "real-market":
        mk = _read_json(os.path.join(DATA, "market.json"))
        if not mk:
            raise SourceError("data/market.json not found — run the research threads first")
        games = []
        for g in mk.get("game_lines", []):
            fav, spread = g.get("spread_favorite"), g.get("spread")
            if not fav or spread in (None, 0.0):
                continue
            home, away = g.get("home"), g.get("away")
            margin = float(spread) if fav == home else -float(spread)
            games.append(Game(
                home=home, away=away, home_margin=margin,
                total=(g.get("total") or None), neutral=bool(g.get("neutral_site", "").startswith("true")),
                date=g.get("date", ""), book=g.get("book", ""), source_url=g.get("source_url", ""),
                week=str(g.get("week", "")),
            ))
        wts = [WinTotal(
            team=w["team"], total=float(w["total"]),
            over_price=_am(w.get("over_price")), under_price=_am(w.get("under_price")),
            book=w.get("book", ""), source_url=w.get("source_url", ""),
            il_retail_only=bool(w.get("il_retail_only")),
        ) for w in mk.get("win_totals", []) if w.get("total") is not None]
        snap = MarketSnapshot(as_of=mk.get("as_of", ""), games=games, win_totals=wts,
                              source="fixture", synthetic=False,
                              notes=["Derived from data/market.json (research-thread output)."])
        return snap, []

    raise SourceError(f"Unknown source: {kind}")


def _am(v) -> int | None:
    try:
        return int(str(v).replace("+", "")) if str(v).strip() not in ("", "None") else None
    except (TypeError, ValueError):
        return None


def _evidence_for(team: str, personnel: dict | None, sentiment: dict | None) -> dict[str, list[str]]:
    """Collect the source URLs each thread used for this team, for overlap scoring."""
    ev: dict[str, list[str]] = {}
    for row in (personnel or {}).get("coaching_changes", []) + (personnel or {}).get("portal_impact", []) \
            + (personnel or {}).get("returning_production", []) + (personnel or {}).get("depth_chart_battles", []):
        if row.get("team") == team and row.get("source_url"):
            ev.setdefault("personnel", []).append(row["source_url"])
    for row in (sentiment or {}).get("teams", []):
        if row.get("team") == team:
            ev.setdefault("sentiment", []).extend(row.get("sources", []))
    for key in ("injuries", "suspensions_eligibility", "qb_situations"):
        for row in (sentiment or {}).get(key, []):
            if row.get("team") == team and row.get("source_url"):
                ev.setdefault("sentiment", []).append(row["source_url"])
    return ev


# One unit = 1% of bankroll. Staking in units rather than raw percentages is how
# anyone actually tracks a season, and it keeps sizing legible when bankroll moves.
UNIT_PCT = 0.01


def run(source: str = "fixture", n_sims: int = 20000, bankroll: float = 0.0,
        unit_pct: float = UNIT_PCT, use_prior: bool = True) -> dict:
    snap, schedule = load_snapshot(source)
    # A synthetic run must read synthetic threads; the real ones name real teams
    # that do not exist in the fixture, so every adjustment would silently no-op
    # and the adjustment layer would go untested.
    if source == "fixture":
        personnel = _read_json(os.path.join(FIXTURES, "synthetic_personnel.json"))
        sentiment = _read_json(os.path.join(FIXTURES, "synthetic_sentiment.json"))
    else:
        personnel = _read_json(os.path.join(DATA, "personnel.json"))
        sentiment = _read_json(os.path.join(DATA, "sentiment.json"))

    sol = solve_ratings(snap.games)

    # Blend in the results-based prior. This is the only part of the model built
    # from information the market did not generate, so it is the only part that can
    # independently say the market is wrong.
    prior_info, divergence = None, {}
    if use_prior and not snap.synthetic:
        fit = build_prior(DATA)
        if fit is not None:
            blended, divergence = blend(sol.ratings, fit.ratings, market_weight=0.75)
            sol.ratings = blended
            prior_info = {
                "seasons": fit.seasons, "n_games": fit.n_games, "n_teams": fit.n_teams,
                "measured_hfa": round(fit.hfa, 2),
                "measured_residual_sd": round(fit.residual_sd, 2),
                "market_weight": 0.75,
                "note": ("Ratings fit from realized scoring margins, independent of any "
                         "betting market. Divergence = market rating minus this prior."),
            }

    adjustments = build_adjustments(personnel, sentiment)
    adjusted, applied = apply_adjustments(sol, adjustments)

    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "SYNTHETIC": snap.synthetic,
        "source": source,
        "as_of": snap.as_of,
        "ratings": {
            "n_teams": sol.n_teams, "n_games": sol.n_games, "ridge": sol.ridge,
            "hfa": round(sol.hfa, 3), "hfa_fitted": sol.hfa_fitted,
            "residual_sd": round(sol.residual_sd, 3),
            "effective_dof": round(sol.effective_dof, 1),
            "warnings": sol.warnings,
            "table": sorted(
                ({"team": t, "rating": round(adjusted[t], 2),
                  "market_rating": round(sol.ratings[t], 2),
                  "adjustment": round(applied.get(t, 0.0), 2),
                  "games_observed": sol.games_observed.get(t, 0),
                  "std_error": round(sol.std_errors.get(t, float('nan')), 2),
                  "identified": sol.is_identified(t)}
                 for t in adjusted),
                key=lambda r: -r["rating"]),
        },
        "empirical_prior": prior_info,
        "divergence": sorted(
            ({"team": t, "market_minus_prior": round(d, 2)} for t, d in divergence.items()),
            key=lambda r: -abs(r["market_minus_prior"]))[:20],
        "unit_definition": {"unit_pct_of_bankroll": unit_pct, "bankroll": bankroll or None},
        "adjustment_coefficients": ADJUSTMENT_COEFFICIENTS,
        "picks": [], "skipped": [], "simulation": None, "notes": list(snap.notes),
    }

    if not schedule:
        result["simulation"] = {
            "available": False,
            "reason": ("No season schedule available from this source. Win-total edges require "
                       "simulating a full schedule; ratings above are still usable for game lines."),
        }
        result["skipped"] = [{"subject": w.team, "reason": "no schedule to simulate"}
                             for w in snap.win_totals]
        _write(result)
        return result

    sim = simulate_season(adjusted, schedule, sol.hfa, n_sims=n_sims,
                          rating_se=sol.std_errors)
    result["simulation"] = {
        "available": True, "n_sims": sim.n_sims, "n_games": sim.n_games,
        "margin_sd": sim.margin_sd, "warnings": sim.warnings,
    }

    ledger = Ledger(DATA)
    for wt in snap.win_totals:
        proj = sim.projections.get(wt.team)
        if proj is None:
            result["skipped"].append({"subject": wt.team, "reason": "not in simulated schedule"})
            continue
        if not sol.is_identified(wt.team):
            result["skipped"].append({
                "subject": wt.team,
                "reason": f"only {sol.games_observed.get(wt.team,0)} observed games "
                          f"(< {MIN_GAMES_FOR_CONFIDENCE}) — rating not identified"})
            continue

        p_over, p_under = proj.prob_over(wt.total), proj.prob_under(wt.total)
        side = "over" if p_over > p_under else "under"
        p_model = max(p_over, p_under)
        price = wt.over_price if side == "over" else wt.under_price
        opp = wt.under_price if side == "over" else wt.over_price

        er = evaluate(wt.team, "win_total", side, wt.total, p_model, price, opp)

        ev_map = _evidence_for(wt.team, personnel, sentiment)
        ind = assess(ev_map) if ev_map else assess({})
        # The simulation is itself an independent quantitative line of evidence.
        eff = ind.effective_threads + 1.0
        tier, tier_reason = confidence_tier(
            eff, contradicted=False,
            price_verified=(price is not None and opp is not None))

        pick = {
            "subject": wt.team, "market": "win_total", "side": side, "line": wt.total,
            "book": wt.book, "il_retail_only": wt.il_retail_only,
            "projected_wins": round(proj.mean_wins, 2),
            "p10_p90": [proj.p10, proj.p90],
            "tier": tier, "tier_reason": tier_reason,
            "effective_threads": round(eff, 2),
            "independence": ind.as_dict(),
            "edge": er.as_dict(),
            "units": round((er.stake_fraction or 0.0) / unit_pct, 2),
            "stake_dollars": (round(bankroll * (er.stake_fraction or 0.0), 2)
                              if bankroll else None),
        }
        result["picks"].append(pick)

        if er.bettable:
            ledger.record(LedgerEntry(
                pick_id=make_pick_id(snap.as_of, wt.team, "win_total", side, wt.total),
                placed_at=snap.as_of, subject=wt.team, market="win_total", side=side,
                line=wt.total, price_american=price, book=wt.book,
                model_prob=p_model, market_prob=er.market_prob, edge=er.edge,
                stake_fraction=er.stake_fraction, tier=tier, effective_threads=eff,
                threads_cited=ind.thread_names + ["simulation"],
                source_urls=[wt.source_url], synthetic=snap.synthetic,
            ))

    result["picks"].sort(key=lambda p: -(p["edge"]["edge"] or -1))
    settled, open_n = ledger.settled(), len(ledger.open_picks())
    result["calibration"] = build_report(settled, open_n).as_dict()
    _write(result)
    return result


def _write(result: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "model_picks.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(OUT, "model_picks.md"), "w") as fh:
        fh.write(render_markdown(result))


def render_markdown(r: dict) -> str:
    L: list[str] = []
    tag = " — SYNTHETIC VALIDATION RUN" if r.get("SYNTHETIC") else ""
    L.append(f"# Model output{tag}\n")
    L.append(f"> {r['disclaimer']}\n")
    if r.get("SYNTHETIC"):
        L.append("> ## ⚠ THIS IS SYNTHETIC DATA\n>\n"
                 "> Generated by `model/make_fixture.py` to validate the engine against known\n"
                 "> truth. The teams are invented. **Nothing here is a bet.** These picks are\n"
                 "> excluded from the calibration ledger.\n")
    L.append(f"Generated `{r['generated']}` · source `{r['source']}` · as of `{r['as_of']}`\n")

    rt = r["ratings"]
    L.append("## Ratings\n")
    L.append(f"{rt['n_teams']} teams from {rt['n_games']} games · HFA "
             f"{rt['hfa']}{'(fitted)' if rt['hfa_fitted'] else ' (default)'} · "
             f"ridge {rt['ridge']} · residual SD {rt['residual_sd']} · "
             f"effective dof {rt['effective_dof']}\n")
    for w in rt["warnings"]:
        L.append(f"- ⚠ {w}")
    if rt["warnings"]:
        L.append("")
    L.append("| Team | Rating | Market | Adj | Games | ±SE | Identified |")
    L.append("|---|---:|---:|---:|---:|---:|:--:|")
    for row in rt["table"][:25]:
        L.append(f"| {row['team']} | {row['rating']:+.2f} | {row['market_rating']:+.2f} | "
                 f"{row['adjustment']:+.2f} | {row['games_observed']} | {row['std_error']} | "
                 f"{'✓' if row['identified'] else '✗'} |")
    L.append("")

    sim = r.get("simulation") or {}
    if not sim.get("available"):
        L.append("## Simulation\n")
        L.append(f"**Unavailable.** {sim.get('reason','')}\n")
    else:
        L.append(f"## Simulation\n\n{sim['n_sims']:,} seasons × {sim['n_games']} games · "
                 f"margin SD {sim['margin_sd']}\n")
        for w in sim.get("warnings", []):
            L.append(f"- ⚠ {w}")
        L.append("")

    picks = r.get("picks", [])
    if picks:
        L.append("## Sized bets\n")
        L.append("| Subject | Bet | Proj | Model p | Mkt p | Edge | Stake | Tier | Eff. threads |")
        L.append("|---|---|---:|---:|---:|---:|---:|---|---:|")
        for p in picks:
            e = p["edge"]
            mp = f"{e['market_prob']:.1%}" if e["market_prob"] is not None else "—"
            ed = f"{e['edge']:+.1%}" if e["edge"] is not None else "—"
            st = f"{e['stake_fraction']:.2%}" if e.get("stake_fraction") else "—"
            flag = "" if e["bettable"] else " ⃠"
            L.append(f"| {p['subject']}{flag} | {p['side'].upper()} {p['line']} | "
                     f"{p['projected_wins']} | {e['model_prob']:.1%} | {mp} | {ed} | {st} | "
                     f"{p['tier']} | {p['effective_threads']} |")
        L.append("\n⃠ = not bettable (see reason in JSON)\n")

        actionable = [p for p in picks if p["edge"]["bettable"]]
        if actionable:
            L.append("### Why each bet, and how independent the evidence is\n")
            for p in actionable:
                ind = p["independence"]
                L.append(f"**{p['subject']} {p['side'].upper()} {p['line']}** — "
                         f"{p['edge']['reason']}. Tier *{p['tier']}*: {p['tier_reason']}.")
                if ind["discount_pct"] > 0:
                    L.append(f"  - Corroboration discounted {ind['discount_pct']}% "
                             f"(shared sources: {', '.join(ind['shared_domains']) or 'none'})")
                L.append("")

    if r.get("skipped"):
        L.append("## Skipped\n")
        L.append("| Subject | Reason |")
        L.append("|---|---|")
        for s in r["skipped"][:40]:
            L.append(f"| {s['subject']} | {s['reason']} |")
        L.append("")

    cal = r.get("calibration")
    if cal:
        L.append("## Calibration\n")
        L.append(f"{cal['verdict']}\n")
        L.append(f"- Settled picks: **{cal['n_settled']}** · open: {cal['n_open']}")
        if cal.get("brier") is not None:
            L.append(f"- Brier {cal['brier']} vs baseline {cal['brier_baseline']} "
                     f"(skill {cal['skill_score']})")
        if cal.get("clv", {}).get("n"):
            L.append(f"- CLV: {cal['clv']['pct_beating_close']:.0%} of picks beat the close")
        for w in cal.get("warnings", []):
            L.append(f"- ⚠ {w}")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="CFB model runner")
    ap.add_argument("--source", default="fixture",
                    choices=["fixture", "odds_api", "real-market"])
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--bankroll", type=float, default=0.0,
                    help="bankroll in dollars; prints per-bet dollar stakes")
    ap.add_argument("--unit-pct", type=float, default=UNIT_PCT,
                    help="fraction of bankroll per unit (default 0.01 = 1%%)")
    ap.add_argument("--no-prior", action="store_true",
                    help="skip the results-based prior (market-only ratings)")
    ap.add_argument("--calibrate", action="store_true", help="calibration report only")
    args = ap.parse_args()

    if args.calibrate:
        led = Ledger(DATA)
        rep = build_report(led.settled(), len(led.open_picks()))
        print(json.dumps(rep.as_dict(), indent=2))
        return

    r = run(source=args.source, n_sims=args.sims, bankroll=args.bankroll,
            unit_pct=args.unit_pct, use_prior=not args.no_prior)
    tag = "SYNTHETIC " if r.get("SYNTHETIC") else ""
    bettable = sum(1 for p in r.get("picks", []) if p["edge"]["bettable"])
    print(f"{tag}run complete: {len(r.get('picks', []))} evaluated, {bettable} bettable, "
          f"{len(r.get('skipped', []))} skipped -> output/model_picks.{{json,md}}")


if __name__ == "__main__":
    main()
