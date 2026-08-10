# CFB Picks Model — 2026 Season

Not financial advice. Research tooling for entertainment and analysis.

Two layers. A **research pipeline** (four agents producing structured JSON) feeds a
**quantitative engine** (ratings → season simulation → sized bets → calibration ledger).
The research layer supplies judgment; the engine turns judgment into numbers you can
size a bet with and later prove wrong.

```
Phase 0  scope check ─────────────────► data/phase0_scope.json
             │
Phase 1  ┌───┴────┬──────────┬───────────┐
      MARKET  PERSONNEL  SENTIMENT  HISTORICAL      ← agents/*.md
         │        │          │           │
    market   personnel  sentiment  historical .json
         │        │          │           │
Phase 2  └───┬────┴──────────┴───────────┘
             ▼
        SYNTHESIS ──────────────────────► output/picks.md
             │
Phase 3      ▼   ENGINE (model/)
     ratings.py    solve power ratings from posted spreads
     simulate.py   Monte Carlo the season → win distributions
     edge.py       de-vig → EV → Kelly stake
     independence  discount correlated "corroboration"
     ledger.py     append-only record of every bet
     calibrate.py  Brier, calibration curve, CLV
             ▼
     output/model_picks.{md,json}
```

## Running it

```bash
python3 -m model.make_fixture                 # regenerate the synthetic league
python3 -m model.run --source fixture         # validation run (synthetic, safe)
python3 -m model.run --source real-market     # ratings from data/market.json
python3 -m model.run --source odds_api        # live (needs egress + ODDS_API_KEY)
python3 -m model.run --calibrate              # calibration report only
python3 tests/test_model.py                   # 23 tests
```

## How the engine works

**Ratings** are solved from posted spreads by ridge-regularized least squares:
`home_margin = r_home − r_away + HFA`. This has a consequence worth stating plainly —
a market-derived rating *cannot by itself disagree with the market*. All edge comes
from the adjustment layer, where thread scores convert to explicit points of spread
(`net_personnel` 1.0 pt/unit, `sentiment` 0.75 pt/unit, capped at ±4). Those
coefficients are **documented priors, not fitted values**; `calibrate.py` exists to
eventually replace them with measured ones.

**Simulation** draws each game from `Φ(margin / 16.5)` across 20,000 seasons, and
redraws ratings from `N(rating, SE)` each season so the output reflects estimation
uncertainty rather than pretending the ratings are exact.

**Sizing** de-vigs the market price, compares it to the model probability, and stakes
quarter-Kelly capped at 5% of bankroll. Below a 3% edge, nothing is bet.

## What the three fixes actually changed

| Weakness | Fix | Where |
|---|---|---|
| No edge quantification — every pick implicitly the same size | Season simulation → win distribution → de-vigged edge → Kelly stake | `simulate.py`, `edge.py` |
| "Two threads agree" counted correlated sources as independent | Overlap-discounted `effective_threads`; identical URLs across threads collapse toward one | `independence.py` |
| Confidence labels were unfalsifiable | Append-only ledger with the probability that justified each bet; Brier, calibration curve, CLV, per-tier hit rates | `ledger.py`, `calibrate.py` |

The independence fix is the sharpest. The old sheet had sentiment and personnel both
citing *the same ESPN article* on Brendan Sorsby and scored that as two-thread
confirmation. Now it scores 1.15 effective threads and drops out of high confidence.
That exact case is pinned in `tests/test_model.py::test_shared_source_discounts_corroboration`.

## Validation

The engine is tested against a synthetic league where true ratings are known, the book's
beliefs are wrong for exactly three teams, and the qualitative threads flag those three.

- Ratings recover the book's ratings at **r = 0.9998**, mean absolute error **0.5 pts**
- Simulation is unbiased: mean error **0.00 wins**, mean absolute error **0.046 wins**
- End to end: **3/3 planted book errors recovered, 0 false positives**

Four real bugs were found *by* this validation and fixed: a logistic/normal curve
mismatch that manufactured edges on 11 of 16 teams; ridge over-shrinking the rating
scale 16%; a fixture rounding bias that handed the UNDER free edge everywhere; and a
fitted HFA of 18.7 points on sparse real data, now guarded to a plausible band.

## Current blocker

**This session's egress policy blocks every live data host.** `api.the-odds-api.com`,
CollegeFootballData, ESPN, Wikipedia and Sports-Reference all return 403 at the proxy,
via both curl and WebFetch. The Odds API key in `.env` is fine — the *host* is denied.

Consequence: the engine runs, is tested, and is correct, but has **never been fed real
market data**. `--source real-market` (using the research threads' 11 sourced spreads)
correctly refuses to produce a single pick: 22 teams with 1 game each, all gated out as
unidentified, and no schedule to simulate. That is the right answer, not a failure.

To go live, allowlist in the environment's egress policy:

| Host | Gives you |
|---|---|
| `api.the-odds-api.com` | live odds, all books — the key is already in `.env` |
| `api.collegefootballdata.com` | schedules and historical results (free key required) |
| `site.api.espn.com` | schedules and scores, no key |

Only the first is strictly required for odds. A **schedule source is what unlocks the
simulator** — without a full-season schedule there are no win-total edges at all.

## Layout

```
agents/          research thread instructions (market, personnel, sentiment, historical, synthesis)
data/            phase 0 scope + four thread JSONs + picks_ledger.jsonl + fixtures/
model/           the engine
output/          picks.md (research sheet) + model_picks.md (engine output)
tests/           23 tests
.env             ODDS_API_KEY — gitignored, never committed
```

## Rules that keep the output honest

1. **Market is the sole price authority.** Numbers are printed verbatim with their book.
2. **Effective, not raw, thread agreement** gates confidence.
3. **Confidence inheritance** — a pick is never stronger than its weakest record.
4. **Empirical vs structural** are tagged separately and never conflated.
5. **Omission over invention.** Every thread carries a `gaps` array.
6. **Synthetic never contaminates real.** Fixture runs are stamped `SYNTHETIC` end to end
   and are excluded from the calibration ledger by construction.

## Honest limits

- Adjustment coefficients are unfitted priors. Until ~100 settled picks exist, the
  qualitative layer is an informed guess wearing a number.
- Margin SD is assumed 16.5. If the real value differs, every edge is biased — the
  fixture demonstrated exactly this failure mode.
- Ratings inherit the market's errors by construction. The model can only beat the
  market where the threads are right and the market is wrong.
- Nothing has been calibrated, because nothing has settled. Every confidence label is
  currently an untested assertion, and `calibrate.py` says so out loud.
