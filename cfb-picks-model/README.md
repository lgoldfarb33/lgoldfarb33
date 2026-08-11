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

A second review pass (2026-08-10) found five more real bugs by the same discipline —
each independently reproduced before being fixed, not taken on faith: `RESULTS_RIDGE`
in `empirical.py` was an unvalidated guess (25.0) that shrank the results-prior's
rating scale ~60% below its cross-validation-optimal spread, which directly
contaminated the "measured" HFA/margin-SD constants reported in an earlier session;
bet side was chosen by raw probability instead of edge, silently rejecting real bets;
CLV compared a de-vigged open price against a raw closing price, inflating it by
~2.2pp per pick; unrated schedule opponents (typically FCS) were silently dropped from
the simulation instead of using a documented fallback; and the market/prior divergence
signal wasn't recentered on shared teams, baking in a fake offset. See `model/*.py`
docstrings for the full writeup and CV numbers behind each fix; `tests/test_model.py`
pins all five as regressions.

### Backtest against real 2024 and 2025 seasons

`model/backtest.py` walks forward through each season week-by-week — no lookahead —
using real closing spreads and final scores from the same reachable dataset
(`raw.githubusercontent.com`, no API key). Full writeup in
[`output/backtest_2025.json`](output/backtest_2025.json). Two findings, replicated
independently across both seasons:

1. **The static, 2022-2024-only results prior measurably worsens margin prediction
   the more weight it's given.** MAE decreases monotonically from weight=0.0 to
   weight=1.0 in both 2024 and 2025. This moved `MARKET_WEIGHT_DEFAULT` from an
   undefended 0.75 to a backtested **0.90**.
2. **No blend weight, and no bet-conviction threshold from 3 to 14 points, produced
   an ATS cover rate whose 95% confidence interval excluded the 52.4% breakeven
   rate.** There is no backtested edge in "market ratings plus old results" alone,
   at any configuration tested.

Neither finding undermines the project — it confirms the premise it was built on. A
lagged historical power rating competing against a market that reprices weekly on
live information has no reason to win, and didn't. **Real edge, if it exists, has to
come from the qualitative threads (personnel/sentiment) supplying current information
the market hasn't priced yet** — which this backtest structurally cannot evaluate,
since no dated thread output exists for past seasons. The only way to test *that*
layer is prospectively, through the ledger and `calibrate.py`, on picks made and
dated in real time going forward.

## Current blocker

**This session's egress policy blocks every live data host.** `api.the-odds-api.com`,
CollegeFootballData, ESPN, Wikipedia and Sports-Reference all return 403 at the proxy,
via both curl and WebFetch. The Odds API key in `.env` is fine — the *host* is denied.

Consequence: the engine runs, is tested, and is correct, but has **never been fed real
market data**. `--source real-market` (using the research threads' 11 sourced spreads)
correctly refuses to produce a single pick: 22 teams with 1 game each, all gated out as
unidentified, and no schedule to simulate. That is the right answer, not a failure.

### Why results work but odds do not

`raw.githubusercontent.com` is on the environment's **default Trusted list**, which is
exactly why the results-based prior works while the odds API does not. Nothing was
configured to make that happen — it was already allowed.

### How to allow the odds API

Network access is set per **cloud environment**, and there is no settings page or direct
URL for it — it lives in a selector inside the session UI.

1. Go to **claude.ai/code**
2. Click the **cloud icon showing the environment name** (e.g. `Default`) in the row
   just above the message box
3. Hover the environment and click the **gear icon** on its right
4. Set **Network access** to **Custom**
5. In **Allowed domains**, one per line:
   ```
   api.the-odds-api.com
   api.collegefootballdata.com
   site.api.espn.com
   ```
6. **Check "Also include default list of common package managers."** Without it you lose
   `raw.githubusercontent.com` and PyPI, which breaks the prior and the numpy/scipy
   install — you would trade one blocker for two.
7. Save.

| Host | Unlocks |
|---|---|
| `api.the-odds-api.com` | live odds across all books; key already in `.env` |
| `api.collegefootballdata.com` | **2026 schedules** (free key) — the other hard blocker |
| `site.api.espn.com` | schedules and scores, no key, useful as a fallback |

**Changes apply to new sessions only.** Running sessions never re-read environment
config, so start a fresh session afterward. Changing the allowed-host list also rebuilds
the environment cache, so the first new session is slower.

The odds API alone is not sufficient for win-total edges: it serves upcoming games, not
full-season schedules. **A schedule source is what unlocks the simulator** — and
cfbfastR-data has not published 2026 yet, which is why CFBD or ESPN matters.

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
