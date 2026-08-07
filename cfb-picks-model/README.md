# CFB Picks Model — 2026 Season

Not financial advice. Research tooling for entertainment and analysis.

## Architecture

Four independent research threads write structured JSON. A synthesis pass reads **only that
JSON** and produces the picks sheet. The separation is deliberate: research and judgment never
happen in the same step, so a pick can always be traced back to the record that justified it.

```
Phase 0  scope check ─────────────────► data/phase0_scope.json
                │
Phase 1  ┌──────┴──────┬─────────────┬──────────────┐
         │             │             │              │
      MARKET      PERSONNEL     SENTIMENT     HISTORICAL
         │             │             │              │
    market.json  personnel.json sentiment.json historical.json
         │             │             │              │
Phase 2  └──────┬──────┴─────────────┴──────────────┘
                ▼
           SYNTHESIS ──► output/picks.md + output/picks.json
```

## Thread responsibilities

| Thread | Owns | Never does |
|---|---|---|
| **Market** | Every number, line, price, book attribution, disagreement flags | Opinions on teams |
| **Personnel** | Coaching changes, staff continuity, portal by position, returning production | Quotes a betting line |
| **Sentiment** | Camp reports, injuries, suspensions, narrative-vs-reality | Invents an injury (the cardinal sin) |
| **Historical** | Base rates, situational factors, market biases | Passes structural reasoning off as data |
| **Synthesis** | Combining, ranking, confidence labels | Re-runs research |

## Rules that make the output trustworthy

1. **Market is the sole price authority.** Synthesis prints numbers verbatim from
   `market.json` with the book attached. No number is ever recalled from memory.
2. **Two-thread alignment for high confidence.** A pick must name which two threads support
   it and how. A price alone is not a reason.
3. **Confidence inheritance.** A pick can never be more confident than its weakest supporting
   record. A thesis resting on a `rumor`-tagged report is Speculative, full stop.
4. **Empirical vs structural.** Sourced statistics and derived reasoning are tagged
   differently and never conflated.
5. **Omission over invention.** Every thread has a `gaps` array. An honest gap is a feature;
   a fabricated line, injury, or ATS record is a disqualifying failure.

## Layout

```
agents/     thread instructions (market, personnel, sentiment, historical, synthesis)
data/       phase 0 scope + the four thread JSON outputs
output/     picks.md and picks.json
```

## Jurisdiction

Built for **Illinois**. Online books: bet365, BetMGM, BetRivers, Caesars, Circa Sports,
DraftKings, Fanatics, FanDuel, Hard Rock Bet, theScore Bet.

Illinois-based college teams (Illinois, Northwestern) are **in-person / retail only** — those
picks are tagged `il_retail_only` and are not available on online apps. Illinois also restricts
college player props.

## Re-running

The thread specs in `agents/` are the durable artifact. Point four agents at them, refresh the
four JSON files, re-run synthesis. Lines move weekly — the sheet is a snapshot, dated in
`generated`.
