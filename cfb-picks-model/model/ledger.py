"""Append-only results ledger.

Half the fix for weakness #3. The old sheet produced picks and forgot them, so it
could never be shown to be wrong. Every pick now gets written here with the
probability that justified it, and settles later against a real outcome.

Append-only on purpose. A ledger you can quietly edit after the fact is a ledger
that will always show you were right.

Closing-line value is tracked alongside win/loss because over any realistic
sample, CLV is the better signal. Fifty picks of win/loss is mostly noise; fifty
picks of consistently beating the close is evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field

LEDGER_FILENAME = "picks_ledger.jsonl"


@dataclass
class LedgerEntry:
    pick_id: str
    placed_at: str          # snapshot date the pick came from
    subject: str
    market: str             # win_total | game_spread | futures | ...
    side: str               # over | under | team name
    line: float
    price_american: int | None
    book: str
    model_prob: float
    market_prob: float | None
    edge: float | None
    stake_fraction: float | None
    tier: str
    effective_threads: float
    threads_cited: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    synthetic: bool = False           # fixture-derived, excluded from calibration

    # populated at settlement
    settled: bool = False
    settled_at: str = ""
    result: str = ""                  # win | loss | push | void
    actual_value: float | None = None  # e.g. actual season wins
    closing_line: float | None = None
    closing_price_american: int | None = None
    clv_prob: float | None = None      # market_prob_at_close - market_prob_at_bet

    def as_dict(self) -> dict:
        return asdict(self)


def make_pick_id(placed_at: str, subject: str, market: str, side: str, line: float) -> str:
    raw = f"{placed_at}|{subject}|{market}|{side}|{line}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


class Ledger:
    def __init__(self, directory: str):
        self.path = os.path.join(directory, LEDGER_FILENAME)
        os.makedirs(directory, exist_ok=True)

    def all(self) -> list[LedgerEntry]:
        """Read the ledger, collapsing each pick_id to its most recent revision."""
        if not os.path.exists(self.path):
            return []
        latest: dict[str, LedgerEntry] = {}
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    latest[json.loads(line)["pick_id"]] = LedgerEntry(**json.loads(line))
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue  # a malformed line must not take the whole ledger down
        return list(latest.values())

    def _append(self, entry: LedgerEntry) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry.as_dict()) + "\n")

    def record(self, entry: LedgerEntry) -> LedgerEntry:
        """Write a new pick. Existing pick_ids are left alone, never overwritten."""
        if any(e.pick_id == entry.pick_id for e in self.all()):
            return entry
        self._append(entry)
        return entry

    def settle(
        self,
        pick_id: str,
        result: str,
        actual_value: float | None = None,
        closing_line: float | None = None,
        closing_price_american: int | None = None,
        settled_at: str = "",
    ) -> LedgerEntry | None:
        """Append a settled revision of an existing pick."""
        current = {e.pick_id: e for e in self.all()}.get(pick_id)
        if current is None:
            return None
        current.settled = True
        current.result = result
        current.actual_value = actual_value
        current.closing_line = closing_line
        current.closing_price_american = closing_price_american
        current.settled_at = settled_at

        if closing_price_american is not None and current.market_prob is not None:
            from .edge import american_to_implied
            current.clv_prob = american_to_implied(closing_price_american) - current.market_prob

        self._append(current)
        return current

    def settled(self) -> list[LedgerEntry]:
        """Settled, non-synthetic picks — the only ones calibration may use."""
        return [e for e in self.all() if e.settled and not e.synthetic and e.result in ("win", "loss")]

    def open_picks(self) -> list[LedgerEntry]:
        return [e for e in self.all() if not e.settled]
