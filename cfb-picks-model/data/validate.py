#!/usr/bin/env python3
"""Validate the four thread outputs before synthesis reads them.

Synthesis trusts these files completely, so this is the last place a malformed or
empty thread gets caught. Exits non-zero if any file is missing or unparseable.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# thread file -> the top-level arrays we expect to carry the payload
THREADS = {
    "market.json": ["championship_futures", "conference_futures", "win_totals",
                    "heisman_futures", "game_lines", "player_props", "disagreement_flags"],
    "personnel.json": ["coaching_changes", "coordinator_changes", "portal_impact",
                       "returning_production", "depth_chart_battles", "team_scores"],
    "sentiment.json": ["teams", "injuries", "suspensions_eligibility", "qb_situations"],
    "historical.json": ["new_coach_trends", "situational_factors", "market_biases"],
}


def count_sourced(records):
    """How many records carry a source_url / sources — the anti-fabrication check."""
    sourced = 0
    for r in records:
        if not isinstance(r, dict):
            continue
        if r.get("source_url") or r.get("sources"):
            sourced += 1
    return sourced


def main():
    failed = False
    for fname, keys in THREADS.items():
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"MISSING  {fname}")
            failed = True
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"INVALID  {fname}: {exc}")
            failed = True
            continue

        total = sourced = 0
        parts = []
        for key in keys:
            records = data.get(key, [])
            if not isinstance(records, list):
                continue
            total += len(records)
            sourced += count_sourced(records)
            if records:
                parts.append(f"{key}={len(records)}")

        gaps = data.get("gaps", [])
        print(f"OK       {fname}: {total} records, {sourced} sourced, {len(gaps)} gaps")
        print(f"         {', '.join(parts) if parts else 'NO RECORDS — thread returned empty'}")
        if total == 0:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
