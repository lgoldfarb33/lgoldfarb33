"""Data adapters.

Two implementations behind one interface:

  OddsAPISource  — live, against the-odds-api.com v4
  FixtureSource  — offline, reads data/fixtures/, so the engine runs with no network

The seam exists because this session's egress policy blocks api.the-odds-api.com.
Everything downstream of `load_games()` / `load_win_totals()` is source-agnostic, so
switching to live data is a one-line config change, not a rewrite.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Literal

# CFB margin-of-victory standard deviation: how far actual margins scatter around
# the expected margin. The single most important constant in the simulator.
#
# MEASURED, not assumed — and re-measured once already. Fit on 3,132 FBS-vs-FBS
# games, 2022-2025, from sportsdataverse/cfbfastR-data. See model/empirical.py.
#
# The first measurement used ridge=25 for the underlying fit and reported 17.48.
# A Fable-run review (2026-08-10) and independent verification found that ridge
# value was an unvalidated guess that shrank the rating scale to ~40% of its
# cross-validation-optimal spread, and that shrinkage leaks directly into this
# number: less-confident (shrunk) ratings explain less of each game's margin, so
# more of it lands in the "unexplained" residual. 5-fold CV selected ridge=1.0
# (RESULTS_RIDGE in empirical.py); refitting at that ridge:
#   unweighted (ridge=1.0)         16.00
#   recency-weighted (ridge=1.0)   15.82   <- used
# Raw (team-strength-inclusive) margin SD is ~24.9; using that instead would
# flatten every win probability toward 50% and erase real edges. The original
# assumed value of 16.5 was, by coincidence, closer to this corrected number than
# the first "measured" one was.
MARGIN_SD_DEFAULT = 15.8

# Home advantage in points. Same games, same ridge correction:
#   unweighted (ridge=1.0)          2.83
#   recency-weighted (ridge=1.0)    3.21   <- used
# Per-season (ridge=1.0): 2022=2.26, 2023=2.80, 2024=3.46, 2025=3.57 — home
# advantage has drifted up in recent seasons, which is why the recency-weighted
# figure is used rather than the flat multi-season average.
# The ridge=25 fit reported 4.50 (unexplained team strength gets absorbed into HFA
# the same way it inflates the residual above). The originally assumed 2.5 was
# too low in the other direction — real HFA looks to be materially above that.
HFA_DEFAULT = 3.2


@dataclass
class Game:
    """One game, normalized so `home_margin` is always points the HOME team is favored by.

    Negative means the home team is the underdog. Neutral-site games carry
    neutral=True so the ratings solver knows not to credit home advantage.
    """
    home: str
    away: str
    home_margin: float | None = None
    total: float | None = None
    neutral: bool = False
    date: str = ""
    book: str = ""
    source_url: str = ""
    week: str = ""

    def key(self) -> str:
        return f"{self.date}|{self.away}@{self.home}"


@dataclass
class WinTotal:
    team: str
    total: float
    over_price: int | None = None   # American odds
    under_price: int | None = None
    book: str = ""
    source_url: str = ""
    il_retail_only: bool = False


@dataclass
class MarketSnapshot:
    """Everything the engine needs from the market, at one point in time."""
    as_of: str
    games: list[Game] = field(default_factory=list)
    win_totals: list[WinTotal] = field(default_factory=list)
    source: Literal["odds_api", "fixture"] = "fixture"
    synthetic: bool = False
    generated_with: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["games"] = [asdict(g) for g in self.games]
        d["win_totals"] = [asdict(w) for w in self.win_totals]
        return d


class SourceError(RuntimeError):
    pass


# ---------------------------------------------------------------- live source

class OddsAPISource:
    """Live client for the-odds-api.com v4.

    Written against the documented v4 contract. It has NOT been executed against
    the live host — this session's egress proxy returns 403 on CONNECT to
    api.the-odds-api.com — so treat the first live run as needing verification.
    Quota headers are surfaced because the free tier is small and silent
    exhaustion looks identical to "no games posted".
    """

    BASE = "https://api.the-odds-api.com/v4"
    SPORT = "americanfootball_ncaaf"

    def __init__(self, api_key: str | None = None, regions: str = "us", timeout: int = 20):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        if not self.api_key:
            raise SourceError("No ODDS_API_KEY set (expected in environment or .env)")
        self.regions = regions
        self.timeout = timeout
        self.quota_remaining: int | None = None
        self.quota_used: int | None = None

    def _get(self, path: str, **params) -> object:
        params["apiKey"] = self.api_key
        url = f"{self.BASE}{path}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                # Quota lives in headers, not the body — capture before parsing.
                self.quota_remaining = _int_or_none(resp.headers.get("x-requests-remaining"))
                self.quota_used = _int_or_none(resp.headers.get("x-requests-used"))
                return json.loads(resp.read().decode())
        except Exception as exc:  # urllib raises a zoo of types; caller wants one
            raise SourceError(f"Odds API request failed for {path}: {exc}") from exc

    def list_sports(self) -> object:
        return self._get("/sports/")

    def fetch(self, as_of: str, markets: str = "spreads,totals") -> MarketSnapshot:
        raw = self._get(
            f"/sports/{self.SPORT}/odds/",
            regions=self.regions,
            markets=markets,
            oddsFormat="american",
        )
        snap = MarketSnapshot(as_of=as_of, source="odds_api", synthetic=False)
        for ev in raw or []:
            game = self._parse_event(ev)
            if game:
                snap.games.append(game)
        snap.notes.append(
            f"Odds API: {len(snap.games)} games; quota used={self.quota_used} "
            f"remaining={self.quota_remaining}"
        )
        return snap

    def _parse_event(self, ev: dict) -> Game | None:
        home, away = ev.get("home_team"), ev.get("away_team")
        if not home or not away:
            return None
        best_margin = best_total = None
        book_name = ""
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") == "spreads" and best_margin is None:
                    for oc in mk.get("outcomes", []):
                        if oc.get("name") == home and oc.get("point") is not None:
                            # API gives the home team's handicap: -11 means home
                            # favored by 11. Flip the sign for home_margin.
                            best_margin = -float(oc["point"])
                            book_name = bk.get("title", "")
                elif mk.get("key") == "totals" and best_total is None:
                    for oc in mk.get("outcomes", []):
                        if oc.get("point") is not None:
                            best_total = float(oc["point"])
                            break
        return Game(
            home=home, away=away, home_margin=best_margin, total=best_total,
            date=(ev.get("commence_time") or "")[:10], book=book_name,
            source_url=f"{self.BASE}/sports/{self.SPORT}/odds/",
        )


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------- offline source

class FixtureSource:
    """Reads a MarketSnapshot from disk.

    Every snapshot it produces is stamped synthetic=True unless the fixture
    explicitly says otherwise, so fixture-derived output can never be mistaken
    for a real picks sheet.
    """

    def __init__(self, path: str):
        self.path = path

    def fetch(self, as_of: str | None = None) -> MarketSnapshot:
        with open(self.path) as fh:
            raw = json.load(fh)
        snap = MarketSnapshot(
            as_of=as_of or raw.get("as_of", ""),
            source="fixture",
            synthetic=bool(raw.get("synthetic", True)),
            generated_with=dict(raw.get("generated_with", {})),
            notes=list(raw.get("notes", [])),
        )
        snap.games = [Game(**g) for g in raw.get("games", [])]
        snap.win_totals = [WinTotal(**w) for w in raw.get("win_totals", [])]
        return snap


def load_source(kind: str, **kw):
    if kind == "odds_api":
        return OddsAPISource(**kw)
    if kind == "fixture":
        return FixtureSource(**kw)
    raise SourceError(f"Unknown source kind: {kind}")
