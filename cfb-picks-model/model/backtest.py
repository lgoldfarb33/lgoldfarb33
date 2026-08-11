"""Walk-forward backtest against the real 2025 season.

This is the test the rest of the engine has never had: does the architecture
(empirical prior + market blend) actually produce value on a season nobody at
build time had seen the outcome of? Everything before this module was either
synthetic validation (known-truth, but invented teams) or a snapshot of 11
real spreads with no season to simulate against. This uses real games, real
closing lines, and real final scores.

Data: sportsdataverse/cfbfastR-data via raw.githubusercontent.com — schedules
(with final scores) and betting/csv/cfb_line_odds.csv.gz (spreads by book,
2018-2025). No API key, no odds-api dependency. The betting file carries
lines but no prices, so ATS results below assume standard -110 both sides —
stated once here, not repeated at every number.

Method, walk-forward, no lookahead:
  - empirical prior: fit ONCE on 2022-2024 only. 2025 never touches it.
  - for each regular-season week W of 2025 (W >= MIN_WARMUP_WEEK):
      - market ratings: solve_ratings() on every 2025 game strictly BEFORE
        week W, using that week's own closing spread as the "posted" input
        (exactly the shape solve_ratings expects from a live market snapshot)
      - blended = blend(market_ratings, prior.ratings, market_weight=w)
      - predict week W's games from `blended`; compare to week W's own
        closing line and to the actual final margin
  - repeat across a market_weight grid to see whether the fixed 0.75 the
    rest of the engine uses is actually a good choice, or whether the data
    prefers something else.

This is a backtest, not a live betting system: it does not touch the ledger,
does not produce picks, and every number it reports is about the PAST.
"""
from __future__ import annotations

import csv
import gzip
import os
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

from .empirical import (FBS_CONFERENCES, ResultGame, build_prior, fit_from_results,
                        load_results, blend)
from .ratings import solve_ratings
from .sources import Game

BETTING_URL = ("https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/"
              "main/betting/csv/cfb_line_odds.csv.gz")

# Standard vig assumption for the ATS test — the betting CSV carries lines but no
# prices for any book. -110/-110 is the overwhelmingly common real-world price for
# a spread; using anything else without evidence would be inventing precision this
# dataset doesn't support.
ASSUMED_ATS_PRICE = -110
BREAKEVEN_ATS_RATE = 110 / 210  # ~0.5238, the cover rate a -110 bet needs to break even

# Need enough within-season games before the market-side ratings are identified at
# all; predicting week 1-3 from zero in-season data would just be testing the prior
# alone, which is a different (also reported) question.
MIN_WARMUP_WEEK = 4


def download_betting_lines(data_dir: str, timeout: int = 60) -> str | None:
    path = os.path.join(data_dir, "raw", "cfb_line_odds.csv.gz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    try:
        with urllib.request.urlopen(BETTING_URL, timeout=timeout) as resp:
            body = resp.read()
        if len(body) < 1_000_000:
            return None
        with open(path, "wb") as fh:
            fh.write(body)
        return path
    except Exception:  # noqa: BLE001 — network/parse failure just means no backtest
        return None


@dataclass
class MarketGame:
    game_id: str
    season: int
    week: int
    season_type: str
    home_team: str
    away_team: str
    home_margin: float          # market's consensus closing line, home-margin convention
    n_books: int
    actual_margin: int | None = None   # filled in from schedules where completed
    opening_home_margin: float | None = None  # only populated when with_opening=True


def load_market_lines(betting_path: str, schedule_paths: list[str], season: int,
                      with_opening: bool = False) -> list[MarketGame]:
    """Join betting-CSV spreads to schedule results for one season.

    Each (game, book) pair contributes one row in the source file, `abbr` naming
    which team that book's `lines` value belongs to. Multiple books per game are
    collapsed to a median — a simple, defensible consensus given no timestamps are
    available to identify a true single closing snapshot.

    `with_opening` also collects each book's `opening_lines` the same way, storing
    the consensus opening home-margin on `MarketGame.opening_home_margin` (None
    where no book reported one — roughly 25% of 2025 spread rows).
    """
    sched_by_id: dict[str, dict] = {}
    for path in schedule_paths:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if int(float(r.get("season", 0))) != season:
                    continue
                sched_by_id[r["game_id"]] = r

    # game_id -> team_name -> {"close": [...], "open": [...]}
    raw: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"close": [], "open": []}))
    with gzip.open(betting_path, "rt") as fh:
        for row in csv.DictReader(fh):
            try:
                if float(row.get("season") or 0) != season:
                    continue
            except ValueError:
                continue
            if row.get("market_type") != "spread":
                continue
            gid = row.get("game_id")
            sched = sched_by_id.get(gid)
            if sched is None or sched.get("completed") != "TRUE":
                continue
            if (sched.get("home_conference") not in FBS_CONFERENCES
                    or sched.get("away_conference") not in FBS_CONFERENCES):
                continue
            team = row.get("abbr", "")
            try:
                raw[gid][team]["close"].append(float(row["lines"]))
            except (TypeError, ValueError):
                continue
            if with_opening and row.get("opening_lines"):
                try:
                    raw[gid][team]["open"].append(float(row["opening_lines"]))
                except ValueError:
                    pass

    def consensus_home_margin(by_team: dict, home: str, away: str, key: str) -> float | None:
        h, a = by_team.get(home, {}).get(key, []), by_team.get(away, {}).get(key, [])
        if not h and not a:
            return None
        # A team's own line is its handicap (negative = favored). Home-margin
        # convention flips the away side's sign so both describe the same market.
        return -median(list(h) + [-x for x in a])

    games: list[MarketGame] = []
    for gid, by_team in raw.items():
        sched = sched_by_id[gid]
        home, away = sched["home_team"], sched["away_team"]
        home_margin_market = consensus_home_margin(by_team, home, away, "close")
        if home_margin_market is None:
            continue
        opening_hm = (consensus_home_margin(by_team, home, away, "open")
                     if with_opening else None)
        n_books = len(by_team.get(home, {}).get("close", [])) + len(by_team.get(away, {}).get("close", []))
        try:
            hp, ap = int(sched["home_points"]), int(sched["away_points"])
            week = int(sched["week"])
        except (TypeError, ValueError):
            continue
        games.append(MarketGame(
            game_id=gid, season=season, week=week, season_type=sched.get("season_type", ""),
            home_team=home, away_team=away, home_margin=home_margin_market,
            n_books=n_books, actual_margin=hp - ap, opening_home_margin=opening_hm,
        ))
    return games


@dataclass
class LineMovementResult:
    n_games: int
    with_move_cover_rate: float | None
    with_move_ci: tuple[float, float] | None
    against_move_cover_rate: float | None
    breakeven: float = BREAKEVEN_ATS_RATE

    def as_dict(self) -> dict:
        return {
            "n_games": self.n_games,
            "with_move_cover_rate": round(self.with_move_cover_rate, 4) if self.with_move_cover_rate else None,
            "with_move_95ci": [round(x, 3) for x in self.with_move_ci] if self.with_move_ci else None,
            "against_move_cover_rate": round(self.against_move_cover_rate, 4) if self.against_move_cover_rate else None,
            "breakeven": round(self.breakeven, 4),
        }


def line_movement_signal(games: list[MarketGame], min_movement: float = 0.5) -> LineMovementResult:
    """Does the DIRECTION the closing line moved from open predict which side covers?

    Two competing hypotheses in betting literature: "follow the steam" (line moves
    because sharp money is right) vs. "fade the public" (line moves because casual
    money overreacts to a name or a storyline, and closes too far). This tests which,
    if either, shows up in 2025 — requires `games` loaded with `with_opening=True`.
    """
    import math
    with_move = against_move = 0
    for g in games:
        if g.opening_home_margin is None or g.actual_margin is None:
            continue
        movement = g.home_margin - g.opening_home_margin
        if abs(movement) < min_movement:
            continue
        cover_margin_home = g.actual_margin - g.home_margin
        if abs(cover_margin_home) < 1e-9:
            continue  # push
        home_covered = cover_margin_home > 0
        moved_toward_home = movement > 0
        if moved_toward_home == home_covered:
            with_move += 1
        else:
            against_move += 1

    n = with_move + against_move
    if n == 0:
        return LineMovementResult(n_games=0, with_move_cover_rate=None, with_move_ci=None,
                                  against_move_cover_rate=None)
    rate = with_move / n
    se = math.sqrt(rate * (1 - rate) / n)
    return LineMovementResult(
        n_games=n, with_move_cover_rate=rate,
        with_move_ci=(rate - 1.96 * se, rate + 1.96 * se),
        against_move_cover_rate=1.0 - rate,
    )


@dataclass
class WeekResult:
    week: int
    n_games: int
    market_weight: float
    blended_mae: float
    market_mae: float
    prior_only_mae: float
    ats_bets: int = 0
    ats_covers: int = 0
    ats_pushes: int = 0

    @property
    def ats_rate(self) -> float | None:
        settled = self.ats_bets - self.ats_pushes
        return (self.ats_covers / settled) if settled > 0 else None


@dataclass
class BacktestReport:
    season: int
    market_weight: float
    weeks: list[WeekResult] = field(default_factory=list)
    disagreement_threshold: float = 3.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        total_games = sum(w.n_games for w in self.weeks)
        total_bets = sum(w.ats_bets for w in self.weeks)
        total_covers = sum(w.ats_covers for w in self.weeks)
        total_pushes = sum(w.ats_pushes for w in self.weeks)
        settled = total_bets - total_pushes
        weighted = lambda attr: (  # noqa: E731
            sum(getattr(w, attr) * w.n_games for w in self.weeks) / total_games
            if total_games else None)
        return {
            "season": self.season, "market_weight": self.market_weight,
            "weeks_tested": len(self.weeks), "total_games": total_games,
            "mae_blended": weighted("blended_mae"), "mae_market_only": weighted("market_mae"),
            "mae_prior_only": weighted("prior_only_mae"),
            "ats_bets": total_bets, "ats_covers": total_covers, "ats_pushes": total_pushes,
            "ats_cover_rate": (total_covers / settled) if settled > 0 else None,
            "breakeven_rate": BREAKEVEN_ATS_RATE,
            "disagreement_threshold_pts": self.disagreement_threshold,
            "warnings": self.warnings,
        }


def run_backtest(
    data_dir: str,
    season: int = 2025,
    market_weight: float = 0.75,
    disagreement_threshold: float = 3.0,
    min_warmup_week: int = MIN_WARMUP_WEEK,
) -> BacktestReport | None:
    """Walk forward through one season's regular-season weeks, no lookahead."""
    raw_dir = os.path.join(data_dir, "raw")
    betting_path = download_betting_lines(data_dir)
    if betting_path is None:
        return None

    prior_years = tuple(y for y in (2022, 2023, 2024, 2025) if y < season) or (2022, 2023, 2024)
    schedule_paths = []
    for y in set(prior_years) | {season}:
        p = os.path.join(raw_dir, f"sched_{y}.csv")
        if not os.path.exists(p):
            from .empirical import download_seasons
            got = download_seasons([y], raw_dir)
            if y not in got:
                continue
        schedule_paths.append(p)

    prior_games = load_results([p for p in schedule_paths if f"_{season}." not in p])
    if not prior_games:
        return None
    prior_fit = fit_from_results(prior_games, decay=1.0)  # flat weight; every prior year is "the past"

    market_games = load_market_lines(betting_path, schedule_paths, season)
    if not market_games:
        return None

    weeks = sorted({g.week for g in market_games if g.season_type == "regular"})
    report = BacktestReport(season=season, market_weight=market_weight,
                            disagreement_threshold=disagreement_threshold)

    for w in weeks:
        if w < min_warmup_week:
            continue
        train = [g for g in market_games if g.season_type == "regular" and g.week < w]
        test = [g for g in market_games if g.season_type == "regular" and g.week == w]
        if len(train) < 20 or not test:
            continue

        train_spreads = [Game(home=g.home_team, away=g.away_team, home_margin=g.home_margin)
                         for g in train]
        try:
            sol = solve_ratings(train_spreads)
        except ValueError:
            continue

        blended, _ = blend(sol.ratings, prior_fit.ratings, market_weight=market_weight)

        errs_blend, errs_market, errs_prior = [], [], []
        bets = covers = pushes = 0
        for g in test:
            if g.home_team not in blended or g.away_team not in blended:
                continue
            pred = blended[g.home_team] - blended[g.away_team] + sol.hfa
            errs_blend.append(abs(pred - g.actual_margin))
            errs_market.append(abs(g.home_margin - g.actual_margin))
            if g.home_team in prior_fit.ratings and g.away_team in prior_fit.ratings:
                pred_prior = (prior_fit.ratings[g.home_team] - prior_fit.ratings[g.away_team]
                             + prior_fit.hfa)
                errs_prior.append(abs(pred_prior - g.actual_margin))

            disagreement = pred - g.home_margin
            if abs(disagreement) < disagreement_threshold:
                continue
            bets += 1
            # Model likes the home side more than the market does -> bet home
            # against the market's own line; result is home_actual_margin vs
            # -market_home_margin (the spread home must beat to cover).
            if disagreement > 0:
                cover_margin = g.actual_margin - g.home_margin
            else:
                cover_margin = g.home_margin - g.actual_margin  # betting away
            if abs(cover_margin) < 1e-9:
                pushes += 1
            elif cover_margin > 0:
                covers += 1

        if not errs_blend:
            continue
        report.weeks.append(WeekResult(
            week=w, n_games=len(errs_blend), market_weight=market_weight,
            blended_mae=sum(errs_blend) / len(errs_blend),
            market_mae=sum(errs_market) / len(errs_market),
            prior_only_mae=(sum(errs_prior) / len(errs_prior)) if errs_prior else float("nan"),
            ats_bets=bets, ats_covers=covers, ats_pushes=pushes,
        ))

    if not report.weeks:
        report.warnings.append("No weeks produced enough training data — season too sparse")
    return report


def sweep_market_weight(data_dir: str, season: int = 2025,
                        weights=(0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0)) -> list[dict]:
    """Which market_weight actually minimizes prediction error on this season?

    Answers the question a prior review raised: is the engine's fixed 0.75 a good
    choice, or arbitrary? 0.0 = prior only, 1.0 = market only (the ratings.py
    behavior before the empirical prior existed at all).
    """
    out = []
    for w in weights:
        rep = run_backtest(data_dir, season=season, market_weight=w)
        if rep is None:
            continue
        s = rep.summary()
        out.append({"market_weight": w, "mae_blended": s["mae_blended"],
                    "ats_cover_rate": s["ats_cover_rate"], "ats_bets": s["ats_bets"]})
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Walk-forward backtest against a real season")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--weight", type=float, default=None,
                    help="single market_weight to test; omit to sweep the standard grid")
    ap.add_argument("--movement", action="store_true",
                    help="also report the line-movement (follow vs fade) signal")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(here, "data")

    if args.weight is not None:
        rep = run_backtest(data_dir, season=args.season, market_weight=args.weight)
        print(json.dumps(rep.summary() if rep else {"error": "unavailable"}, indent=2))
    else:
        print(json.dumps(sweep_market_weight(data_dir, season=args.season), indent=2))

    if args.movement:
        betting_path = download_betting_lines(data_dir)
        sched_path = os.path.join(data_dir, "raw", f"sched_{args.season}.csv")
        if betting_path and os.path.exists(sched_path):
            games = load_market_lines(betting_path, [sched_path], season=args.season, with_opening=True)
            print(json.dumps(line_movement_signal(games).as_dict(), indent=2))
