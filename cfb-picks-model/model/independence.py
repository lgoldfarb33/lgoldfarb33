"""Source-independence accounting.

This is the fix for weakness #2. The original sheet's alignment rule counted
threads: "two threads agree, therefore high confidence." But the threads read
overlapping sources. Sentiment and personnel both cited the same ESPN article on
Brendan Sorsby and then "independently agreed" — one source wearing two hats,
scored as corroboration.

Here, agreement is discounted by how much the supporting threads' evidence
actually overlaps. Two threads citing the identical URL contribute roughly one
thread's worth of confirmation. Two threads citing different outlets contribute
two. The alignment rule then runs on `effective_threads` instead of a raw count.

Domain-level overlap is penalized at half weight: two different ESPN articles are
more independent than the same ESPN article twice, but less independent than ESPN
plus a local beat writer, because they share an editorial pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

DOMAIN_OVERLAP_WEIGHT = 0.5

# Aggregators restate other outlets' reporting. Two threads landing on the same
# aggregator is especially weak corroboration, so they carry extra penalty.
AGGREGATOR_DOMAINS = {
    "sportsbettingdime.com", "vegasinsider.com", "oddsshark.com", "covers.com",
    "teamrankings.com", "yardbarker.com", "sports.yahoo.com", "msn.com",
}


def normalize_url(url: str) -> str:
    if not url:
        return ""
    u = url.strip().lower().split("#")[0].rstrip("/")
    u = re.sub(r"[?&](utm_[^=]+|ref|src)=[^&]*", "", u)
    return u


def domain_of(url: str) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url if "://" in url else "https://" + url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return ""


def _jaccard(a: set, b: set) -> float:
    a, b = {x for x in a if x}, {x for x in b if x}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class IndependenceReport:
    raw_threads: int
    effective_threads: float
    thread_names: list[str]
    shared_urls: list[str] = field(default_factory=list)
    shared_domains: list[str] = field(default_factory=list)
    pair_overlaps: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def discount(self) -> float:
        if self.raw_threads == 0:
            return 0.0
        return 1.0 - (self.effective_threads / self.raw_threads)

    def as_dict(self) -> dict:
        return {
            "raw_threads": self.raw_threads,
            "effective_threads": round(self.effective_threads, 2),
            "discount_pct": round(self.discount * 100, 1),
            "thread_names": self.thread_names,
            "shared_urls": self.shared_urls,
            "shared_domains": self.shared_domains,
            "pair_overlaps": self.pair_overlaps,
            "notes": self.notes,
        }


def assess(evidence: dict[str, list[str]]) -> IndependenceReport:
    """Score how independent a set of supporting threads really is.

    `evidence` maps thread name -> list of source URLs that thread used for this
    specific claim. Returns effective thread count after overlap discounting.
    """
    threads = [t for t, urls in evidence.items() if urls]
    raw = len(threads)
    report = IndependenceReport(raw_threads=raw, effective_threads=float(raw),
                               thread_names=threads)
    if raw < 2:
        if raw == 1:
            report.notes.append("Single thread — no corroboration to discount")
        else:
            report.notes.append("No sourced evidence supplied")
        return report

    url_sets = {t: {normalize_url(u) for u in evidence[t] if u} for t in threads}
    dom_sets = {t: {domain_of(u) for u in evidence[t] if u} for t in threads}

    total_penalty = 0.0
    all_shared_urls: set[str] = set()
    all_shared_domains: set[str] = set()

    for i in range(raw):
        for j in range(i + 1, raw):
            t1, t2 = threads[i], threads[j]
            shared_u = url_sets[t1] & url_sets[t2]
            shared_d = (dom_sets[t1] & dom_sets[t2]) - {""}

            url_j = _jaccard(url_sets[t1], url_sets[t2])
            # Domain overlap beyond what the shared URLs already explain.
            dom_only = shared_d - {domain_of(u) for u in shared_u}
            dom_j = _jaccard(dom_sets[t1], dom_sets[t2]) if dom_only else 0.0

            penalty = url_j + DOMAIN_OVERLAP_WEIGHT * dom_j
            if shared_d & AGGREGATOR_DOMAINS:
                penalty += 0.15
                report.notes.append(
                    f"{t1}/{t2} share an aggregator domain — restated reporting, "
                    f"weak corroboration"
                )
            penalty = min(penalty, 1.0)
            total_penalty += penalty

            all_shared_urls |= shared_u
            all_shared_domains |= shared_d

            if penalty > 0:
                report.pair_overlaps.append({
                    "threads": [t1, t2],
                    "url_jaccard": round(url_j, 3),
                    "domain_jaccard": round(dom_j, 3),
                    "penalty": round(penalty, 3),
                    "shared_urls": sorted(shared_u),
                })

    report.effective_threads = max(1.0, raw - total_penalty)
    report.shared_urls = sorted(all_shared_urls)
    report.shared_domains = sorted(all_shared_domains)

    if report.discount > 0.25:
        report.notes.append(
            f"Corroboration discounted {report.discount:.0%}: the supporting threads "
            f"are substantially reading the same sources"
        )
    return report


# Confidence tiers now key off effective threads, not raw agreement.
def confidence_tier(
    effective_threads: float,
    contradicted: bool = False,
    price_verified: bool = True,
    weakest_confidence: str = "verified",
) -> tuple[str, str]:
    """Return (tier, reason).

    Rules, in order — the first that fires wins:
      contradicted by any thread          -> speculative
      any supporting record is a rumor    -> speculative (confidence inheritance)
      effective >= 2 and price verified   -> high
      effective >= 2, price unverified    -> medium
      effective >= 1.5                    -> medium
      otherwise                           -> speculative
    """
    if contradicted:
        return "speculative", "a thread materially contradicts this pick"
    if weakest_confidence == "rumor":
        return "speculative", "rests on a rumor-tagged record (confidence inheritance)"
    if effective_threads >= 2.0:
        if price_verified:
            return "high", f"{effective_threads:.1f} effective independent threads, price verified"
        return "medium", f"{effective_threads:.1f} effective threads but the price is unverified"
    if effective_threads >= 1.5:
        return "medium", f"only {effective_threads:.1f} effective threads after overlap discount"
    return "speculative", f"{effective_threads:.1f} effective threads — no real corroboration"
