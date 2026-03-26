"""
src/crawl/crawler.py
Web crawler for the Sepsis Knowledge Graph project.

Three data sources:
1. PubMed API (Entrez) - scientific abstracts about sepsis (~200 words each)
2. PMC API (PubMed Central) - full text open access articles (~5000 words each)
3. Wikipedia API - general context pages about sepsis-related concepts

Outputs:
    data/samples/crawler_output.jsonl  (one JSON record per line)

Usage (standalone):
    python -m src.crawl.crawler
"""

from __future__ import annotations
import json
import logging
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

PUBMED_QUERIES = [
    # Original queries
    "sepsis diagnosis biomarkers",
    "sepsis treatment antibiotics",
    "septic shock pathophysiology",
    "sepsis bacteria pathogens",
    "sepsis organ failure",
    # Enriched queries
    "sepsis mortality prediction",
    #"sepsis pediatric children",
    "sepsis immunology cytokines",
    #"sepsis coagulation DIC",
    "sepsis ICU management",
    "sepsis machine learning",
    #"sepsis blood culture diagnosis",
    "sepsis inflammatory response",
    #"sepsis fluid therapy vasopressor",
    "sepsis procalcitonin lactate",
]

PUBMED_MAX_RESULTS = 100  # per query → ~1500 abstracts total

PMC_MAX_RESULTS = 25    # per query × 15 queries = ~375 full text articles

WIKIPEDIA_URLS: list[str] = [
    "https://en.wikipedia.org/wiki/Sepsis",
    "https://en.wikipedia.org/wiki/Septic_shock",
    "https://en.wikipedia.org/wiki/Procalcitonin",
    "https://en.wikipedia.org/wiki/Bacteremia",
    "https://en.wikipedia.org/wiki/Escherichia_coli",
    "https://en.wikipedia.org/wiki/Staphylococcus_aureus",
    "https://en.wikipedia.org/wiki/Streptococcus_pneumoniae",
    "https://en.wikipedia.org/wiki/Vancomycin",
    "https://en.wikipedia.org/wiki/Fluid_resuscitation",
    "https://en.wikipedia.org/wiki/Multiple_organ_dysfunction_syndrome",
]

OUTPUT_FILE = Path("data/samples/crawler_output.jsonl")
MIN_WORDS_ABSTRACT = 50
MIN_WORDS_FULLTEXT = 500
REQUEST_DELAY = 0.4
USER_AGENT = "KnowledgeGraphBot/1.0 (educational project; +mailto:estelle.sobesky@edu.devinci.fr)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source 1: PubMed API (abstracts)
# ---------------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = PUBMED_MAX_RESULTS) -> list[str]:
    """
    Search PubMed for *query* and return a list of PubMed IDs (PMIDs).
    Uses the ESearch endpoint.
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    response = requests.get(
        f"{PUBMED_BASE_URL}/esearch.fcgi",
        params=params,
        timeout=10,
    )
    response.raise_for_status()
    pmids = response.json()["esearchresult"]["idlist"]
    log.info("PubMed query '%s' → %d articles found", query, len(pmids))
    return pmids


def fetch_pubmed_abstracts(pmids: list[str]) -> list[dict]:
    """
    Fetch abstracts for a list of PubMed IDs using the EFetch endpoint.
    Returns a list of records with title, abstract, and source URL.
    """
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    response = requests.get(
        f"{PUBMED_BASE_URL}/efetch.fcgi",
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    records: list[dict] = []

    for article in root.findall(".//PubmedArticle"):
        title_el = article.find(".//ArticleTitle")
        title = title_el.text if title_el is not None else ""

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(
            (el.text or "") for el in abstract_parts if el.text
        ).strip()

        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else "unknown"

        if not abstract:
            continue

        text = f"{title}. {abstract}" if title else abstract

        if len(text.split()) < MIN_WORDS_ABSTRACT:
            continue

        records.append({
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "word_count": len(text.split()),
            "text": text,
            "source": "pubmed_abstract",
        })

    log.info("Fetched %d abstracts from PubMed", len(records))
    return records


def crawl_pubmed(
    queries: list[str] = PUBMED_QUERIES,
    max_results: int = PUBMED_MAX_RESULTS,
) -> list[dict]:
    """
    Run all PubMed queries, deduplicate by URL, return records.
    """
    seen_urls: set[str] = set()
    all_records: list[dict] = []

    for query in queries:
        pmids = search_pubmed(query, max_results)
        time.sleep(REQUEST_DELAY)

        records = fetch_pubmed_abstracts(pmids)
        time.sleep(REQUEST_DELAY)

        for rec in records:
            if rec["url"] not in seen_urls:
                seen_urls.add(rec["url"])
                all_records.append(rec)

    log.info("PubMed crawl done: %d unique abstracts", len(all_records))
    return all_records


# ---------------------------------------------------------------------------
# Source 2: PMC API (full text articles)
# ---------------------------------------------------------------------------

def search_pmc(query: str, max_results: int = PMC_MAX_RESULTS) -> list[str]:
    """
    Search PMC for *query* and return a list of PMC IDs.
    Only fetches open access articles via the open access filter.
    """
    params = {
        "db": "pmc",
        "term": f"{query} AND open access[filter]",
        "retmax": max_results,
        "retmode": "json",
    }
    response = requests.get(
        f"{PUBMED_BASE_URL}/esearch.fcgi",
        params=params,
        timeout=10,
    )
    response.raise_for_status()
    pmcids = response.json()["esearchresult"]["idlist"]
    log.info("PMC query '%s' → %d articles found", query, len(pmcids))
    return pmcids

def fetch_pmc_fulltext(pmcids: list[str]) -> list[dict]:
    """
    Fetch full text articles from PMC using the EFetch endpoint.
    Returns records with complete article text (title + abstract + body).
    """
    if not pmcids:
        return []

    params = {
        "db": "pmc",
        "id": ",".join(pmcids),
        "retmode": "xml",
        "rettype": "full",
    }
    response = requests.get(
        f"{PUBMED_BASE_URL}/efetch.fcgi",
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as e:
        log.warning("XML parse error for PMC articles: %s", e)
        return []

    records: list[dict] = []

    for article in root.findall(".//article"):

        # Extract PMC ID - try multiple possible locations in XML
        pmcid = None

        # Try 1: article-id with pub-id-type pmc
        pmcid_el = article.find(".//article-id[@pub-id-type='pmc']")
        if pmcid_el is not None:
            pmcid = pmcid_el.text

        # Try 2: article-id with pub-id-type pmcid
        if not pmcid:
            pmcid_el = article.find(".//article-id[@pub-id-type='pmcid']")
            if pmcid_el is not None:
                pmcid = pmcid_el.text

        # Try 3: front/article-meta/article-id
        if not pmcid:
            pmcid_el = article.find(".//front//article-id[@pub-id-type='pmc']")
            if pmcid_el is not None:
                pmcid = pmcid_el.text

        # Fallback: use index as unique ID
        if not pmcid:
            pmcid = f"idx_{len(records)}"

        # Extract title
        title_el = article.find(".//article-title")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        # Extract abstract
        abstract_parts = article.findall(".//abstract//p")
        abstract = " ".join(
            "".join(p.itertext()) for p in abstract_parts
        ).strip()

        # Extract full body text
        body_parts = article.findall(".//body//p")
        body = " ".join(
            "".join(p.itertext()) for p in body_parts
        ).strip()

        # Combine all text parts
        text_parts = [t for t in [title, abstract, body] if t]
        text = " ".join(text_parts).strip()

        if len(text.split()) < MIN_WORDS_FULLTEXT:
            log.info("PMC article too short, skipping: PMC%s", pmcid)
            continue

        records.append({
            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
            "word_count": len(text.split()),
            "text": text,
            "source": "pmc_fulltext",
        })
        log.info("PMC article saved: PMC%s (%d words)", pmcid, len(text.split()))

    log.info("Fetched %d full text articles from PMC", len(records))
    return records



def crawl_pmc(
    queries: list[str] = PUBMED_QUERIES,
    max_results: int = PMC_MAX_RESULTS,
) -> list[dict]:
    """
    Run all PMC queries, deduplicate by URL, return full text records.
    """
    seen_urls: set[str] = set()
    all_records: list[dict] = []

    for query in queries:
        log.info("Fetching PMC full text: '%s'", query)
        pmcids = search_pmc(query, max_results)
        time.sleep(REQUEST_DELAY)

        records = fetch_pmc_fulltext(pmcids)
        time.sleep(REQUEST_DELAY)

        for rec in records:
            if rec["url"] not in seen_urls:
                seen_urls.add(rec["url"])
                all_records.append(rec)

    log.info("PMC crawl done: %d unique full text articles", len(all_records))
    return all_records


# ---------------------------------------------------------------------------
# Source 3: Wikipedia (via trafilatura + robots.txt)
# ---------------------------------------------------------------------------

_robots_cache: dict[str, RobotFileParser] = {}


def _get_robots(base_url: str) -> RobotFileParser:
    """
    Return a cached RobotFileParser for *base_url*.
    Uses a proper User-Agent header to avoid 403 errors.
    """
    if base_url not in _robots_cache:
        rp = RobotFileParser()
        robots_url = f"{base_url}/robots.txt"
        rp.set_url(robots_url)
        try:
            req = urllib.request.Request(
                robots_url,
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(req) as response:
                content = response.read().decode("utf-8")
                rp.parse(content.splitlines())
        except Exception as exc:
            log.warning(
                "Could not fetch %s (%s) – assuming allowed.",
                robots_url,
                exc,
            )
        _robots_cache[base_url] = rp
    return _robots_cache[base_url]


def is_allowed(url: str, user_agent: str = USER_AGENT) -> bool:
    """Return True if *user_agent* is allowed to fetch *url* per robots.txt."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return _get_robots(base).can_fetch(user_agent, url)


def fetch_wikipedia_page(url: str) -> dict | None:
    """
    Fetch a Wikipedia page using trafilatura.
    Returns a record dict or None if the page is too short or failed.
    """
    if not is_allowed(url):
        log.warning("Disallowed by robots.txt: %s", url)
        return None

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        log.warning("Download failed: %s", url)
        return None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_recall=True,
    )

    if not text or len(text.split()) < MIN_WORDS_FULLTEXT:
        log.warning("Too short or empty: %s", url)
        return None

    return {
        "url": url,
        "word_count": len(text.split()),
        "text": text,
        "source": "wikipedia",
    }


def crawl_wikipedia(
    seed_urls: list[str] = WIKIPEDIA_URLS,
    delay: float = REQUEST_DELAY,
) -> list[dict]:
    """
    Crawl Wikipedia seed URLs and return valid records.
    """
    records: list[dict] = []
    for url in seed_urls:
        log.info("Fetching Wikipedia: %s", url)
        record = fetch_wikipedia_page(url)
        if record:
            records.append(record)
            log.info("Saved (%d words): %s", record["word_count"], url)
        time.sleep(delay)

    log.info("Wikipedia crawl done: %d pages saved", len(records))
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def crawl(output_file: Path = OUTPUT_FILE) -> list[dict]:
    """
    Run all three crawlers, merge results, save to *output_file* (JSONL).
    Returns the full list of saved records.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log.info("=== Starting PubMed abstracts crawl ===")
    pubmed_records = crawl_pubmed()

    log.info("=== Starting PMC full text crawl ===")
    pmc_records = crawl_pmc()

    log.info("=== Starting Wikipedia crawl ===")
    wiki_records = crawl_wikipedia()

    all_records = pubmed_records + pmc_records + wiki_records
    log.info("=== Total: %d records ===", len(all_records))
    log.info("  PubMed abstracts : %d", len(pubmed_records))
    log.info("  PMC full text    : %d", len(pmc_records))
    log.info("  Wikipedia        : %d", len(wiki_records))

    with open(output_file, "w", encoding="utf-8") as fout:
        for record in all_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Saved to %s", output_file)
    return all_records


# ---------------------------------------------------------------------------
# JSONL loader (reused by other modules)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path | str) -> list[dict]:
    """Load a JSONL file and return a list of records."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    crawl()