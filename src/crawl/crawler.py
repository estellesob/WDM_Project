"""
Web crawler for the Sepsis knowledge graph project.

Two data sources:
1. PubMed API ("Entrez") - scientific abstracts about sepsis
2. Wikipedia - general context pages about sepsis-related concepts

Outputs:
    data/crawler_output.jsonl  (one JSON record per line)

Usage (standalone):
    python -m src.crawl.crawler
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import requests
import trafilatura



# Configuration


# PubMed API settings
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_QUERIES = ["sepsis diagnosis biomarkers","sepsis treatment antibiotics","septic shock pathophysiology","sepsis bacteria pathogens",
    "sepsis organ failure",]
PUBMED_MAX_RESULTS = 40  # per query : approximately 200 abstracts total

# Wikipedia seed urlss for general context
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
MIN_WORDS = 50          # lower threshold for abstracts (shorter than web pages)
REQUEST_DELAY = 0.4     # seconds between requests (PubMed allows 3 req/sec)
USER_AGENT = "KnowledgeGraphBot/1.0 (educational project)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# Source 1: PubMed API

def search_pubmed(query: str, max_results: int = PUBMED_MAX_RESULTS) -> list[str]:
    """
    Search PubMed for *query* and return a list of PubMed IDs (PMIDs).
    Uses the ESearch endpoint.
    """
    params = {"db": "pubmed","term": query,
        "retmax": max_results,"retmode": "json",
    }

    response = requests.get(f"{PUBMED_BASE_URL}/esearch.fcgi", params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    pmids = data["esearchresult"]["idlist"]
    log.info("PubMed query '%s' → %d articles found", query, len(pmids))
    return pmids


def fetch_pubmed_abstracts(pmids: list[str]) -> list[dict]:
    """
    Fetch abstracts for a list of PubMed IDs using the EFetch endpoint.
    Returns a list of records with title, abstract, and source URL.
    """
    if not pmids:
        return []

    # Fetch in one batch (comma separated IDs)
    params = {"db": "pubmed","id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    response = requests.get(f"{PUBMED_BASE_URL}/efetch.fcgi", params=params, timeout=30)
    response.raise_for_status()

    # Parse XML manually (no extra library needed)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.text)

    records: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        # Extract title
        title_el = article.find(".//ArticleTitle")
        title = title_el.text if title_el is not None else ""

        # Extract abstract (may have multiple AbstractText elements)
        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(
            (el.text or "") for el in abstract_parts if el.text
        ).strip()

        # Extract PMID for building the URL
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else "unknown"

        if not abstract:
            continue  # skip articles with no abstract

        text = f"{title}. {abstract}" if title else abstract

        records.append({"url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/","word_count": len(text.split()),
            "text": text,"source": "pubmed",})

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


# Source 2: Wikipedia (via trafilatura)

_robots_cache: dict[str, RobotFileParser] = {}


def _get_robots(base_url: str) -> RobotFileParser:
    """Return a cached RobotFileParser for *base_url*."""
    if base_url not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(f"{base_url}/robots.txt")
        try:
            rp.read()
        except Exception as exc:
            log.warning("Could not fetch robots.txt (%s) – assuming allowed.", exc)
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

    if not text or len(text.split()) < 500:
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


# Main pipeline


def crawl(output_file: Path = OUTPUT_FILE) -> list[dict]:
    """
    Run both crawlers, merge results, save to *output_file* (JSONL).
    Returns the full list of saved records.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log.info("=== Starting PubMed crawl ===")
    pubmed_records = crawl_pubmed()

    log.info("=== Starting Wikipedia crawl ===")
    wiki_records = crawl_wikipedia()

    all_records = pubmed_records + wiki_records
    log.info("=== Total: %d records ===", len(all_records))

    with open(output_file, "w", encoding="utf-8") as fout:
        for record in all_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Saved to %s", output_file)
    return all_records


# JSONL loader (reused by other modules)

def load_jsonl(path: Path | str) -> list[dict]:
    """Load a JSONL file and return a list of records."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records




if __name__ == "__main__":
    crawl()