"""
src/crawl/crawler.py
Web crawler for the Space Exploration Knowledge Graph project.

Responsibilities:
- Fetch seed URLs using trafilatura (handles JS-light pages well)
- Respect robots.txt via urllib.robotparser
- Apply a polite delay between requests
- Filter out low-quality pages (< MIN_WORDS words)
- Persist results in JSONL format (one JSON object per line)

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
import trafilatura


# Configuration

SEED_URLS: list[str] = [
    # Wikipedia – broad overviews
    "https://en.wikipedia.org/wiki/Space_exploration",
    "https://en.wikipedia.org/wiki/Astronaut",
    "https://en.wikipedia.org/wiki/NASA",
    "https://en.wikipedia.org/wiki/European_Space_Agency",
    "https://en.wikipedia.org/wiki/SpaceX",
    "https://en.wikipedia.org/wiki/Roscosmos",
    "https://en.wikipedia.org/wiki/International_Space_Station",
    "https://en.wikipedia.org/wiki/Moon_landing",
    "https://en.wikipedia.org/wiki/Mars_exploration",
    "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
    "https://en.wikipedia.org/wiki/Hubble_Space_Telescope",
    "https://en.wikipedia.org/wiki/Orbital_mechanics",
    "https://en.wikipedia.org/wiki/Space_launch",
    "https://en.wikipedia.org/wiki/Rocket",
    "https://en.wikipedia.org/wiki/Satellite",
    "https://en.wikipedia.org/wiki/Space_policy",
    "https://en.wikipedia.org/wiki/Space_law",
    "https://en.wikipedia.org/wiki/Space_station",
    "https://en.wikipedia.org/wiki/Exoplanet",
    "https://en.wikipedia.org/wiki/Black_hole",
]

OUTPUT_FILE = Path("data/crawler_output.jsonl")
MIN_WORDS = 500          # to discard pages with fewer words
REQUEST_DELAY = 1.5      # seconds between requests (polite crawling)
USER_AGENT = "KnowledgeGraphBot/1.0 (educational project; +mailto:estelle.sobesky@edu.devinci.fr)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# robots.txt helpers (as required in TD1 but not necessary)

_robots_cache: dict[str, RobotFileParser] = {}


def _get_robots(base_url: str) -> RobotFileParser:
    """Return a cached RobotFileParser for *base_url*."""
    if base_url not in _robots_cache:
        rp = RobotFileParser()
        robots_url = f"{base_url}/robots.txt"
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception as exc:
            log.warning("Could not fetch %s (%s) – assuming allowed.", robots_url, exc)
        _robots_cache[base_url] = rp
    return _robots_cache[base_url]


def is_allowed(url: str, user_agent: str = USER_AGENT) -> bool:
    """Return True if *user_agent* is allowed to fetch *url* per robots.txt."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = _get_robots(base)
    return rp.can_fetch(user_agent, url)


# Fetch and clean data

def fetch_and_clean(url: str) -> str | None:
    """
    Download *url* and extract its main textual content.

    Returns the cleaned text or None if fetching / extraction failed
    or the page has fewer than MIN_WORDS words.
    """
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

    if not text:
        log.warning("Extraction returned nothing: %s", url)
        return None

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        log.info("Too short (%d words): %s", word_count, url)
        return None

    return text


def is_useful(text: str, min_words: int) -> bool: 
    """Quality gate: True when the text has at least MIN_WORDS words."""
    return bool(text) and len(text.split()) >= min_words


# Main crawl loop

def crawl(
    seed_urls: list[str] = SEED_URLS,
    output_file: Path = OUTPUT_FILE,
    min_words: int = MIN_WORDS,
    delay: float = REQUEST_DELAY,
) -> list[dict]:
    """
    Crawl *seed_urls*, filter, and write results to *output_file* (JSONL).

    Returns the list of saved records.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []

    with open(output_file, "w", encoding="utf-8") as fout:
        for url in seed_urls:
            log.info("→ Fetching: %s", url)

            # robots.txt check
            if not is_allowed(url):
                log.warning("  ✗ Disallowed by robots.txt: %s", url)
                time.sleep(delay)
                continue

            text = fetch_and_clean(url)

            if text and is_useful(text, min_words):

                word_count = len(text.split())
                record = {
                    "url": url,
                    "word_count": word_count,
                    "text": text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved.append(record)
                log.info("Saved  (%d words)", word_count)
            
            else:
                log.warning("Page rejected: too short or empty.")

            time.sleep(delay)

    log.info("Crawl complete. %d / %d pages saved to %s", len(saved), len(seed_urls), output_file)
    return saved


# JSONL loader (reusable by other modules afterwards)

def load_jsonl(path: Path | str) -> list[dict]:
    """Load a JSONL file and return a list of records."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Entry-point

if __name__ == "__main__":
    crawl()
