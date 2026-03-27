"""
src/kg/alignment.py
===================
Entity alignment between the Sepsis KG and Wikidata + DBpedia.

Strategy (as recommended in the lab):
1. Wikidata Search API (wbsearchentities) - case insensitive
2. DBpedia Lookup API as fallback
3. If not found anywhere → entity stays local

Step 2 of the KB Construction lab:
- Entity linking with confidence scores
- owl:sameAs triples for aligned entities
- Mapping table: private entity | external URI | confidence

Usage (standalone):
    python -m src.kg.alignment
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from rdflib import Graph, Literal, Namespace, OWL, URIRef
from rdflib.namespace import XSD

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENTITIES_FILE = Path("data/samples/extracted_knowledge.csv")
ALIGNMENT_FILE = Path("kg_artifacts/alignment.ttl")
MAPPING_FILE = Path("kg_artifacts/alignment_mapping.csv")

# Namespaces
BASE = Namespace("http://sepsis-kg.org/")
PROP = Namespace("http://sepsis-kg.org/prop/")
WD = Namespace("http://www.wikidata.org/entity/")
DBP = Namespace("http://dbpedia.org/resource/")

# API endpoints
WIKIDATA_SEARCH_API = "https://www.wikidata.org/w/api.php"
DBPEDIA_LOOKUP_API = "https://lookup.dbpedia.org/api/search"

HEADERS = {"User-Agent": "SepsisKGBot/1.0 (educational project)"}

# Only align these labels
#ALIGN_LABELS = {"DISEASE", "BACTERIA", "BIOMARKER", "TREATMENT"}
#ALIGN_LABELS = {"DISEASE", "BACTERIA", "BIOMARKER", "TREATMENT", "ORG", "GPE", "PRODUCT"}
#REQUEST_DELAY = 0.5  # seconds between API calls

ALIGN_LABELS = {"DISEASE", "BACTERIA", "BIOMARKER", "TREATMENT", "GPE", "PRODUCT"}


REQUEST_DELAY = 1.0  # augmente de 0.5 à 1.0

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Convert a string to a valid URI fragment."""
    text = text.strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text


def entity_uri(entity: str) -> URIRef:
    """Return the local URI for an entity."""
    return BASE[slugify(entity)]


# ---------------------------------------------------------------------------
# Wikidata Search API
# ---------------------------------------------------------------------------

def search_wikidata(entity: str) -> dict | None:
    """
    Search Wikidata using the official wbsearchentities API.

    Both our entity and Wikidata labels are lowercased before
    comparison — so casing differences never cause mismatches.
    """
    try:
        response = requests.get(
            WIKIDATA_SEARCH_API,
            params={
                "action": "wbsearchentities",
                "search": entity,
                "language": "en",
                "limit": 10,
                "format": "json",
            },
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        results = response.json().get("search", [])

        if not results:
            return None

        # Normalize both sides to lowercase before comparing
        entity_lower = entity.lower().strip()

        # Look for exact match (case insensitive)
        exact_matches = [
            r for r in results
            if r.get("label", "").lower() == entity_lower
            or r.get("match", {}).get("text", "").lower() == entity_lower
        ]

        if exact_matches:
            best = exact_matches[0]
            confidence = 1.0 if len(exact_matches) == 1 else 0.9
        else:
            # No exact match → take first result with lower confidence
            best = results[0]
            confidence = 0.7

        return {
            "external_id": best["id"],
            "external_uri": f"http://www.wikidata.org/entity/{best['id']}",
            "confidence": confidence,
            "source": "wikidata",
        }

    except Exception as exc:
        log.warning("Wikidata API failed for '%s': %s", entity, exc)
        return None


# ---------------------------------------------------------------------------
# DBpedia Lookup API (fallback)
# ---------------------------------------------------------------------------

def search_dbpedia(entity: str) -> dict | None:
    """
    Search DBpedia using the official Lookup API as fallback.

    Both our entity and DBpedia labels are lowercased before
    comparison — so casing differences never cause mismatches.
    """
    try:
        response = requests.get(
            DBPEDIA_LOOKUP_API,
            params={
                "query": entity,
                "maxResults": 5,
                "format": "json",
            },
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        results = response.json().get("docs", [])

        if not results:
            return None

        # Normalize both sides to lowercase
        entity_lower = entity.lower().strip()

        exact_matches = [
            r for r in results
            if any(
                label.lower() == entity_lower
                for label in r.get("label", [])
            )
        ]

        if not exact_matches:
            return None

        best = exact_matches[0]
        resource_uri = best.get("resource", [""])[0]

        if not resource_uri:
            return None

        confidence = 1.0 if len(exact_matches) == 1 else 0.85

        return {
            "external_id": resource_uri.split("/")[-1],
            "external_uri": resource_uri,
            "confidence": confidence,
            "source": "dbpedia",
        }

    except Exception as exc:
        log.warning("DBpedia API failed for '%s': %s", entity, exc)
        return None


# ---------------------------------------------------------------------------
# Main alignment pipeline
# ---------------------------------------------------------------------------

def run_alignment(
    entities_file: Path = ENTITIES_FILE,
    alignment_file: Path = ALIGNMENT_FILE,
    mapping_file: Path = MAPPING_FILE,
) -> pd.DataFrame:
    """
    Align entities from our KG with Wikidata + DBpedia.

    Priority:
    1. Wikidata Search API (confidence 0.7-1.0)
    2. DBpedia Lookup API (confidence 0.85-1.0)
    3. Not found → stays local
    """
    alignment_file.parent.mkdir(parents=True, exist_ok=True)

    # Load entities
    df = pd.read_csv(entities_file)
    log.info("Loaded %d entities from %s", len(df), entities_file)

    # Keep only unique entities worth aligning
    df_align = (
        df[df["label"].isin(ALIGN_LABELS)]
        .drop_duplicates(subset=["entity"])
        .reset_index(drop=True)
    )
    log.info("Entities to align: %d", len(df_align))

    # Build RDF alignment graph
    g = Graph()
    g.bind("base", BASE)
    g.bind("owl", OWL)
    g.bind("wd", WD)
    g.bind("dbp", DBP)
    g.bind("prop", PROP)

    mapping_records = []

    for _, row in df_align.iterrows():
        entity = str(row["entity"])
        label = str(row["label"])
        local_uri = entity_uri(entity)

        log.info("Aligning: %s (%s)", entity, label)

        result = None

        # --- Step 1: Wikidata Search API ---
        result = search_wikidata(entity)
        time.sleep(REQUEST_DELAY)

        # --- Step 2: DBpedia Lookup API fallback ---
        if not result:
            log.info("  → Trying DBpedia...")
            result = search_dbpedia(entity)
            time.sleep(REQUEST_DELAY)

        # --- Save result ---
        if result:
            ext_uri = URIRef(result["external_uri"])

            # Add owl:sameAs triple
            g.add((local_uri, OWL.sameAs, ext_uri))

            # Add confidence score
            g.add((local_uri, PROP.alignmentConfidence,
                   Literal(result["confidence"], datatype=XSD.float)))

            mapping_records.append({
                "private_entity": entity,
                "label": label,
                "local_uri": str(local_uri),
                "external_uri": result["external_uri"],
                "external_id": result["external_id"],
                "confidence": result["confidence"],
                "source": result["source"],
                "status": "aligned",
            })
            log.info("  ✓ %s → %s (confidence: %.2f)",
                     result["source"], result["external_id"],
                     result["confidence"])
        else:
            mapping_records.append({
                "private_entity": entity,
                "label": label,
                "local_uri": str(local_uri),
                "external_uri": None,
                "external_id": None,
                "confidence": 0.0,
                "source": None,
                "status": "not_found",
            })
            log.info("  ✗ Not found anywhere")

    # Save alignment RDF
    g.serialize(destination=str(alignment_file), format="turtle")
    log.info("Alignment saved → %s (%d triples)", alignment_file, len(g))

    # Save mapping CSV
    df_mapping = pd.DataFrame(mapping_records)
    df_mapping.to_csv(mapping_file, index=False, encoding="utf-8")
    log.info("Mapping saved → %s (%d rows)", mapping_file, len(df_mapping))

    # Summary
    aligned = df_mapping[df_mapping["status"] == "aligned"]
    log.info("Summary: %d / %d entities aligned",
             len(aligned), len(df_mapping))

    for source in ["wikidata", "dbpedia"]:
        count = len(df_mapping[df_mapping["source"] == source])
        if count > 0:
            log.info("  → %s: %d entities", source, count)

    return df_mapping


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_alignment()