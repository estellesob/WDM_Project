"""
src/kg/expansion.py
===================
KB Expansion via SPARQL queries on Wikidata.

Step 4 of the KB Construction lab:
- For each aligned entity, fetch all triples from Wikidata (1-hop)
- For key entities (sepsis, bacteria, treatments), fetch 2-hop
- Target: 50,000 - 200,000 triples

Outputs:
    kg_artifacts/expanded.nt   – expanded KB in N-Triples format
    kg_artifacts/kb_stats.txt  – statistics report

Usage (standalone):
    python -m src.kg.expansion
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from rdflib import ConjunctiveGraph, Graph, Namespace, URIRef

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAPPING_FILE = Path("kg_artifacts/alignment_mapping.csv")
KG_FILE = Path("kg_artifacts/sepsis_kg.ttl")
EXPANDED_FILE = Path("kg_artifacts/expanded.nt")
STATS_FILE = Path("kg_artifacts/kb_stats.txt")

BASE = Namespace("http://sepsis-kg.org/")
PROP = Namespace("http://sepsis-kg.org/prop/")

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "SepsisKGBot/1.0 (educational project)",
    "Accept": "application/sparql-results+json",
}

REQUEST_DELAY = 1.0
LIMIT_PER_ENTITY = 2000   # était 1000
LIMIT_2HOP = 1000         # était 500

TWO_HOP_ENTITIES = {
    "sepsis", "septic shock", "bacteremia", "pneumonia",
    "escherichia coli", "staphylococcus aureus", "klebsiella pneumoniae",
    "vancomycin", "antibiotics", "procalcitonin", "lactate",
    "meningitis", "endocarditis", "peritonitis",
    "norepinephrine", "dexamethasone", "heparin",
    "white blood cell", "platelet", "acinetobacter",
    "corticosteroid", "mechanical ventilation",
    "acute kidney injury", "respiratory failure",
    "disseminated intravascular coagulation",
    "c-reactive protein", "troponin", "d-dimer",
    "fluid resuscitation", "blood culture",
    "piperacillin", "vasopressor", "oxygen therapy",
}



log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------

def fetch_1hop(wikidata_id: str) -> list[tuple]:
    """
    Fetch all triples for a Wikidata entity (1-hop expansion).
    Returns a list of (subject, predicate, object) tuples.
    """
    query = f"""
    SELECT ?p ?o WHERE {{
        wd:{wikidata_id} ?p ?o .
    }}
    LIMIT {LIMIT_PER_ENTITY}
    """
    return _run_sparql(query, wikidata_id, hop=1)


def fetch_2hop(wikidata_id: str) -> list[tuple]:
    """
    Fetch 2-hop triples for a Wikidata entity.
    Finds entities connected to our entity and their properties.
    """
    query = f"""
    SELECT ?o ?p2 ?o2 WHERE {{
        wd:{wikidata_id} ?p1 ?o .
        ?o ?p2 ?o2 .
        FILTER(isIRI(?o))
    }}
    LIMIT {LIMIT_2HOP}
    """
    return _run_sparql_2hop(query, wikidata_id)


def _run_sparql(query: str, entity_id: str, hop: int) -> list[tuple]:
    """Execute a SPARQL query and return (subject, predicate, object) tuples."""
    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()["results"]["bindings"]

        triples = []
        subject = f"http://www.wikidata.org/entity/{entity_id}"
        for row in results:
            pred = row["p"]["value"]
            obj = row["o"]["value"]
            triples.append((subject, pred, obj))

        log.info("  %d-hop: %d triples for %s", hop, len(triples), entity_id)
        return triples

    except Exception as exc:
        log.warning("SPARQL failed for %s: %s", entity_id, exc)
        return []


def _run_sparql_2hop(query: str, entity_id: str) -> list[tuple]:
    """Execute a 2-hop SPARQL query."""
    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()["results"]["bindings"]

        triples = []
        for row in results:
            subj = row["o"]["value"]
            pred = row["p2"]["value"]
            obj = row["o2"]["value"]
            triples.append((subj, pred, obj))

        log.info("  2-hop: %d triples for %s", len(triples), entity_id)
        return triples

    except Exception as exc:
        log.warning("2-hop SPARQL failed for %s: %s", entity_id, exc)
        return []


# ---------------------------------------------------------------------------
# Main expansion pipeline
# ---------------------------------------------------------------------------

def run_expansion(
    mapping_file: Path = MAPPING_FILE,
    kg_file: Path = KG_FILE,
    expanded_file: Path = EXPANDED_FILE,
    stats_file: Path = STATS_FILE,
) -> Graph:
    """
    Expand the KB by fetching triples from Wikidata for each aligned entity.
    """
    expanded_file.parent.mkdir(parents=True, exist_ok=True)

    # Load our existing KG
    log.info("Loading existing KG from %s", kg_file)
    g = Graph()
    g.parse(str(kg_file), format="turtle")
    initial_triples = len(g)
    log.info("Initial KG: %d triples", initial_triples)

    # Load alignment mapping
    df = pd.read_csv(mapping_file)
    aligned = df[df["status"] == "aligned"].reset_index(drop=True)
    log.info("Aligned entities to expand: %d", len(aligned))

    total_new_triples = 0

    for _, row in aligned.iterrows():
        entity = str(row["private_entity"])
        wikidata_id = str(row["external_id"])
        source = str(row["source"])

        # Only expand Wikidata entities
        if source not in ["wikidata", "manual"]:
            continue

        # Skip Wikidata IDs that don't start with Q
        if not wikidata_id.startswith("Q"):
            continue

        log.info("Expanding: %s (%s)", entity, wikidata_id)

        # --- 1-hop expansion ---
        triples_1hop = fetch_1hop(wikidata_id)
        time.sleep(REQUEST_DELAY)

        # --- 2-hop expansion for key entities ---
        triples_2hop = []
        if entity.lower() in TWO_HOP_ENTITIES:
            triples_2hop = fetch_2hop(wikidata_id)
            time.sleep(REQUEST_DELAY)

        # Add all triples to graph
        all_triples = triples_1hop + triples_2hop
        for subj, pred, obj in all_triples:
            try:
                s = URIRef(subj)
                p = URIRef(pred)
                # Object can be URI or literal
                o = URIRef(obj) if obj.startswith("http") else obj
                g.add((s, p, o))
            except Exception:
                continue

        new_count = len(g) - initial_triples - total_new_triples
        total_new_triples += new_count
        log.info("  Total so far: %d triples", len(g))

    # Save expanded KB in N-Triples format
    g.serialize(destination=str(expanded_file), format="nt")
    log.info("Expanded KB saved → %s", expanded_file)

    # Generate statistics
    final_triples = len(g)
    stats = f"""
KB Statistics
=============
Initial triples:     {initial_triples}
New triples added:   {final_triples - initial_triples}
Final triples:       {final_triples}
Entities expanded:   {len(aligned)}
2-hop entities:      {len(TWO_HOP_ENTITIES)}
Output format:       N-Triples (.nt)

Entity breakdown:
{df.groupby('label')['status'].value_counts().to_string()}
"""
    stats_file.write_text(stats)
    log.info("Stats saved → %s", stats_file)
    log.info("Final KB: %d triples (target: 50,000-200,000)", final_triples)

    return g


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_expansion()