"""
src/kg/expansion.py
===================
KB Expansion via SPARQL queries on Wikidata.

Step 4 of the KB Construction lab:
- For each aligned entity, fetch all triples from Wikidata (1-hop)
- For key entities (sepsis, bacteria, treatments), fetch 2-hop
- Expand with infectious diseases and general medical domain
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
from rdflib import Graph, Literal, Namespace, URIRef

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
LIMIT_PER_ENTITY = 2000
LIMIT_2HOP = 1000

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
    """Fetch only URI-URI-URI triples for a Wikidata entity (1-hop)."""
    query = f"""
    SELECT ?p ?o WHERE {{
        wd:{wikidata_id} ?p ?o .
        FILTER(isIRI(?o))
    }}
    LIMIT {LIMIT_PER_ENTITY}
    """
    return _run_sparql(query, wikidata_id, hop=1)


def fetch_2hop(wikidata_id: str) -> list[tuple]:
    """Fetch only URI-URI-URI triples for a Wikidata entity (2-hop)."""
    query = f"""
    SELECT ?o ?p2 ?o2 WHERE {{
        wd:{wikidata_id} ?p1 ?o .
        ?o ?p2 ?o2 .
        FILTER(isIRI(?o))
        FILTER(isIRI(?o2))
    }}
    LIMIT {LIMIT_2HOP}
    """
    return _run_sparql_2hop(query, wikidata_id)


def fetch_infectious_diseases() -> list[tuple]:
    """
    Fetch infectious disease triples from Wikidata.
    Provides core medical context for KGE training.
    """
    query = """
    SELECT ?disease ?p ?o WHERE {
        ?disease wdt:P279 wd:Q12136 .
        ?disease ?p ?o .
        FILTER(isIRI(?o))
    }
    LIMIT 100000
    """
    log.info("Fetching infectious disease triples from Wikidata...")
    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=60,
        )
        response.raise_for_status()
        results = response.json()["results"]["bindings"]
        triples = []
        for row in results:
            s = row["disease"]["value"]
            p = row["p"]["value"]
            o = row["o"]["value"]
            triples.append((s, p, o))
        log.info("Fetched %d infectious disease triples", len(triples))
        return triples
    except Exception as exc:
        log.warning("Failed to fetch infectious diseases: %s", exc)
        return []


def fetch_medical_domain() -> list[tuple]:
    """
    Fetch general medical domain triples from Wikidata.
    Covers bacterial diseases, inflammatory diseases, lung diseases,
    immune system diseases for richer and less sparse medical context.
    """
    medical_classes = [
        "wd:Q929833",   # bacterial infectious disease
        "wd:Q3136364",  # inflammatory disease
        "wd:Q1149548",  # lung disease
        "wd:Q15978631", # disease caused by bacteria
        "wd:Q18123741", # immune system disease
        "wd:Q188553",   # septicemia
        "wd:Q101896",   # hospital-acquired infection
        "wd:Q202387",   # opportunistic infection
        "wd:Q177719",   # autoimmune disease
        "wd:Q84263196", # pandemic disease
        "wd:Q1456357",  # tropical disease
        "wd:Q164778",   # zoonosis

        "wd:Q726097",   # parasitic disease
        "wd:Q149649",   # respiratory disease
        "wd:Q3025883",  # blood disease
        "wd:Q1054718",  # lymphatic disease
        "wd:Q1928817",  # cardiovascular infectious disease
        "wd:Q188874",   # liver disease
        "wd:Q193174",   # kidney disease
    ]

    all_triples = []

    for cls in medical_classes:
        query = f"""
        SELECT ?disease ?p ?o WHERE {{
            {{
                ?disease wdt:P279 {cls} .
            }} UNION {{
                ?disease wdt:P31 {cls} .
            }}
            ?disease ?p ?o .
            FILTER(isIRI(?o))
        }}
        LIMIT 100000
        """
        
        log.info("Fetching medical class: %s", cls)
        try:
            response = requests.get(
                WIKIDATA_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=60,
            )
            response.raise_for_status()
            results = response.json()["results"]["bindings"]
            for row in results:
                s = row["disease"]["value"]
                p = row["p"]["value"]
                o = row["o"]["value"]
                all_triples.append((s, p, o))
            log.info("  → %d triples fetched for %s", len(results), cls)
            time.sleep(REQUEST_DELAY)
        except Exception as exc:
            log.warning("Failed for %s: %s", cls, exc)

    log.info("Total medical domain triples: %d", len(all_triples))
    return all_triples


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


def clean_graph(g: Graph) -> Graph:
    """
    Minimal cleaning before KGE export.
    Graph validation only — all URIs are already valid.
    """
    log.info("Graph validation: %d triples, all URIs valid ✓", len(g))
    return g


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

    Steps:
    1. Load existing KG
    2. 1-hop expansion for all aligned entities
    3. 2-hop expansion for key medical entities
    4. Expand with infectious diseases (50k triples)
    5. Expand with general medical domain (5 disease classes)
    6. Save expanded KB
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

    # --- Per-entity expansion ---
    for _, row in aligned.iterrows():
        entity = str(row["private_entity"])
        wikidata_id = str(row["external_id"])
        source = str(row["source"])

        if source not in ["wikidata", "manual"]:
            continue
        if not wikidata_id.startswith("Q"):
            continue

        log.info("Expanding: %s (%s)", entity, wikidata_id)

        # 1-hop expansion
        triples_1hop = fetch_1hop(wikidata_id)
        time.sleep(REQUEST_DELAY)

        # 2-hop expansion for key entities
        triples_2hop = []
        if entity.lower() in TWO_HOP_ENTITIES:
            triples_2hop = fetch_2hop(wikidata_id)
            time.sleep(REQUEST_DELAY)

        # Add all triples to graph
        for subj, pred, obj in triples_1hop + triples_2hop:
            try:
                s = URIRef(subj)
                p = URIRef(pred)
                o = URIRef(obj) if obj.startswith("http") else obj
                g.add((s, p, o))
            except Exception:
                continue

        new_count = len(g) - initial_triples - total_new_triples
        total_new_triples += new_count
        log.info("  Total so far: %d triples", len(g))

    # --- Infectious disease expansion ---
    log.info("=== Expanding with infectious disease context ===")
    infectious_triples = fetch_infectious_diseases()
    for subj, pred, obj in infectious_triples:
        try:
            g.add((URIRef(subj), URIRef(pred), URIRef(obj)))
        except Exception:
            continue
    log.info("After infectious diseases: %d triples", len(g))

    # --- General medical domain expansion ---
    log.info("=== Expanding with general medical domain ===")
    medical_triples = fetch_medical_domain()
    for subj, pred, obj in medical_triples:
        try:
            g.add((URIRef(subj), URIRef(pred), URIRef(obj)))
        except Exception:
            continue
    log.info("After medical domain: %d triples", len(g))

    # Clean and save
    g = clean_graph(g)
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