"""
src/kge/prepare.py
==================
Data preparation for Knowledge Graph Embedding.

Reads:
    kg_artifacts/expanded.nt   – expanded KB in N-Triples format

Produces:
    data/kge/train.txt   – 80% of triples
    data/kge/valid.txt   – 10% of triples
    data/kge/test.txt    – 10% of triples

Usage (standalone):
    python -m src.kge.prepare
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.term import Literal, BNode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPANDED_FILE = Path("kg_artifacts/expanded.nt")
KGE_DIR = Path("data/kge")

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO  = 0.10

RANDOM_SEED = 42

# Predicates to remove (literals, not useful for KGE)
SKIP_PREDICATES = {
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#definition",
    "http://schema.org/description",
    "http://schema.org/name",
    "http://sepsis-kg.org/prop/extractedFrom",
    "http://sepsis-kg.org/prop/sourceUrl",
    "http://sepsis-kg.org/prop/alignmentConfidence",
}

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def load_and_clean(expanded_file: Path) -> list[tuple[str, str, str]]:
    """
    Load the expanded KB and clean it for KGE.

    Removes:
    - Duplicate triples (rdflib handles this)
    - Literal objects (KGE needs URI-URI-URI triples)
    - Skip-listed predicates
    - Blank nodes
    - Invalid URIs
    - Infrequent entities (< MIN_ENTITY_FREQ appearances)
    - Infrequent relations (< MIN_RELATION_FREQ appearances)
    """
    from collections import Counter

    MIN_ENTITY_FREQ = 2
    MIN_RELATION_FREQ = 70

    log.info("Loading expanded KB from %s", expanded_file)
    g = Graph()
    g.parse(str(expanded_file), format="nt")
    log.info("Loaded %d triples", len(g))

    triples = []
    skipped = 0

    for s, p, o in g:
        # Skip blank nodes
        if isinstance(s, BNode) or isinstance(o, BNode):
            skipped += 1
            continue
        # Skip literal objects
        if isinstance(o, Literal):
            skipped += 1
            continue
        # Skip listed predicates
        if str(p) in SKIP_PREDICATES:
            skipped += 1
            continue
        # Keep only URI-URI-URI triples
        if not (isinstance(s, URIRef) and
                isinstance(p, URIRef) and
                isinstance(o, URIRef)):
            skipped += 1
            continue
        triples.append((str(s), str(p), str(o)))

    # Deduplicate
    triples = list(set(triples))
    log.info("After cleaning: %d triples (skipped %d)", len(triples), skipped)

    # --- Frequency filtering ---
    # Count entity and relation frequencies
    entity_counts = Counter()
    for s, p, o in triples:
        entity_counts[s] += 1
        entity_counts[o] += 1

    relation_counts = Counter(p for s, p, o in triples)

    # Keep only frequent entities and relations
    triples = [
        (s, p, o) for s, p, o in triples
        if entity_counts[s] >= MIN_ENTITY_FREQ
        and entity_counts[o] >= MIN_ENTITY_FREQ
        and relation_counts[p] >= MIN_RELATION_FREQ
    ]

    # Stats after filtering
    remaining_entities = {e for s, p, o in triples for e in [s, o]}
    remaining_relations = {p for s, p, o in triples}

    log.info("After frequency filtering: %d triples", len(triples))
    log.info("Entities kept : %d", len(remaining_entities))
    log.info("Relations kept: %d", len(remaining_relations))

    return triples

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_triples(
    triples: list[tuple[str, str, str]],
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list, list, list]:
    """
    Split triples into train/valid/test sets.
    Ensures entities appearing 3+ times are in all splits.
    Rare entities (appearing once) go to train only.
    """
    from collections import Counter

    random.seed(seed)

    # Count entity appearances
    entity_counts = Counter()
    for s, p, o in triples:
        entity_counts[s] += 1
        entity_counts[o] += 1

    # Separate frequent vs rare triples
    # Frequent = both entities appear 3+ times → can be split
    # Rare = at least one entity appears once → goes to train
    frequent_triples = []
    rare_triples = []

    for s, p, o in triples:
        if entity_counts[s] >= 2 and entity_counts[o] >= 2:
            frequent_triples.append((s, p, o))
        else:
            rare_triples.append((s, p, o))

    log.info("Frequent triples: %d | Rare triples: %d",
             len(frequent_triples), len(rare_triples))

    # Split only frequent triples 80/10/10
    random.shuffle(frequent_triples)
    n = len(frequent_triples)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = frequent_triples[:n_train] + rare_triples
    valid = frequent_triples[n_train:n_train + n_valid]
    test  = frequent_triples[n_train + n_valid:]

    random.shuffle(train)

    log.info("Split: train=%d, valid=%d, test=%d",
             len(train), len(valid), len(test))

    return train, valid, test

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_split(
    triples: list[tuple[str, str, str]],
    path: Path,
) -> None:
    """Save triples to a tab-separated file (subject predicate object)."""
    with open(path, "w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")
    log.info("Saved %d triples → %s", len(triples), path)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(
    train: list,
    valid: list,
    test: list,
) -> None:
    """Print statistics about the splits."""
    all_triples = train + valid + test

    entities = set()
    relations = set()
    for s, p, o in all_triples:
        entities.add(s)
        entities.add(o)
        relations.add(p)

    log.info("=== KGE Dataset Statistics ===")
    log.info("Total triples : %d", len(all_triples))
    log.info("Train         : %d (%.1f%%)",
             len(train), 100 * len(train) / len(all_triples))
    log.info("Valid         : %d (%.1f%%)",
             len(valid), 100 * len(valid) / len(all_triples))
    log.info("Test          : %d (%.1f%%)",
             len(test), 100 * len(test) / len(all_triples))
    log.info("Entities      : %d", len(entities))
    log.info("Relations     : %d", len(relations))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_preparation(
    expanded_file: Path = EXPANDED_FILE,
    kge_dir: Path = KGE_DIR,
) -> tuple[list, list, list]:
    """
    Full preparation pipeline:
    1. Load and clean expanded KB
    2. Split into train/valid/test
    3. Save splits
    4. Print statistics
    """
    kge_dir.mkdir(parents=True, exist_ok=True)

    # Load and clean
    triples = load_and_clean(expanded_file)

    # Split
    train, valid, test = split_triples(triples)

    # Save
    save_split(train, kge_dir / "train.txt")
    save_split(valid, kge_dir / "valid.txt")
    save_split(test,  kge_dir / "test.txt")

    # Stats
    print_stats(train, valid, test)

    return train, valid, test


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_preparation()