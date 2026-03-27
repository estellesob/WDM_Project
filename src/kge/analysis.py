"""
src/kge/analysis.py
===================
Embedding Analysis for the Sepsis Knowledge Graph.

Section 6 of the KGE lab:
6.1 Nearest Neighbors
6.2 Clustering analysis (t-SNE)
6.3 Relation Behavior

Usage (standalone):
    python -m src.kge.analysis
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from pykeen.triples import TriplesFactory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KGE_DIR = Path("data/kge")
RESULTS_DIR = Path("data/kge/results")
PLOTS_DIR = Path("data/kge/plots")

# Entities to analyze for nearest neighbors
QUERY_ENTITIES = [
    "http://www.wikidata.org/entity/Q183134",   # sepsis
    "http://www.wikidata.org/entity/Q25419",    # E. coli
    "http://www.wikidata.org/entity/Q424027",   # vancomycin
    "http://www.wikidata.org/entity/Q12192",    # pneumonia
    "http://www.wikidata.org/entity/Q786600",   # procalcitonin
]

# Entity labels for display
ENTITY_LABELS = {
    "http://www.wikidata.org/entity/Q183134": "sepsis",
    "http://www.wikidata.org/entity/Q25419":  "E. coli",
    "http://www.wikidata.org/entity/Q424027": "vancomycin",
    "http://www.wikidata.org/entity/Q12192":  "pneumonia",
    "http://www.wikidata.org/entity/Q786600": "procalcitonin",
}

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Load model and embeddings
# ---------------------------------------------------------------------------

def load_model_and_factory(model_name: str = "distmult"):
    """
    Load a trained PyKEEN model and its TriplesFactory.
    Returns (model, entity_embeddings, entity_to_id, id_to_entity)
    """
    import torch
    model_path = RESULTS_DIR / model_name / "trained_model.pkl"
    factory_path = RESULTS_DIR / model_name / "training_triples"

    log.info("Loading model from %s", model_path)
    model = torch.load(str(model_path), map_location="cpu")
    model.eval()

    log.info("Loading factory from %s", factory_path)
    factory = TriplesFactory.from_path_binary(factory_path)

    # Extract entity embeddings
    with torch.no_grad():
        embeddings = model.entity_representations[0](
            indices=None
        ).numpy()

    log.info("Entity embeddings shape: %s", embeddings.shape)

    entity_to_id = factory.entity_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}

    return model, embeddings, entity_to_id, id_to_entity


# ---------------------------------------------------------------------------
# 6.1 Nearest Neighbors
# ---------------------------------------------------------------------------

def find_nearest_neighbors(
    embeddings: np.ndarray,
    entity_to_id: dict,
    id_to_entity: dict,
    query_uri: str,
    k: int = 10,
) -> list[tuple[str, float]]:
    """
    Find k nearest neighbors of a query entity in embedding space.
    Uses cosine similarity.
    """
    if query_uri not in entity_to_id:
        log.warning("Entity not found: %s", query_uri)
        return []

    query_id = entity_to_id[query_uri]
    query_vec = embeddings[query_id]

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normalized = embeddings / norms
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    similarities = normalized @ query_norm
    similarities[query_id] = -1  # exclude self

    top_k_ids = np.argsort(similarities)[::-1][:k]

    neighbors = []
    for idx in top_k_ids:
        uri = id_to_entity[idx]
        sim = similarities[idx]
        # Get short label
        label = uri.split("/")[-1] if "/" in uri else uri
        neighbors.append((label, float(sim)))

    return neighbors


def run_nearest_neighbors(
    embeddings: np.ndarray,
    entity_to_id: dict,
    id_to_entity: dict,
) -> None:
    """Run nearest neighbor analysis for key medical entities."""
    log.info("=== 6.1 Nearest Neighbors Analysis ===")

    for uri in QUERY_ENTITIES:
        label = ENTITY_LABELS.get(uri, uri.split("/")[-1])
        neighbors = find_nearest_neighbors(
            embeddings, entity_to_id, id_to_entity, uri, k=5
        )

        if not neighbors:
            continue

        log.info("\nNearest neighbors of '%s':", label)
        for i, (neighbor, sim) in enumerate(neighbors, 1):
            log.info("  %d. %s (similarity: %.4f)", i, neighbor, sim)


# ---------------------------------------------------------------------------
# 6.2 t-SNE Clustering
# ---------------------------------------------------------------------------

def run_tsne(
    embeddings: np.ndarray,
    entity_to_id: dict,
    plots_dir: Path,
    max_entities: int = 2000,
) -> None:
    """
    Apply t-SNE to reduce embeddings to 2D and plot.
    Colors entities by their namespace (Wikidata vs Sepsis-KG).
    """
    log.info("=== 6.2 t-SNE Clustering Analysis ===")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Sample entities if too many
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    n = min(max_entities, len(embeddings))
    indices = np.random.choice(len(embeddings), n, replace=False)

    sampled_embeddings = embeddings[indices]
    sampled_uris = [id_to_entity[i] for i in indices]

    # Assign colors by entity type
    colors = []
    labels = []
    for uri in sampled_uris:
        if "wikidata" in uri:
            colors.append("steelblue")
            labels.append("Wikidata")
        elif "sepsis-kg" in uri:
            # Check entity type from URI
            if any(t in uri.lower() for t in ["disease", "bacteria", "treatment", "biomarker"]):
                colors.append("tomato")
                labels.append("Sepsis-KG Medical")
            else:
                colors.append("orange")
                labels.append("Sepsis-KG Other")
        else:
            colors.append("gray")
            labels.append("Other")

    log.info("Running t-SNE on %d entities...", n)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        n_iter=1000,
    )
    embeddings_2d = tsne.fit_transform(sampled_embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot by category
    unique_labels = list(set(labels))
    color_map = {
        "Wikidata": "steelblue",
        "Sepsis-KG Medical": "tomato",
        "Sepsis-KG Other": "orange",
        "Other": "gray",
    }

    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]
        ax.scatter(x, y, c=color_map[lbl], label=lbl,
                   alpha=0.5, s=10)

    # Highlight key medical entities
    for uri, name in ENTITY_LABELS.items():
        if uri in entity_to_id:
            idx = entity_to_id[uri]
            if idx in indices:
                pos = np.where(indices == idx)[0][0]
                ax.scatter(
                    embeddings_2d[pos, 0],
                    embeddings_2d[pos, 1],
                    c="red", s=100, zorder=5
                )
                ax.annotate(
                    name,
                    (embeddings_2d[pos, 0], embeddings_2d[pos, 1]),
                    fontsize=8, fontweight="bold",
                )

    ax.set_title("t-SNE visualization of entity embeddings (DistMult)")
    ax.legend()
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")

    plot_path = plots_dir / "tsne_embeddings.png"
    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    log.info("t-SNE plot saved → %s", plot_path)


# ---------------------------------------------------------------------------
# 6.3 Relation Behavior
# ---------------------------------------------------------------------------

def run_relation_analysis(
    model,
    factory: TriplesFactory,
) -> None:
    """
    Analyze relation embeddings:
    - Symmetric relations: r ≈ -r ?
    - Inverse relations: r1 ≈ -r2 ?
    """
    log.info("=== 6.3 Relation Behavior Analysis ===")

    with torch.no_grad():
        rel_embeddings = model.relation_representations[0](
            indices=None
        ).numpy()

    id_to_relation = {v: k for k, v in factory.relation_to_id.items()}

    log.info("Total relations: %d", len(rel_embeddings))
    log.info("Relation embedding dim: %d", rel_embeddings.shape[1])

    # Check symmetric relations (||r|| should be small for symmetric)
    norms = np.linalg.norm(rel_embeddings, axis=1)
    sorted_ids = np.argsort(norms)

    log.info("\nRelations with smallest norms (potentially symmetric):")
    for i in sorted_ids[:5]:
        rel = id_to_relation[i].split("/")[-1]
        log.info("  %s: norm=%.4f", rel, norms[i])

    log.info("\nRelations with largest norms (strong directional):")
    for i in sorted_ids[-5:]:
        rel = id_to_relation[i].split("/")[-1]
        log.info("  %s: norm=%.4f", rel, norms[i])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis() -> None:
    """Run full embedding analysis."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load best model (DistMult)
    model, embeddings, entity_to_id, id_to_entity = load_model_and_factory(
        "distmult"
    )

    # Load factory for relation analysis
    factory = TriplesFactory.from_path_binary(
        RESULTS_DIR / "distmult" / "training_triples"
    )

    # 6.1 Nearest Neighbors
    run_nearest_neighbors(embeddings, entity_to_id, id_to_entity)

    # 6.2 t-SNE
    run_tsne(embeddings, entity_to_id, PLOTS_DIR)

    # 6.3 Relation Behavior
    run_relation_analysis(model, factory)

    log.info("Analysis complete! Plots saved to %s", PLOTS_DIR)


if __name__ == "__main__":
    run_analysis()