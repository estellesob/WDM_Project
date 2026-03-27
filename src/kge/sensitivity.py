"""
src/kge/sensitivity.py
======================
KB Size Sensitivity analysis.

Tests how KGE performance scales with dataset size:
- 20k triples
- 50k triples  
- Full dataset (~40k train)

Usage (standalone):
    python -m src.kge.sensitivity
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KGE_DIR = Path("data/kge")
RESULTS_DIR = Path("data/kge/results/sensitivity")

CONFIG = {
    "embedding_dim": 100,
    "num_epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.01,
    "random_seed": 42,
}

SUBSAMPLE_SIZES = [20000, 50000]

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_metrics(result) -> dict:
    """Extract MRR and Hits@K from PyKEEN result."""
    metrics = result.metric_results.to_dict()
    both = metrics.get("both", {})
    realistic = both.get("realistic", {})
    return {
        "mrr":     realistic.get("inverse_harmonic_mean_rank", 0.0),
        "hits@1":  realistic.get("hits_at_1", 0.0),
        "hits@3":  realistic.get("hits_at_3", 0.0),
        "hits@10": realistic.get("hits_at_10", 0.0),
    }


def subsample_triples(
    factory: TriplesFactory,
    n: int,
    seed: int = 42,
) -> TriplesFactory:
    """
    Subsample n triples from a TriplesFactory.
    Returns a new TriplesFactory with n triples.
    """
    import torch
    random.seed(seed)
    total = factory.num_triples
    n = min(n, total)
    indices = random.sample(range(total), n)
    indices_tensor = torch.tensor(indices)
    subsampled = factory.mapped_triples[indices_tensor]
    return TriplesFactory(
        mapped_triples=subsampled,
        entity_to_id=factory.entity_to_id,
        relation_to_id=factory.relation_to_id,
    )


def train_and_eval(
    model_name: str,
    training: TriplesFactory,
    validation: TriplesFactory,
    testing: TriplesFactory,
    output_dir: Path,
) -> dict:
    """Train a model and return metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result = pipeline(
        model=model_name,
        training=training,
        validation=validation,
        testing=testing,
        training_kwargs=dict(
            num_epochs=CONFIG["num_epochs"],
            batch_size=CONFIG["batch_size"],
        ),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=CONFIG["learning_rate"]),
        model_kwargs=dict(embedding_dim=CONFIG["embedding_dim"]),
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(filtered=True),
        random_seed=CONFIG["random_seed"],
        device="cpu",
    )

    result.save_to_directory(str(output_dir))
    return extract_metrics(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sensitivity() -> None:
    """
    Run KB size sensitivity analysis.
    Tests DistMult (best model) on different dataset sizes.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load full dataset
    log.info("Loading full dataset...")
    training = TriplesFactory.from_path(
        KGE_DIR / "train.txt", delimiter="\t"
    )
    validation = TriplesFactory.from_path(
        KGE_DIR / "valid.txt", delimiter="\t",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    testing = TriplesFactory.from_path(
        KGE_DIR / "test.txt", delimiter="\t",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )

    results = []

    # Test on subsampled sizes
    for size in SUBSAMPLE_SIZES:
        log.info("=" * 50)
        log.info("Training with %dk triples", size // 1000)
        log.info("=" * 50)

        sub_training = subsample_triples(training, size)
        log.info("Subsampled: %d triples", sub_training.num_triples)

        m = train_and_eval(
            "DistMult",
            sub_training,
            validation,
            testing,
            RESULTS_DIR / f"distmult_{size // 1000}k",
        )

        results.append({
            "size": size,
            "label": f"{size // 1000}k",
            **m,
        })

        log.info("MRR=%.4f, Hits@10=%.4f", m["mrr"], m["hits@10"])

    # Full dataset
    log.info("=" * 50)
    log.info("Training with full dataset (%d triples)", training.num_triples)
    log.info("=" * 50)

    m = train_and_eval(
        "DistMult",
        training,
        validation,
        testing,
        RESULTS_DIR / "distmult_full",
    )
    results.append({
        "size": training.num_triples,
        "label": "full",
        **m,
    })

    # Print summary table
    log.info("=" * 50)
    log.info("KB SIZE SENSITIVITY RESULTS")
    log.info("=" * 50)
    log.info("%-10s %8s %8s %8s %8s",
             "Size", "MRR", "Hits@1", "Hits@3", "Hits@10")
    log.info("-" * 50)
    for r in results:
        log.info("%-10s %8.4f %8.4f %8.4f %8.4f",
                 r["label"], r["mrr"],
                 r["hits@1"], r["hits@3"], r["hits@10"])


if __name__ == "__main__":
    run_sensitivity()