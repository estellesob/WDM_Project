"""
src/kge/train.py
================
Knowledge Graph Embedding training using PyKEEN.

Trains two models:
- TransE
- DistMult

Evaluates using:
- MRR (Mean Reciprocal Rank)
- Hits@1, Hits@3, Hits@10

Usage (standalone):
    python -m src.kge.train
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KGE_DIR = Path("data/kge")
RESULTS_DIR = Path("data/kge/results")

CONFIG = {
    "embedding_dim": 100,
    "num_epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.01,
    "random_seed": 42,
}

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_triples() -> tuple:
    """
    Load train/valid/test splits using PyKEEN TriplesFactory.
    All splits share the same entity/relation mappings from training.
    """
    log.info("Loading triples from %s", KGE_DIR)

    training = TriplesFactory.from_path(
        KGE_DIR / "train.txt",
        delimiter="\t",
    )
    validation = TriplesFactory.from_path(
        KGE_DIR / "valid.txt",
        delimiter="\t",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    testing = TriplesFactory.from_path(
        KGE_DIR / "test.txt",
        delimiter="\t",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )

    log.info("Training triples  : %d", training.num_triples)
    log.info("Validation triples: %d", validation.num_triples)
    log.info("Test triples      : %d", testing.num_triples)
    log.info("Entities          : %d", training.num_entities)
    log.info("Relations         : %d", training.num_relations)

    return training, validation, testing


# ---------------------------------------------------------------------------
# Extract metrics from PyKEEN result
# ---------------------------------------------------------------------------

def extract_metrics(result) -> dict:
    """
    Extract MRR and Hits@K from PyKEEN pipeline result.
    Handles nested dict structure: {head/tail/both: {realistic: {metric: value}}}
    """
    metrics = result.metric_results.to_dict()

    # Navigate nested structure
    both = metrics.get("both", {})
    realistic = both.get("realistic", {})

    mrr    = realistic.get("inverse_harmonic_mean_rank", 0.0)
    hits1  = realistic.get("hits_at_1", 0.0)
    hits3  = realistic.get("hits_at_3", 0.0)
    hits10 = realistic.get("hits_at_10", 0.0)

    return {
        "mrr": mrr,
        "hits@1": hits1,
        "hits@3": hits3,
        "hits@10": hits10,
    }


# ---------------------------------------------------------------------------
# Train a model
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    training: TriplesFactory,
    validation: TriplesFactory,
    testing: TriplesFactory,
) -> dict:
    """
    Train a KGE model using PyKEEN pipeline.
    Returns evaluation metrics.
    """
    log.info("Training %s...", model_name)

    output_dir = RESULTS_DIR / model_name.lower()
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
        optimizer_kwargs=dict(
            lr=CONFIG["learning_rate"],
        ),
        model_kwargs=dict(
            embedding_dim=CONFIG["embedding_dim"],
        ),
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(
            filtered=True,
        ),
        random_seed=CONFIG["random_seed"],
        device="cpu",
    )

    # Save results
    result.save_to_directory(str(output_dir))

    # Extract metrics
    m = extract_metrics(result)

    log.info("=== %s Results ===", model_name)
    log.info("MRR    : %.4f", m["mrr"])
    log.info("Hits@1 : %.4f", m["hits@1"])
    log.info("Hits@3 : %.4f", m["hits@3"])
    log.info("Hits@10: %.4f", m["hits@10"])

    return {
        "model": model_name,
        "mrr": m["mrr"],
        "hits@1": m["hits@1"],
        "hits@3": m["hits@3"],
        "hits@10": m["hits@10"],
        "pipeline_result": result,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_training() -> list[dict]:
    """
    Train TransE and DistMult, compare results.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    training, validation, testing = load_triples()

    results = []

    # Train TransE
    log.info("=" * 50)
    log.info("MODEL 1: TransE")
    log.info("=" * 50)
    transe = train_model("TransE", training, validation, testing)
    results.append(transe)

    # Train DistMult
    log.info("=" * 50)
    log.info("MODEL 2: DistMult")
    log.info("=" * 50)
    distmult = train_model("DistMult", training, validation, testing)
    results.append(distmult)

    # Train ComplEx
    log.info("=" * 50)
    log.info("MODEL 3: ComplEx")
    log.info("=" * 50)
    complex_result = train_model("ComplEx", training, validation, testing)
    results.append(complex_result)

    # Train RotatE
    log.info("=" * 50)
    log.info("MODEL 4: RotatE")
    log.info("=" * 50)
    rotate_result = train_model("RotatE", training, validation, testing)
    results.append(rotate_result)

    # Print comparison table
    log.info("=" * 50)
    log.info("MODEL COMPARISON")
    log.info("=" * 50)
    log.info("%-12s %8s %8s %8s %8s",
             "Model", "MRR", "Hits@1", "Hits@3", "Hits@10")
    log.info("-" * 50)
    for r in results:
        log.info("%-12s %8.4f %8.4f %8.4f %8.4f",
                 r["model"], r["mrr"],
                 r["hits@1"], r["hits@3"], r["hits@10"])

    return results


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_training()