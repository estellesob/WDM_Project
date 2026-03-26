"""
Information Extraction for the Sepsis Knowledge Graph.

Two tasks:
1. Named Entity Recognition (NER) : nodes of the future graph
   Labels kept: PERSON, ORG, GPE, LOC, DATE, PRODUCT
   + Medical whitelist: DISEASE, BACTERIA, BIOMARKER, TREATMENT, CARE_UNIT

2. Relation Extraction via dependency parsing → candidate edges
   Pattern: nsubj ← VERB → (dobj | prep → pobj)
   Only keeps triples where BOTH subject AND object span contain
   at least one named entity of an interesting type.

Outputs:
  data/samples/extracted_knowledge.csv   – flat entity table
  data/samples/extracted_relations.csv   – candidate triples table

Usage (standalone):
    python -m src.ie.extractor
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import spacy

from src.crawl.crawler import load_jsonl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = Path("data/samples/crawler_output.jsonl")
ENTITIES_FILE = Path("data/samples/extracted_knowledge.csv")
RELATIONS_FILE = Path("data/samples/extracted_relations.csv")

# spaCy labels to keep — we keep everything, dates included
KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "PRODUCT"}

# Medical whitelist: terms spaCy misses or mislabels
# Format: "term (lowercase)" -> "custom label"
MEDICAL_WHITELIST: dict[str, str] = {
    # Diseases / conditions
    "sepsis": "DISEASE",
    "septic shock": "DISEASE",
    "bacteremia": "DISEASE",
    "pneumonia": "DISEASE",
    "meningitis": "DISEASE",
    "endocarditis": "DISEASE",
    "peritonitis": "DISEASE",
    "multiple organ failure": "DISEASE",
    "organ dysfunction": "DISEASE",
    "acute kidney injury": "DISEASE",
    "respiratory failure": "DISEASE",
    "disseminated intravascular coagulation": "DISEASE",
    "dic": "DISEASE",
    # Biomarkers
    "procalcitonin": "BIOMARKER",
    "pct": "BIOMARKER",
    "lactate": "BIOMARKER",
    "interleukin-6": "BIOMARKER",
    "il-6": "BIOMARKER",
    "crp": "BIOMARKER",
    "c-reactive protein": "BIOMARKER",
    "white blood cell": "BIOMARKER",
    "wbc": "BIOMARKER",
    "platelet": "BIOMARKER",
    "ferritin": "BIOMARKER",
    "d-dimer": "BIOMARKER",
    "troponin": "BIOMARKER",
    "thrombomodulin": "BIOMARKER",
    "sofa score": "BIOMARKER",
    "qsofa": "BIOMARKER",
    # Bacteria / pathogens
    "escherichia coli": "BACTERIA",
    "e. coli": "BACTERIA",
    "staphylococcus aureus": "BACTERIA",
    "klebsiella pneumoniae": "BACTERIA",
    "pseudomonas aeruginosa": "BACTERIA",
    "streptococcus pneumoniae": "BACTERIA",
    "candida": "BACTERIA",
    "acinetobacter": "BACTERIA",
    "enterococcus": "BACTERIA",
    "stenotrophomonas maltophilia": "BACTERIA",
    # Treatments
    "vancomycin": "TREATMENT",
    "antibiotic": "TREATMENT",
    "antibiotics": "TREATMENT",
    "fluid resuscitation": "TREATMENT",
    "vasopressor": "TREATMENT",
    "norepinephrine": "TREATMENT",
    "meropenem": "TREATMENT",
    "piperacillin": "TREATMENT",
    "corticosteroid": "TREATMENT",
    "dexamethasone": "TREATMENT",
    "hydrocortisone": "TREATMENT",
    "insulin": "TREATMENT",
    "heparin": "TREATMENT",
    "blood culture": "TREATMENT",
    "mechanical ventilation": "TREATMENT",
    "oxygen therapy": "TREATMENT",
    # Care settings
    "icu": "CARE_UNIT",
    "intensive care unit": "CARE_UNIT",
    "nicu": "CARE_UNIT",
    "picu": "CARE_UNIT",
    "emergency department": "CARE_UNIT",
    "ed": "CARE_UNIT",
}

# All labels used for relation extraction
ALL_LABELS = KEEP_LABELS | set(MEDICAL_WHITELIST.values())

# Dependency relations that mark the direct object side of a verb
OBJ_DEPS = {"dobj", "attr", "oprd", "pobj", "obl", "nsubjpass"}

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# NER helpers
# ---------------------------------------------------------------------------

def extract_entities(doc: spacy.tokens.Doc, source_url: str) -> list[dict]:
    """
    Return a deduplicated list of entity dicts from *doc*.

    Combines spaCy NER results with medical whitelist matching.
    Each dict has: entity, label, source_url.
    """
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []

    # --- Part 1: spaCy detected entities ---
    for ent in doc.ents:
        if ent.label_ not in KEEP_LABELS:
            continue
        entity = ent.text.strip()
        if len(entity) < 2:
            continue
        key = (entity.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "entity": entity,
            "label": ent.label_,
            "source_url": source_url,
        })

    # --- Part 2: medical whitelist matching ---
    text_lower = doc.text.lower()
    for term, label in MEDICAL_WHITELIST.items():
        # Use word boundary matching to avoid partial matches
        # e.g. "icu" should not match "nicu"
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            key = (term, label)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "entity": term,
                "label": label,
                "source_url": source_url,
            })

    return rows


# ---------------------------------------------------------------------------
# Relation extraction helpers
# ---------------------------------------------------------------------------

def _subtree_text(token: spacy.tokens.Token) -> str:
    """Return the full noun-phrase text rooted at *token*."""
    tokens = sorted(token.subtree, key=lambda t: t.i)
    return " ".join(t.text for t in tokens).strip()


def _ents_in_char_span(
    doc: spacy.tokens.Doc, start: int, end: int
) -> list:
    """
    Return named entities whose character span overlaps [start, end).
    Combines spaCy entities + medical whitelist matches.
    """
    results = []

    # --- spaCy entities ---
    for ent in doc.ents:
        if ent.label_ in KEEP_LABELS:
            if not (ent.end_char <= start or ent.start_char >= end):
                results.append(ent)

    # --- Medical whitelist entities ---
    span_text = doc.text[start:end].lower()
    for term, label in MEDICAL_WHITELIST.items():
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, span_text):
            fake = type('FakeEnt', (), {
                'text': term,
                'label_': label,
                'start_char': start,
                'end_char': end,
            })()
            results.append(fake)

    return results


def extract_relations(doc: spacy.tokens.Doc, source_url: str) -> list[dict]:
    """
    Extract (subject, relation, object) triples from *doc* via dep-parsing.
    Both subject and object must overlap with a named entity,
    including medical whitelist entities.
    """
    rows: list[dict] = []

    for sent in doc.sents:
        for tok in sent:
            if tok.pos_ != "VERB":
                continue

            subjects = [c for c in tok.children if c.dep_ in {"nsubj", "nsubjpass"}]
            if not subjects:
                continue

            # Direct objects
            objects = [c for c in tok.children if c.dep_ in OBJ_DEPS]
            # Prepositional objects (verb → prep → pobj)
            for prep in (c for c in tok.children if c.dep_ == "prep"):
                objects.extend(c for c in prep.children if c.dep_ == "pobj")

            if not objects:
                continue

            for subj in subjects:
                subj_text = _subtree_text(subj)
                subj_start = subj.idx
                subj_end = subj.idx + len(subj_text)
                subj_ents = _ents_in_char_span(doc, subj_start, subj_end)
                if not subj_ents:
                    continue

                for obj in objects:
                    obj_text = _subtree_text(obj)
                    obj_start = obj.idx
                    obj_end = obj.idx + len(obj_text)
                    obj_ents = _ents_in_char_span(doc, obj_start, obj_end)
                    if not obj_ents:
                        continue

                    # Build relation label: verb lemma [+ preposition]
                    relation = tok.lemma_
                    prep_tok = next(
                        (c for c in tok.children if c.dep_ == "prep"), None
                    )
                    if prep_tok:
                        relation = f"{relation} {prep_tok.text}"

                    best_subj = max(subj_ents, key=lambda e: len(e.text))
                    best_obj = max(obj_ents, key=lambda e: len(e.text))

                    rows.append({
                        "subject": best_subj.text,
                        "subject_type": best_subj.label_,
                        "relation": relation,
                        "object": best_obj.text,
                        "object_type": best_obj.label_,
                        "sentence": sent.text.strip(),
                        "source_url": source_url,
                    })

    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_extraction(
    input_file: Path = INPUT_FILE,
    entities_file: Path = ENTITIES_FILE,
    relations_file: Path = RELATIONS_FILE,
    model: str = "en_core_web_trf",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load crawled pages, run NER + relation extraction, save CSVs.

    Returns (df_entities, df_relations).
    """
    entities_file.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading spaCy model: %s", model)
    nlp = spacy.load(model)

    records = load_jsonl(input_file)
    log.info("Loaded %d pages from %s", len(records), input_file)

    all_entities: list[dict] = []
    all_relations: list[dict] = []

    for rec in records:
        url = rec.get("url", "")
        text = rec.get("text", "")
        if not text:
            continue

        log.info("Processing: %s", url)
        doc = nlp(text)

        all_entities.extend(extract_entities(doc, url))
        all_relations.extend(extract_relations(doc, url))

    # ---- Entities ----
    df_ent = (
        pd.DataFrame(all_entities)
        .drop_duplicates()
        .sort_values(["label", "entity"])
        .reset_index(drop=True)
    )
    df_ent.to_csv(entities_file, index=False, encoding="utf-8")
    log.info("Entities saved: %d rows → %s", len(df_ent), entities_file)

    # ---- Relations ----
    df_rel = (
        pd.DataFrame(all_relations)
        .drop_duplicates(subset=["subject", "relation", "object"])
        .reset_index(drop=True)
    )
    df_rel.to_csv(relations_file, index=False, encoding="utf-8")
    log.info("Relations saved: %d rows → %s", len(df_rel), relations_file)

    return df_ent, df_rel


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_extraction()