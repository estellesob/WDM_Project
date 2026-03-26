"""
src/kg/builder.py
=================
RDF Knowledge Graph builder for the Sepsis Knowledge Graph.

Reads:
    data/samples/extracted_knowledge.csv   – entities (nodes)
    data/samples/extracted_relations.csv   – relations (edges)

Produces:
    kg_artifacts/ontology.ttl    – RDFS schema (classes + properties)
    kg_artifacts/sepsis_kg.ttl   – the actual knowledge graph (instances)

Usage (standalone):
    python -m src.kg.builder
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import XSD

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENTITIES_FILE = Path("data/samples/extracted_knowledge.csv")
RELATIONS_FILE = Path("data/samples/extracted_relations.csv")
ONTOLOGY_FILE = Path("kg_artifacts/ontology.ttl")
KG_FILE = Path("kg_artifacts/sepsis_kg.ttl")

# Base namespaces
BASE = Namespace("http://sepsis-kg.org/")
PROP = Namespace("http://sepsis-kg.org/prop/")
TYPE = Namespace("http://sepsis-kg.org/type/")

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """
    Convert a string to a valid URI fragment.
    Example: "Staphylococcus aureus" -> "Staphylococcus_aureus"
    """
    text = text.strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text


def entity_uri(entity: str) -> URIRef:
    """Return the URI for an entity."""
    return BASE[slugify(entity)]


def property_uri(relation: str) -> URIRef:
    """Return the URI for a property/relation."""
    return PROP[slugify(relation)]


def class_uri(label: str) -> URIRef:
    """Return the URI for a class (entity type)."""
    return TYPE[label]


# ---------------------------------------------------------------------------
# Ontology builder
# ---------------------------------------------------------------------------

def build_ontology() -> Graph:
    """
    Build the RDFS schema: declare all classes and properties.
    Returns an RDF graph.
    """
    g = Graph()
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("type", TYPE)
    g.bind("prop", PROP)
    g.bind("base", BASE)

    # --- Declare classes ---
    classes = {
        "Disease":    "A medical condition or disease (e.g. sepsis, pneumonia)",
        "Bacteria":   "A bacterial pathogen (e.g. E. coli, Staphylococcus aureus)",
        "Biomarker":  "A biological marker used in diagnosis (e.g. procalcitonin)",
        "Treatment":  "A medical treatment or drug (e.g. vancomycin, antibiotics)",
        "CareUnit":   "A medical care setting (e.g. ICU, NICU)",
        "Person":     "A researcher or clinician",
        "Org":        "An organization or institution",
        "Location":   "A geographical location",
        "Date":       "A date or time period",
        "Product":    "A medical product or tool",
    }

    for class_name, comment in classes.items():
        class_ref = class_uri(class_name)
        g.add((class_ref, RDF.type, RDFS.Class))
        g.add((class_ref, RDFS.label, Literal(class_name)))
        g.add((class_ref, RDFS.comment, Literal(comment)))

    # --- Declare core properties ---
    properties = {
        "causes":          ("Disease", "Disease",   "A pathogen or condition causes a disease"),
        "treats":          ("Treatment", "Disease", "A treatment is used to treat a disease"),
        "isBiomarkerOf":   ("Biomarker", "Disease", "A biomarker indicates a disease"),
        "isAssociatedWith":("Disease", "Disease",   "Two conditions are associated"),
        "isolatedFrom":    ("Bacteria", "CareUnit", "A bacterium isolated from a care setting"),
        "conductedAt":     ("Org", "Location",      "A study conducted at a location"),
        "enrolledIn":      ("CareUnit", "Disease",  "Patients enrolled in a study"),
    }

    for prop_name, (domain, range_, comment) in properties.items():
        prop_ref = property_uri(prop_name)
        g.add((prop_ref, RDF.type, RDF.Property))
        g.add((prop_ref, RDFS.label, Literal(prop_name)))
        g.add((prop_ref, RDFS.domain, class_uri(domain)))
        g.add((prop_ref, RDFS.range, class_uri(range_)))
        g.add((prop_ref, RDFS.comment, Literal(comment)))

    log.info("Ontology built: %d triples", len(g))
    return g


# ---------------------------------------------------------------------------
# Label → class mapping
# ---------------------------------------------------------------------------

# Maps NER labels to ontology classes
LABEL_TO_CLASS = {
    "DISEASE":   "Disease",
    "BACTERIA":  "Bacteria",
    "BIOMARKER": "Biomarker",
    "TREATMENT": "Treatment",
    "CARE_UNIT": "CareUnit",
    "PERSON":    "Person",
    "ORG":       "Org",
    "GPE":       "Location",
    "LOC":       "Location",
    "DATE":      "Date",
    "PRODUCT":   "Product",
}


# ---------------------------------------------------------------------------
# KG builder
# ---------------------------------------------------------------------------

def build_kg(
    entities_file: Path = ENTITIES_FILE,
    relations_file: Path = RELATIONS_FILE,
) -> Graph:
    """
    Build the knowledge graph from extracted entities and relations.
    Returns an RDF graph.
    """
    g = Graph()
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("type", TYPE)
    g.bind("prop", PROP)
    g.bind("base", BASE)

    # --- Load CSVs ---
    df_ent = pd.read_csv(entities_file)
    df_rel = pd.read_csv(relations_file)
    log.info("Loaded %d entities and %d relations", len(df_ent), len(df_rel))

    # --- Add entities as RDF instances ---
    for _, row in df_ent.iterrows():
        entity = str(row["entity"])
        label = str(row["label"])
        source = str(row["source_url"])

        uri = entity_uri(entity)
        class_name = LABEL_TO_CLASS.get(label, "Product")
        class_ref = class_uri(class_name)

        g.add((uri, RDF.type, class_ref))
        g.add((uri, RDFS.label, Literal(entity)))
        g.add((uri, PROP.sourceUrl, Literal(source, datatype=XSD.anyURI)))

    log.info("Entities added: %d triples so far", len(g))

    # --- Add relations as RDF triples ---
    for _, row in df_rel.iterrows():
        subject = str(row["subject"])
        relation = str(row["relation"])
        obj = str(row["object"])
        sentence = str(row["sentence"])
        source = str(row["source_url"])

        subj_uri = entity_uri(subject)
        obj_uri = entity_uri(obj)
        prop_ref = property_uri(relation)

        # Skip self-relations (subject == object)
        if subject.lower() == obj.lower():
            continue

        g.add((subj_uri, prop_ref, obj_uri))
        g.add((prop_ref, RDF.type, RDF.Property))
        g.add((prop_ref, RDFS.label, Literal(relation)))

        # Add provenance: which sentence this came from
        g.add((subj_uri, PROP.extractedFrom, Literal(sentence)))
        g.add((subj_uri, PROP.sourceUrl, Literal(source, datatype=XSD.anyURI)))

    log.info("Relations added: %d triples total", len(g))
    return g


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_builder(
    ontology_file: Path = ONTOLOGY_FILE,
    kg_file: Path = KG_FILE,
) -> tuple[Graph, Graph]:
    """
    Build ontology + KG, save both as Turtle files.
    Returns (ontology_graph, kg_graph).
    """
    ontology_file.parent.mkdir(parents=True, exist_ok=True)

    # Build and save ontology
    ontology = build_ontology()
    ontology.serialize(destination=str(ontology_file), format="turtle")
    log.info("Ontology saved → %s (%d triples)", ontology_file, len(ontology))

    # Build and save KG
    kg = build_kg()
    kg.serialize(destination=str(kg_file), format="turtle")
    log.info("KG saved → %s (%d triples)", kg_file, len(kg))

    return ontology, kg


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_builder()