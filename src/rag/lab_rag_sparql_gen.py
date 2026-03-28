"""

RAG with RDF/SPARQL and a local small LLM (Gemma 2B via Ollama).

Pipeline:
1. Load RDF knowledge graph (expanded.nt)
2. Build schema summary (prefixes, predicates, classes)
3. Baseline: ask LLM directly (no KG)
4. RAG: NL → SPARQL → rdflib → grounded answer
5. Self-repair loop if SPARQL fails
6. CLI demo

Usage:
    python -m src.rag.lab_rag_sparql_gen
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import List, Tuple

import requests
from rdflib import Graph

# Configuration


KG_FILE     = Path("kg_artifacts/sepsis_kg.ttl")
OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

MAX_PREDICATES = 80
MAX_CLASSES    = 40
SAMPLE_TRIPLES = 20


# 0) Call local LLM (Ollama)


def ask_local_llm(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Send a prompt to Ollama and return the response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama API error {response.status_code}: {response.text}"
        )
    return response.json().get("response", "")


# 1) Load RDF graph


def load_graph(path: Path = KG_FILE) -> Graph:
    """Load our Sepsis KB from Turtle format."""
    g = Graph()
    g.parse(str(path), format="turtle")
    print(f"Loaded {len(g)} triples from {path}")
    return g


# 2) Build schema summary


def get_prefix_block(g: Graph) -> str:
    defaults = {
        "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl":  "http://www.w3.org/2002/07/owl#",
        "base": "http://sepsis-kg.org/",
        "prop": "http://sepsis-kg.org/prop/",
        "type": "http://sepsis-kg.org/type/",
    }
    ns_map = {p: str(ns) for p, ns in g.namespace_manager.namespaces()}
    for k, v in defaults.items():
        ns_map.setdefault(k, v)
    lines = [f"PREFIX {p}: <{ns}>" for p, ns in sorted(ns_map.items())]
    return "\n".join(lines)


def list_distinct_predicates(g: Graph, limit: int = MAX_PREDICATES) -> List[str]:
    q = f"SELECT DISTINCT ?p WHERE {{ ?s ?p ?o . }} LIMIT {limit}"
    return [str(row.p) for row in g.query(q)]


def list_distinct_classes(g: Graph, limit: int = MAX_CLASSES) -> List[str]:
    q = f"SELECT DISTINCT ?cls WHERE {{ ?s a ?cls . }} LIMIT {limit}"
    return [str(row.cls) for row in g.query(q)]


def sample_triples(g: Graph, limit: int = SAMPLE_TRIPLES) -> List[Tuple]:
    q = f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . }} LIMIT {limit}"
    return [(str(r.s), str(r.p), str(r.o)) for r in g.query(q)]


def build_schema_summary(g: Graph) -> str:
    """Build a compact schema description for prompting the LLM."""
    prefixes = get_prefix_block(g)
    preds    = list_distinct_predicates(g)
    classes  = list_distinct_classes(g)
    samples  = sample_triples(g)

    pred_lines   = "\n".join(f"- {p}" for p in preds)
    class_lines  = "\n".join(f"- {c}" for c in classes)
    sample_lines = "\n".join(f"- {s} {p} {o}" for s, p, o in samples)

    return f"""
{prefixes}

# Predicates (up to {MAX_PREDICATES})
{pred_lines}

# Classes / rdf:type (up to {MAX_CLASSES})
{class_lines}

# Sample triples (up to {SAMPLE_TRIPLES})
{sample_lines}
""".strip()

# 3) NL : SPARQL prompting


SPARQL_INSTRUCTIONS = """
You are a SPARQL generator for a Sepsis medical knowledge graph.

ONLY use these prefixes:
  PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
  PREFIX type: <http://sepsis-kg.org/type/>
  PREFIX prop: <http://sepsis-kg.org/prop/>
  PREFIX base: <http://sepsis-kg.org/>

EXACT query patterns to use:
- diseases:    SELECT ?d WHERE { ?d rdf:type type:Disease . }
- treatments:  SELECT ?t WHERE { ?t rdf:type type:Treatment . }
- bacteria:    SELECT ?b WHERE { ?b rdf:type type:Bacteria . }
- biomarkers:  SELECT ?bm WHERE { ?bm rdf:type type:Biomarker . }
- care units:  SELECT ?cu WHERE { ?cu rdf:type type:CareUnit . }

Return ONLY the SPARQL in a ```sparql block. Nothing else.
"""

def make_sparql_prompt(schema: str, question: str) -> str:
    return f"""{SPARQL_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema}

QUESTION:
{question}

Return only the SPARQL query in a ```sparql code block.
"""


CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sparql(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    if m:
        query = m.group(1).strip()
    else:
        query = text.strip()
    
    # Remove garbage tokens Gemma adds at the end
    for garbage in ["</start_of_turn>", "<start_of_turn>", "```", "<end_of_turn>"]:
        query = query.replace(garbage, "").strip()
    
    return query


def generate_sparql(question: str, schema: str) -> str:
    raw = ask_local_llm(make_sparql_prompt(schema, question))
    return extract_sparql(raw)


# 4) Execute SPARQL + self-repair


def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    res  = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows  = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows


REPAIR_INSTRUCTIONS = """
The SPARQL query below failed. Using the SCHEMA SUMMARY and the ERROR,
return a corrected SPARQL 1.1 SELECT query.
- Use only known prefixes/IRIs from the schema.
- Keep it simple and robust.
- Return ONLY the corrected SPARQL in a ```sparql code block.
"""


def repair_sparql(schema: str, question: str, bad_query: str, error: str) -> str:
    prompt = f"""{REPAIR_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema}

QUESTION:
{question}

BAD SPARQL:
{bad_query}

ERROR:
{error}

Return only the corrected SPARQL in a code block.
"""
    raw = ask_local_llm(prompt)
    return extract_sparql(raw)


# 5) RAG pipeline


def answer_with_rag(
    g: Graph,
    schema: str,
    question: str,
    try_repair: bool = True,
) -> dict:
    """Full RAG pipeline: NL → SPARQL → execute → (repair if needed)."""
    sparql = generate_sparql(question, schema)

    try:
        vars_, rows = run_sparql(g, sparql)
        return {
            "query": sparql, "vars": vars_, "rows": rows,
            "repaired": False, "error": None,
        }
    except Exception as e:
        err = str(e)
        if try_repair:
            repaired = repair_sparql(schema, question, sparql, err)
            try:
                vars_, rows = run_sparql(g, repaired)
                return {
                    "query": repaired, "vars": vars_, "rows": rows,
                    "repaired": True, "error": None,
                }
            except Exception as e2:
                return {
                    "query": repaired, "vars": [], "rows": [],
                    "repaired": True, "error": str(e2),
                }
        return {
            "query": sparql, "vars": [], "rows": [],
            "repaired": False, "error": err,
        }


# 6) Baseline (no RAG)


def answer_baseline(question: str) -> str:
    """Ask the LLM directly without any KG context."""
    prompt = f"Answer the following question as best as you can:\n\n{question}"
    return ask_local_llm(prompt)



# 7) Pretty print


def pretty_print(result: dict) -> None:
    if result.get("error"):
        print(f"\n[Error] {result['error']}")

    print(f"\n[SPARQL Query]")
    print(result["query"])
    print(f"\n[Repaired?] {result['repaired']}")

    vars_ = result.get("vars", [])
    rows  = result.get("rows", [])

    if not rows:
        print("\n[No results returned]")
        return

    print(f"\n[Results] ({len(rows)} rows)")
    print(" | ".join(vars_))
    print("-" * 60)
    for r in rows[:20]:
        print(" | ".join(r))
    if len(rows) > 20:
        print(f"... ({len(rows)} total rows)")


# 8) Evaluation : 5 test questions


TEST_QUESTIONS = [
    "What diseases are in the knowledge graph?",
    "What treatments are mentioned in the knowledge graph?",
    "What bacteria are associated with sepsis?",
    "What biomarkers are in the knowledge graph?",
    "What care units are mentioned in the knowledge graph?",
]


def run_evaluation(g: Graph, schema: str) -> None:
    """Run evaluation on 5 test questions and compare baseline vs RAG."""
    print("\n" + "=" * 70)
    print("EVALUATION — Baseline vs SPARQL-generation RAG")
    print("=" * 70)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}: {question}")
        print(f"{'='*70}")

        print("\n--- Baseline (No RAG) ---")
        baseline = answer_baseline(question)
        print(baseline[:300] + "..." if len(baseline) > 300 else baseline)

        print("\n--- SPARQL-generation RAG ---")
        result = answer_with_rag(g, schema, question)
        pretty_print(result)


# CLI demo


if __name__ == "__main__":
    # Load graph
    g      = load_graph()
    schema = build_schema_summary(g)

    print("\n" + "=" * 70)
    print("Sepsis KG — RAG with SPARQL generation (Gemma 2B + rdflib)")
    print("=" * 70)
    print(f"Graph: {KG_FILE} | Model: {OLLAMA_MODEL}")

    # Run evaluation on 5 questions
    run_evaluation(g, schema)

    # Interactive CLI
    print("\n" + "=" * 70)
    print("Interactive mode — type your question (or 'quit' to exit)")
    print("=" * 70)

    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break

        print("\n--- Baseline (No RAG) ---")
        print(answer_baseline(q))

        print("\n--- SPARQL-generation RAG ---")
        result = answer_with_rag(g, schema, q)
        pretty_print(result)