"""
src/rag/app.py
==============
Flask web interface for the Sepsis KG RAG chatbot.

Usage:
    python -m src.rag.app
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.lab_rag_sparql_gen import (
    load_graph, build_schema_summary,
    answer_with_rag, answer_baseline, ask_local_llm,
)

app = Flask(__name__, template_folder="templates")

# Load graph once at startup
print("Loading Sepsis Knowledge Graph...")
g      = load_graph()
schema = build_schema_summary(g)
print("Ready!")


def format_nl_answer(question: str, rows: list) -> str:
    """Ask LLM to formulate a natural language answer from SPARQL results."""
    if not rows:
        return "No results found in the knowledge graph for this question."

    results_text = ", ".join(
        " / ".join(
            cell.split("/")[-1].replace("_", " ").title()
            for cell in row
        )
        for row in rows[:15]
    )

    prompt = f"""You are a medical assistant. Based on the following data retrieved 
from a Sepsis knowledge graph, answer the question in 1-2 clear and informative sentences.

Question: {question}
Data from knowledge graph: {results_text}

Write a concise answer using only the data above. Do not add information not present in the data."""

    return ask_local_llm(prompt)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data     = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # Baseline — LLM without KB
    baseline = answer_baseline(question)

    # RAG — LLM + KB via SPARQL
    result = answer_with_rag(g, schema, question)

    # Format raw results
    rows_formatted = []
    if result.get("rows"):
        for row in result["rows"][:20]:
            formatted = [
                cell.split("/")[-1].replace("_", " ").title()
                for cell in row
            ]
            rows_formatted.append(formatted)

    # Natural language answer from KB results
    nl_answer = format_nl_answer(question, result.get("rows", []))

    return jsonify({
        "baseline":  baseline,
        "nl_answer": nl_answer,
        "sparql":    result.get("query", ""),
        "vars":      result.get("vars", []),
        "rows":      rows_formatted,
        "repaired":  result.get("repaired", False),
        "error":     result.get("error"),
        "row_count": len(result.get("rows", [])),
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
