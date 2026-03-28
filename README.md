# Sepsis Knowledge Graph Project
## Web Data Mining & Semantics — WDM Project

A complete Knowledge Graph pipeline for the medical domain of **Sepsis**, including:
- Web crawling (PubMed, PMC, Wikipedia)
- Named Entity Recognition (spaCy)
- RDF Knowledge Graph construction
- Wikidata alignment & SPARQL expansion
- Knowledge Graph Embedding (KGE) with PyKEEN
- SWRL reasoning
- RAG with local LLM (Gemma 2B + Ollama)

---

## Team
- Koralie
- Estelle

---

## Project Structure
```
WDM_Project/
├── src/
│   ├── crawl/crawler.py          # Web crawler (PubMed + PMC + Wikipedia)
│   ├── ie/extractor.py           # NER extraction (spaCy)
│   ├── kg/builder.py             # RDF graph construction
│   ├── kg/alignment.py           # Wikidata entity alignment
│   ├── kg/predicate_alignment.py # Wikidata predicate alignment
│   ├── kg/expansion.py           # SPARQL KB expansion
│   ├── kge/prepare.py            # KGE data preparation
│   ├── kge/train.py              # KGE training (TransE/DistMult/ComplEx/RotatE)
│   ├── kge/sensitivity.py        # KB size sensitivity analysis
│   ├── rag/lab_rag_sparql_gen.py # RAG with SPARQL generation
│   └── reason/swrl_sepsis.py     # SWRL reasoning
├── notebooks/
│   ├── kge_analysis.ipynb        # KGE embedding analysis
│   └── swrl_reasoning.ipynb      # SWRL + Section 8 comparison
├── data/
│   ├── samples/                  # Crawler output + extracted entities
│   └── kge/                      # Train/valid/test splits + results
├── kg_artifacts/                 # RDF files (ontology, KB, alignment)
└── requirements.txt
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/WDM_Project.git
cd WDM_Project
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install lxml_html_clean
```

### 4. Install Ollama (for RAG)
```bash
brew install ollama
ollama serve &
ollama pull gemma:2b
```

---

## How to Run

### Full pipeline (in order)
```bash
# 1. Crawl data (PubMed + PMC + Wikipedia)
python -m src.crawl.crawler

# 2. Extract entities and relations (NER)
python -m src.ie.extractor

# 3. Build RDF Knowledge Graph
python -m src.kg.builder

# 4. Align entities with Wikidata
python -m src.kg.alignment

# 5. Align predicates with Wikidata
python -m src.kg.predicate_alignment

# 6. Expand KB via SPARQL
python -m src.kg.expansion

# 7. Prepare KGE data (train/valid/test split)
python -m src.kge.prepare

# 8. Train KGE models (TransE, DistMult, ComplEx, RotatE)
python -m src.kge.train

# 9. KB size sensitivity analysis
python -m src.kge.sensitivity

# 10. RAG with SPARQL generation
python -m src.rag.lab_rag_sparql_gen
```

### Notebooks
```bash
jupyter notebook notebooks/kge_analysis.ipynb    # KGE analysis
jupyter notebook notebooks/swrl_reasoning.ipynb  # SWRL reasoning
```

---

## KB Statistics

| Metric | Value |
|--------|-------|
| Total triples | 260,834 |
| URI-URI-URI triples (KGE) | 50,307 |
| Entities | 13,118 |
| Relations | 196 |
| Aligned entities (Wikidata) | 55/55 |

---

## KGE Results

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
|-------|-----|--------|--------|---------|
| TransE | 0.044 | 0.009 | 0.034 | 0.124 |
| **DistMult** | **0.193** | **0.137** | **0.213** | **0.292** |
| ComplEx | 0.033 | 0.015 | 0.034 | 0.064 |
| RotatE | 0.094 | 0.058 | 0.100 | 0.164 |

---

## RAG Results

| Question | Baseline | RAG | Correct? |
|----------|----------|-----|---------|
| What diseases are in the KB? |  No access | ✅ 13 diseases | RAG ✅ |
| What treatments are mentioned? |  No access | ✅ 16 treatments | RAG ✅ |
| What bacteria are in the KB? |  Generic | ✅ 10 bacteria | RAG ✅ |
| What biomarkers are in the KB? |  Wrong | ✅ 16 biomarkers | RAG ✅ |
| What care units are mentioned? |  No access | ✅ 6 care units | RAG ✅ |

---

## Requirements

- Python 3.11+
- Ollama (for RAG)
- ~4GB disk space (models + data)
- ~8GB RAM recommended
