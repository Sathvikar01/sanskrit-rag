# SRAG — Sanskrit RAG with Graph-Enhanced Linguistic Re-ranking

A hybrid Retrieval-Augmented Generation system for the **Bhagavad Gita**, combining vector search, knowledge graph traversal, and BM25 lexical matching with adaptive re-ranking and LangGraph orchestration.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full pipeline breakdown.

## Quick Start

### Prerequisites

- Python 3.11+
- Neo4j 5.x running on `bolt://localhost:7687`
- MiMo API key (in `.env` as `MIMO_API_KEY`)
- ~4 GB RAM for embedding model

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "MIMO_API_KEY=your_key_here" > .env

# Preprocess XML corpus (3507 chunks across 18 chapters)
python main.py preprocess

# Build Neo4j knowledge graph
python main.py build-graph

# Build FAISS + BM25 indices (~75 min on CPU)
python main.py build-indices
```

### Query

```bash
# Standard pipeline
python main.py query --query "What is dharma?"

# LangGraph pipeline (iterative expansion)
python main.py query --query "What is dharma?" --langgraph

# Local fallback (no MiMo API)
python main.py query --query "What is dharma?" --local
```

### Web UI

```bash
# Start API server + React frontend
python api_server.py
# Open http://localhost:8000
```

## Project Structure

```
SRAG/
├── configs/config.yaml          # All configuration
├── data/
│   ├── raw/                     # Source XML files
│   ├── processed/               # Chunks, FAISS index, graph import
│   └── evaluation/              # QA datasets + evaluation reports
├── src/
│   ├── preprocessing/           # XML parsing, chunking, concept extraction
│   ├── retrieval/               # Vector (FAISS), BM25, Neo4j graph
│   ├── reranking/               # 9-feature adaptive re-ranking
│   ├── generation/              # MiMo prompt templates + generation
│   ├── langchain_components/    # LangGraph state machine
│   └── utils/                   # Config, logger
├── tests/                       # 48 tests across 4 modules
├── web/                         # React + Vite frontend
├── main.py                      # CLI entry point
├── api_server.py                # FastAPI backend
├── evaluate.py                  # Single-dataset evaluation
└── evaluate_comprehensive.py    # 3-dataset evaluation
```

## How It Works

1. **Query Processing** — MiMo v2.5 converts any language query to IAST, extracts philosophical concepts (dharma, karma, yoga, etc.), and classifies query type
2. **Hybrid Retrieval** — Three parallel searches: FAISS semantic, Neo4j graph traversal, BM25 lexical (with adaptive weights per query type)
3. **Fusion** — Weighted Reciprocal Rank Fusion merges results from all three retrievers
4. **Re-ranking** — 9 linguistic features (vector score, graph centrality, lemma overlap, morphological match, compound match, etc.) with dynamic weight profiles
5. **Answer Generation** — MiMo v2.5 generates a structured markdown answer from the top-5 reranked results, with a brief scholarly context section at the end

## Adaptive Query Types

| Query Type | Example | Vector | Graph | BM25 |
|-----------|---------|--------|-------|------|
| `concept_short` | "What is dharma?" | 0.45 | 0.20 | 0.05 |
| `factual_short` | "Who is Krishna?" | 0.35 | 0.30 | 0.08 |
| `complex_long` | Long philosophical question | 0.35 | 0.15 | 0.20 |
| `concept_medium` | "Explain karma yoga" | 0.40 | 0.22 | 0.08 |
| `general_medium` | Default | 0.40 | 0.18 | 0.12 |

## Evaluation Results

Tested across 3 external datasets (45 samples):

| Dataset | Samples | Semantic Similarity |
|---------|---------|-------------------|
| Gita Guidance QA | 21 | 0.5013 |
| ISKCON VedaBase | 15 | 0.3016 |
| Edwin Arnold QA | 9 | 0.2693 |
| **Overall** | **45** | **0.3884** |

Run your own evaluation:
```bash
python evaluate_comprehensive.py --samples 30 --iskcon-samples 15
```

## Tests

```bash
pytest tests/ -v
# 48 passed
```

## Tech Stack

- **Embeddings**: `sanganaka/bge-m3-sanskritFT` (FAISS)
- **Retrieval**: FAISS + Neo4j 5.x + rank-bm25
- **Re-ranking**: Custom 9-feature adaptive scoring
- **Generation**: MiMo v2.5 (via OpenAI-compatible API)
- **Orchestration**: LangGraph state machine
- **Backend**: FastAPI
- **Frontend**: React + Vite (saffron/gold theme)
