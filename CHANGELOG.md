# Changelog

All notable changes, decisions, and implementation notes for SansRAG (formerly SRAG).

## 2026-05-24 — 15-Sample Evaluation with Normalization

### Changed
- **Evaluation**: Expanded from 10 to 15 samples per dataset (75 total)
  - Unnormalized overall: 0.4431
  - Normalized (minmax) overall: 0.4354
- **README.md**: Updated evaluation section with 15-sample tables, conclusion refined
- **docs/architecture.md**: Updated evaluation section to match
- **generate_report.py**: Chart data updated to 15-sample values
- **PDF report**: Regenerated with updated chart and evaluation data

### Added
- `data/evaluation/eval_unnormalized_15.json` — raw 15-sample unnormalized results
- `data/evaluation/eval_normalized_15.json` — raw 15-sample normalized results
- `data/evaluation/normalization_comparison_15.json` — comparison summary

### Key Findings
- MinMax helps verse-specific (HF Gita: +0.024) and guidance (Gita Guidance: +0.015)
- MinMax hurts factual (Edwin Arnold: -0.022) and commentary (ISKCON: -0.039)
- Overall difference is marginal (-0.008); default remains `minmax` for dominant use case

### Commands
```bash
python evaluate_comprehensive.py --samples 15 --normalize none --output data/evaluation/eval_unnormalized_15.json
python evaluate_comprehensive.py --samples 15 --normalize minmax --output data/evaluation/eval_normalized_15.json
python evaluate_comprehensive.py --compare data/evaluation/eval_unnormalized_15.json data/evaluation/eval_normalized_15.json
```

---

## 2026-05-24 — Feature Normalization (Phase 1)

### Added
- `src/reranking/feature_extractors.py` — three normalization functions:
  - `normalize_features_minmax()` — scales each feature to [0,1]
  - `normalize_features_l2()` — L2 unit norm per feature vector
  - `normalize_features_zscore()` — zero mean, unit variance
  - `normalize_feature_matrix()` — dispatcher by mode
- Integration into `src/reranking/linguistic_reranker.py` — features normalized before weighted scoring
- Config key: `reranking.normalize: "minmax"` (default) — options: `"none"`, `"minmax"`, `"l2"`, `"zscore"`
- API parameter `normalize` on `/api/query` — passed through to pipeline
- UI dropdown in side panel (React) — selects normalization mode
- 13 new tests in `tests/test_reranking.py` — covering all 3 modes + edge cases

### Changed
- `api_server.py` — accepts `normalize` param, passes to pipeline
- `web/src/App.jsx` — dropdown for normalization mode in side panel
- `web/src/App.css` — select/option styles matching parchment theme

### Decision
- Default is `minmax` because it's bounded [0,1], preserves feature ranking, and avoids outlier domination. L2 and zscore available for experimentation.

---

## 2026-05-24 — SQLite Commentary Store & Pipeline Inspector UI

### Added
- `src/storage/commentary_store.py` — `CommentaryStore` class backed by SQLite
  - Verses table (627 verses, indexed by chapter:verse)
  - Commentaries table (1440 commentaries across 3 scholars: Adi Shankaracharya, Swami Vivekananda, Sri Aurobindo)
  - Methods: `get_commentary(chapter, verse)`, `get_commentaries_for_verses(verses)`, `search_by_scholar()`, etc.
  - 8 tests in `tests/test_commentary_store.py`
- `migrate_commentaries.py` — populates SQLite from `data/chunks.jsonl` (extracts embedded commentary data)
- Both SRAGPipeline and SRAGGraphPipeline return intermediate results after each retrieval stage
- API `/api/query` returns `intermediate` (reranked results) and `commentaries` (top-5 commentaries)
- React UI: collapsing side panel with pipeline stages (vector, graph, BM25, fused, reranked)
  - Toggle switches per retrieval method (enable/disable vector, graph, BM25 independently)
  - Commentary display for each verse

### Changed
- `configs/config.yaml` — no changes needed (default weights handle fusion)
- `main.py` — `query_with_intermediates()` returns structured intermediate dict
- `src/langchain_components/graph.py` — LangGraph pipeline tracks intermediate results per stage
- `src/retrieval/hybrid_fusion.py` — accepts toggle dict, skips disabled methods

### Decision
- Commentaries stored separately from Neo4j/FAISS/BM25 to keep retrieval indexes clean and allow independent updates
- SQLite chosen over JSON file for queryability by scholar, verse range, etc.
- Only top-5 reranked verses get commentaries (performance optimization)

---

## 2026-05-24 — Prompt Restructuring (Answer-First)

### Changed
- `src/generation/prompt_templates.py` — system prompt rewritten to enforce:
  1. Explanation-first body (model's own synthesis)
  2. Cited verses as evidence within explanation
  3. Single most relevant commentary only at the very end, prefixed by "---"
- Context-building logic passes verses and commentaries separately with clear labels

### Decision
- Commentary-heavy output was the key UX problem. Solution: relegate commentaries to appendix status.
- Single commentary chosen over multiple to keep answers concise and focused on the model's own understanding.

---

## 2026-05-24 — Project Name Change: SRAG → SansRag

### Changed
- `README.md`, `docs/architecture.md`, `generate_report.py`, `api_server.py` — references updated
- Logo/banner concept: "SansRAG — Sanskrit Retrieval-Augmented Generation"

### Decision
- "SRAG" was generic; "SansRAG" clearly communicates Sanskrit + RAG

---

## 2026-05-24 — Documentation Phase

### Added
- `README.md` — full project documentation with setup, architecture, evaluation results
- `docs/architecture.md` — detailed pipeline documentation with diagrams
- `docs/SRAG_Project_Report.pdf` — 12-section report with 5 diagrams:
  - Architecture diagram (Mermaid)
  - Adaptive weight profiles
  - Evaluation results with normalization
  - Data flow
  - LangGraph state machine
- `.env.example` — template for environment variables
- `LICENSE` — MIT

### Changed
- `generate_report.py` — creates all diagrams and PDF in a single run

---

## 2026-05-24 — Comprehensive Evaluation (5 Datasets)

### Added
- `evaluate_comprehensive.py` — evaluation script supporting multiple datasets
- 5 evaluation datasets:
  1. **Gita Guidance QA** (711 pairs) — modern life Q&A, custom
  2. **Edwin Arnold QA** (500 pairs) — factual Q&A from "The Song Celestial"
  3. **HuggingFace Gita QA** (3,500 pairs) — verse-specific from JDhruv14/Bhagavad-Gita-QA
  4. **Kaggle Gita QA** (12,902 pairs) — modern life Q&A by rambo011
  5. **ISKCON VedaBase** (657 pairs) — verse commentary, Gaudiya Vaishnava tradition

### Changed
- Evaluation metrics: semantic similarity (all-MiniLM-L6-v2), word overlap, verse recall

### Decision
- Semantic similarity chosen as primary metric because ground truth answers vary in style across datasets
- Categories: High (>0.5), Med (0.25-0.5), Low (<0.25)

---

## 2026-05-24 — LangGraph Integration

### Added
- `src/langchain_components/graph.py` — LangGraph state machine
  - 5 nodes: analyze_query, retrieve, rerank, generate, evaluate
  - Conditional edge from evaluate back to retrieve (max 2 iterations)
  - Confidence threshold: 0.3
  - Router node for query type classification
- `configs/config.yaml` — `langgraph` section:
  ```yaml
  langgraph:
    enabled: true
    max_iterations: 2
    confidence_threshold: 0.3
  ```

### Changed
- `main.py` — `SRAGGraphPipeline` class using LangGraph
- `src/retrieval/hybrid_fusion.py` — adaptive RRF weights per query type
- `src/reranking/linguistic_reranker.py` — weight profiles per query type

### Decision
- LangGraph over linear pipeline to enable iterative improvement on low-confidence answers
- 2 iterations max to keep latency acceptable (<10s per query)

---

## 2026-05-24 — Adaptive Re-Ranking with 5 Query Type Profiles

### Added
- `src/reranking/adaptive_reranker.py` — 5 weight profiles:
  - `concept_short`: abstract concepts (weight: vector=0.4/graph=0.4/BM25=0.2)
  - `factual_short`: factual queries (vector=0.5/graph=0.2/BM25=0.3)
  - `complex_long`: multi-part queries (vector=0.3/graph=0.3/BM25=0.4)
  - `concept_medium`: medium concepts (vector=0.35/graph=0.35/BM25=0.3)
  - `general_medium`: general queries (vector=0.4/graph=0.3/BM25=0.3)
- 9 linguistic features per verse-score pair:
  1. Semantic similarity (cosine score from retriever)
  2. Keyword overlap (Jaccard)
  3. Length match (normalized length ratio)
  4. Entity presence (Sanskrit named entities)
  5. Sentiment alignment
  6. Part-of-speech diversity
  7. Readability score (verse complexity)
  8. Concept match (graph relationship overlap)
  9. Query type bias (from router)

### Changed
- `src/reranking/linguistic_reranker.py` — adaptive weight application
- `src/retrieval/hybrid_fusion.py` — query type routing

---

## 2026-05-24 — Bug Fixes

### Fixed
- **Neo4j Cypher syntax**: `size((v)--())` deprecated in Neo4j 5.x → replaced with `OPTIONAL MATCH` + `count()`
- **BM25 tokenization mismatch**: Sanskrit suffix stripping wasn't matching between indexing and query time
  - `_lemmatize_token()` added in `src/retrieval/bm25_retriever.py`
  - Strips: aḥ, am, ena, āya, etc., adds root + "a" forms
- **Morphological profile**: Dead feature in adaptive re-ranker → replaced with actual POS-diversity metric
- **Score propagation**: RRF scores weren't consistently normalized → fixed in `hybrid_fusion.py`
- **Windows UTF-8 encoding**: Added `encoding="utf-8"` to all file operations

---

## 2026-05-24 — Initial Setup

### Added
- Project structure:
  ```
  src/
    langchain_components/  # LangGraph nodes and state
    preprocessing/         # Chunking and parsing
    retrieval/             # FAISS, Neo4j, BM25
    reranking/             # Linguistic feature scoring
    generation/            # Prompt templates, MiMo client
    utils/                 # Logging, helpers
    api.py                 # FastAPI definitions
    pipeline.py            # Core SRAG pipeline
  web/                     # React + Vite frontend
  configs/                 # YAML configuration
  tests/                   # Pytest suite
  data/                    # Chunks, embeddings, indices
  ```

### Key Decisions
- **Embedding model**: `sanganaka/bge-m3-sanskritFT` (1024-dim, Sanskrit-optimized)
- **Generation model**: MiMo v2.5 via OpenAI-compatible API (`mimo-v2.5` — lowercase required)
- **Retrieval**: FAISS (vector) + Neo4j (graph) + rank-bm25 hybrid
- **Script handling**: IAST primary for Neo4j/graph queries, Devanagari for embeddings
- **Frontend**: React + Vite with ancient parchment theme (IM Fell English font, sepia tones, quill icon)
- **Pipeline**: LangGraph state machine for orchestration

### Data
- 3,507 chunks parsed from Bhagavad Gita with verse metadata
- FAISS index built with 1024-dim embeddings
- BM25 index built with Sanskrit-aware tokenization
- Neo4j graph with Verse nodes and relationships
