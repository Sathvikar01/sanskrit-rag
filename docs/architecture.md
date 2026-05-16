# SRAG Architecture

## Overview

SRAG is a hybrid Retrieval-Augmented Generation system for the Bhagavad Gita. It combines three retrieval methods (vector, graph, BM25) with adaptive re-ranking and LangGraph orchestration to generate scholarly answers with proper verse citations.

## Data Pipeline

### 1. Source Corpus

Three XML files from the Bhagavad Gita corpus:

- **dataset.xml** — 700 verses (IAST + Devanagari) across 18 chapters with commentaries from 3 scholars (Sridhara Swamin, Visvanatha Chakravarti, Baladeva Vidyabhushana)
- **dataset.lemma-morphosyntax.xml** — Morphological annotations (case, gender, tense, mood) for each token
- **dataset.segmentation-lemma.xml** — Lemma-segmented text mapping surface forms to dictionary forms

### 2. Preprocessing (`src/preprocessing/`)

```
XML files
  ↓ xml_parser.py
Verses + Commentaries + Morphological data
  ↓ chunker.py
3507 chunks (verse + commentary + combined)
  ↓ concept_extractor.py
26 philosophical concepts (dharma, karma, yoga, etc.)
  ↓ graph_builder.py + graph_import.py
Neo4j import files
  ↓
chunks.jsonl (persistent)
```

**Chunking strategy**: Each verse becomes a chunk with:
- IAST text, Devanagari text, word count
- Lemmas (for BM25 matching)
- Morphological profile (case, gender, tense)
- Concept associations
- Chunk type: `verse`, `commentary`, or `combined`

### 3. Index Building

#### Vector Index (`src/retrieval/vector_store.py`)

- Model: `sanganaka/bge-m3-sanskritFT` (Sanskrit fine-tuned BGE-M3)
- Embedding dimension: 1024
- Storage: FAISS `IndexFlatIP` (inner product / cosine similarity)
- Encodes Devanagari text of each chunk
- Output: `verse_vectors.faiss` + `verse_metadata.json`

#### BM25 Index (`src/retrieval/bm25_retriever.py`)

- Library: `rank-bm25`
- Tokenization: IAST-aware tokenization with Sanskrit suffix stripping
- Query expansion: Expands query tokens to candidate lemmas (e.g., "karmaṇi" → "karma" + "a")
- Index: In-memory BM25Okapi over lemma tokens

#### Knowledge Graph (`src/retrieval/graph_retriever.py`)

- Database: Neo4j 5.x
- Node types: `Verse`, `Chapter`, `Concept`, `Commentator`
- Relationships: `VERSE_OF_CHAPTER`, `HAS_CONCEPT`, `HAS_COMMENTARY`, `MENTIONS_CONCEPT`
- Full-text index: `verse_text_ft` on `text_iast` field
- Query methods:
  - Full-text search on IAST text
  - Concept neighborhood traversal (verse → concept → related verses)
  - Combined search (full-text + concept, deduplicated)

## Query Pipeline

### Standard Pipeline (`main.py:SRAGPipeline`)

```
User Query (any language)
  ↓ QueryProcessor
  ├── Language detection (English/Hindi/IAST/Devanagari)
  ├── IAST conversion via MiMo v2.5
  ├── Concept extraction (dharma, karma, yoga, etc.)
  └── Query type classification
  ↓
  ├── Vector Search (FAISS, Devanagari query)
  ├── Graph Search (Neo4j, IAST + concepts)
  └── BM25 Search (rank-bm25, IAST + expanded lemmas)
  ↓ HybridRetriever.fuse_results()
  ├── Adaptive RRF weights based on query type
  └── Reciprocal Rank Fusion
  ↓ LinguisticReranker.rerank()
  ├── 9 feature extraction per candidate
  ├── Dynamic weight selection (per query type)
  └── Top-5 selected
  ↓ AnswerGenerator.generate()
  ├── Build prompt (verses separated from commentaries)
  ├── MiMo v2.5 completion
  └── Extract verse citations + confidence
  ↓
Response with answer, citations, top verses, confidence
```

### LangGraph Pipeline (`src/langchain_components/graph.py`)

State machine with conditional routing:

```
process_query → retrieve → fuse → rerank → [expand?] → generate → END
                                    ↑
                                    └── expand (if confidence < 0.3, max 2 iterations)
```

**Nodes:**
1. `process_query` — IAST conversion + concept extraction
2. `retrieve` — Vector + graph + BM25 parallel search
3. `fuse` — Adaptive RRF fusion
4. `rerank` — 9-feature linguistic re-ranking
5. `generate` — MiMo v2.5 answer generation

**Conditional edge**: If average reranking confidence < 0.3, loop back to `process_query` with expanded concepts (up to 2 iterations).

## Re-ranking

### 9 Features (`src/reranking/linguistic_reranker.py`)

| # | Feature | Weight (default) | Description |
|---|---------|-----------------|-------------|
| 1 | `score_vector` | 0.40 | FAISS cosine similarity |
| 2 | `score_graph` | 0.20 | Graph retrieval score |
| 3 | `score_bm25` | 0.10 | BM25 lexical score |
| 4 | `score_lemma` | 0.12 | Query-document lemma overlap |
| 5 | `score_morph` | 0.08 | Morphological case match (Sanskrit suffix analysis) |
| 6 | `score_compound` | 0.03 | Bigram/compound match score |
| 7 | `score_commentary` | 0.04 | Commentary chunk relevance |
| 8 | `score_concept` | 0.02 | Concept overlap score |
| 9 | `score_graph_centrality` | 0.01 | Graph node centrality (degree / max_degree) |

### Adaptive Profiles (`src/reranking/adaptive_reranker.py`)

Weights shift based on query type:

- **concept_short** → Vector 0.45 (semantic similarity dominates)
- **factual_short** → Graph 0.30 (graph relationships matter)
- **complex_long** → BM25 0.20 (exact term matching for specificity)
- **concept_medium** → Balanced (vector 0.40, graph 0.22)
- **general_medium** → Balanced default

## Retrieval Fusion

### Reciprocal Rank Fusion (RRF)

```python
rrf_score = sum(weight_i / (k + rank_i))  # k = 60
```

Each document gets an RRF score from each retriever, weighted by the adaptive profile. Documents appearing in multiple retrievers get boosted scores.

### Score Propagation

Each fused result carries individual scores:
- `vector_score` — Original FAISS similarity
- `graph_score` — Neo4j graph score
- `bm25_score` — BM25 lexical score
- `rrf_score` — Combined fusion score
- `sources` — Which retrievers found this document

## Generation

### Prompt Structure (`src/generation/prompt_templates.py`)

**System prompt** instructs:
1. Explanation-first: Model writes its own synthesis of retrieved verses
2. Verses as evidence: Quote 1-2 lines, explain in plain language
3. Commentaries as appendix: Single most relevant commentary at the end
4. Markdown formatting with headings, bold key terms, bullet points

**User prompt sections:**
```
## User Question
{query}

## Retrieved Verses (Primary Source Material)
{verse entries with IAST + Devanagari}

## Traditional Commentary (Reference Only)
{single highest-confidence commentary}
```

### Model

- Provider: MiMo v2.5 (via OpenAI-compatible API)
- API: `https://api.xiaomimimo.com/v1`
- Temperature: 0.3 (low creativity, high factual accuracy)
- Max tokens: 2048

## Confidence Scoring

### Pipeline Confidence (`src/reranking/confidence.py`)

```python
overall = 0.3 × retrieval + 0.5 × reranking + 0.2 × generation
```

- **Retrieval confidence**: Max score from fusion, normalized with sigmoid
- **Reranking confidence**: Average of top-5 reranking scores
- **Generation confidence**: Based on citation count (1.0 if ≥ 3 citations, decreases otherwise)

### Query Expansion Decision

If `reranking_confidence < 0.3`:
- Expand concepts (add related concepts from concept extractor)
- Re-run retrieval + reranking
- Max 2 iterations before forced generation

## Configuration

All settings in `configs/config.yaml`:

- `langgraph.enabled: true` — Use LangGraph pipeline
- `langgraph.max_iterations: 2` — Max expansion iterations
- `langgraph.confidence_threshold: 0.3` — Threshold for expansion
- `retrieval.adaptive_weights: true` — Enable adaptive fusion weights
- `reranking.adaptive: true` — Enable adaptive re-ranking weights
- `generation.provider: "mimo"` — LLM provider

## Evaluation

### Datasets

- **Gita Guidance QA** (711 pairs) — Modern life questions answered with Gita wisdom
- **Edwin Arnold QA** (500 pairs) — Short factual questions from Edwin Arnold's "The Song Celestial"
- **ISCKON VedaBase** (657 entries) — Verse-by-verse commentary from Gaudiya Vaishnava tradition

### Metrics

- **Semantic similarity**: `all-MiniLM-L6-v2` cosine similarity between SRAG answer and ground truth
- **Word overlap**: Jaccard-style word overlap
- **Verse recall**: Fraction of ground-truth verses cited by SRAG

### Results (45 samples)

| Source | Avg Semantic Similarity |
|--------|------------------------|
| Gita Guidance QA | 0.5013 |
| ISCKON VedaBase | 0.3016 |
| Edwin Arnold QA | 0.2693 |
| **Overall** | **0.3884** |
