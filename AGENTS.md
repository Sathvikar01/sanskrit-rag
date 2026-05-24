# Session Log

## 2026-05-24 — Verse ID Retrieval Fixes + Rule-Based LangGraph Node

### Problem
Dir% in Neo4j (~78%) far exceeded WithID-rerank R@1 (~14%). Two root causes identified.

### Fix 1: `search_by_verse_ref` — CONTAINS → exact match
- **File**: `src/retrieval/graph_retriever.py`
- **Change**: Cypher query `v.ref CONTAINS $ref` → `v.ref = $ref`
- **Why**: CONTAINS over-matched (e.g., "BhG 14.1" matched "BhG 14.10"–"BhG 14.19"), polluting graph results with false positives.
- **Added**: `normalize_verse_ref()` — normalizes "BG X.Y" → "BhG X.Y", strips whitespace, handles sanskrit/Bhagavad Gita prefixes.
- **Added**: Returns `graph_score=100.0` and `chunk_type="verse"` directly in search results.

### Fix 2: Fusion — field merge + top-k promotion
- **File**: `src/retrieval/hybrid_fusion.py`
- **Change 1 (field merge)**: `data_map` changed from first-source-wins to field-level merge. Later sources (graph) fill in `verse_ref`, `chunk_type` etc. missing from earlier sources (vector).
- **Change 2 (promotion)**: In `fuse_results`, entries with `graph_score > 50` are guaranteed a spot in the top-k, overriding the lowest-RRF non-verse entry if needed.
- **Why**: RRF rank-based scoring under-weights graph-only results, so exact Neo4j matches got truncated. Vector results lack `verse_ref`/`chunk_type`, so `check_verse_in_reranked` filtered them out.

### Fix 3: Rule-based LangGraph node replaces short-circuit hack
- **Files**: `src/langchain_components/graph.py`, `src/langchain_components/state.py`
- **What**: Removed `graph_score > 50` threshold-based short-circuit from `_node_rerank`.
- **Replaced with**: New `_node_verse_ref_retrieval` that uses regex (`(?:BhG|BG) \d+\.\d+`) to detect verse refs in the query. If found, calls `search_by_verse_ref` and skips the entire `retrieve → fuse → rerank` pipeline.
- **Flow**:
  ```
  process_query → verse_ref_check → generate (if verse found)
                  ↓ (else)
                retrieve → fuse → rerank → expand↺ or generate
  ```
- **Why**: Cleaner separation of concerns (each node one job), more efficient (skips expensive retrieval), no magic score thresholds, self-contained regex check.

### Test refactoring
- **File**: `test_verse_lookup.py`
- **Change**: Replaced manual component setup with `SRAGGraphPipeline`. Removed standalone short-circuit in favor of calling `pipeline._node_verse_ref_retrieval` directly.

### Results
- Gap closed: Dir% (78%) ≈ WithID-rerank R@1 (78.26%)
- Remaining failures are Neo4j misses (verse not in graph), not pipeline issues
- Rule-based node correctly intercepts "BhG X.Y: question" queries

### Files changed
- `src/retrieval/graph_retriever.py` — exact match, normalizer, graph_score/chunk_type fields
- `src/retrieval/hybrid_fusion.py` — field merge in both fusion functions, promotion in fuse_results
- `src/langchain_components/state.py` — added `verse_ref_detected`, `verse_ref` fields
- `src/langchain_components/graph.py` — rule-based node, router, removed short-circuit
- `test_verse_lookup.py` — pipeline-based test, no more standalone short-circuit
