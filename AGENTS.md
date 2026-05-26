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

## 2026-05-25 — Data Quality Fixes & Range Expansion

### Problem
Dir% (91%) ≠ 100% because ~9% of test verse refs couldn't be found in Neo4j. Two causes:
1. **Range-vs-individual mismatch** (~5.5%): Multi-verse chunks (e.g., "BhG 1.4-6") stored as single Verse nodes, but exact match `v.ref = "BhG 1.5"` failed.
2. **Parser bugs** (~3.5%): Refs in `<l>` tags not extracted, multi-verse divs not split, BhG 8.11 corrupted ref.

### Fix 1: XML Parser — detect refs in `<l>` tags
- **File**: `src/preprocessing/xml_parser.py`
- **Change**: When processing `<lg>` elements, check if any `<l>` starts with "BhG X.Y" and extract as ref.
- **Added**: `parse_verse_range()` helper to extract range bounds from refs like "BhG 1.4-6".
- **Recovered**: 6 verses (BhG 1.12, 3.31, 13.13, 18.11, 18.13, 18.68)

### Fix 2: XML Parser — split multi-verse divs
- **File**: `src/preprocessing/xml_parser.py`
- **Change**: `_extract_div_content` → `_extract_div_contents` returns `list[dict]`, one entry per verse within a div. When a new BhG ref is found mid-div, finalize the current verse and start a new one.
- **Applied to**: `_extract_morpho_div`, `_extract_segmentation_div` too.
- **Recovered**: 4 verses from 2 multi-verse divs (BhG 13.14, 13.15, 13.16 from div 456; BhG 13.12 from div 455)

### Fix 3: XML Parser — fix BhG 8.11 corruption
- **File**: `src/preprocessing/xml_parser.py`
- **Change**: Ref extraction uses `re.match(r"(BhG\s+\d+\.\d+(?:-\d+)?)", text)` instead of `text.strip()`, preventing `<p>` child content from leaking into the ref.
- **Applied to**: All three `_extract_*_div` methods.

### Fix 4: Range expansion in chunker & graph import
- **Files**: `src/preprocessing/chunker.py`, `src/preprocessing/graph_import.py`
- **Change (chunker)**: `create_all_chunks` now expands range refs (e.g., "BhG 1.4-6" → individual "BhG 1.4", "BhG 1.5", "BhG 1.6" chunks). Each chunk keeps the full range text but with an individual ref.
- **Change (graph_import)**: `generate_verse_nodes`, `generate_commentary_nodes`, `generate_relationships` all expand ranges.
- **Result**: 682 verse nodes in Neo4j (up from 627), 686 verse chunks in FAISS (up from 627).

### Fix 5: Sub-verse (pada) chunking
- **File**: `src/preprocessing/chunker.py`
- **Added**: `create_pada_chunks()` — splits verse into per-line chunks with `overlap_lines=1` surrounding context. Each pada chunk has `chunk_type="pada"` and `pada_index` field.
- **Added**: `pada_index` field to `Chunk` dataclass.
- **Result**: 1940 pada chunks created.

### Fix 6: search_commentary_consensus Cypher
- **File**: `src/retrieval/graph_retriever.py`
- **Change**: `DISCUSSES_CONCEPT` (non-existent) → `MENTIONS_CONCEPT` on Verse nodes + `HAS_COMMENTARY` to Commentary nodes.

### Results
- Dir%: **91% → 97%**, WithID-rerank R@1: **91% → 97%**
- Per-dataset: gita_guidance_qa 96%, hf_gita_qa 96%, kaggle_gita_qa 100%, iskcon_vedabase 96%
- Remaining ~3% gap = 17 verses truly absent from source XML (not parser/storage issues)
- To close remaining 3%: need supplementary data source for the missing 17 verses

### Files changed
- `src/preprocessing/xml_parser.py` — `<l>` tag refs, multi-verse div splitting, BhG 8.11 ref fix, `parse_verse_range()` helper
- `src/preprocessing/chunker.py` — range expansion in `create_all_chunks`, `create_pada_chunks()`, `pada_index` field
- `src/preprocessing/graph_import.py` — range expansion in verse/commentary/relationship generation
- `src/retrieval/graph_retriever.py` — fixed `search_commentary_consensus` Cypher
- `main.py` — `verse_only=True` (reverted from False due to ~2hr build time; set to False for all-chunk FAISS)
- `src/langchain_components/graph.py`, `evaluate_retrieval.py` — same verse_only revert

### Note
Full FAISS index with all chunk types (verse + commentary + combined + pada = 5780 vectors) requires ~2 hours to build on CPU. Current setup uses verse-only FAISS (686 vectors) for quick iteration. Set `verse_only=False` in `main.py:113` and `graph.py:119` to rebuild with all chunks.

## 2026-05-26 — Supplement 17 Missing Verses + Validation Filter

### Problem
Dir% stuck at 97% because 17 verses were absent from the source XML. Additionally, gita_guidance_qa dataset had 18 entries with LLM-hallucinated verse refs (e.g., BhG 10.43 where Ch 10 only has 42 verses).

### Fix 1: Supplementary verse fetcher
- **File**: `src/preprocessing/supplement_verses.py` (new)
- **What**: Fetches 17 missing verses from `vedicscriptures.github.io/slok/{ch}/{v}` API (free, returns IAST + Devanagari).
- **Normalization**: Handles API line format — removes `||ch-v||` markers, converts `. ` → `|` line endings, detects speaker.
- **Caching**: Saves to `data/processed/graph_import/supplementary_verses.json` for reuse, avoiding API dependency during builds.
- **Integration**: `main.py:preprocess()` loads supplementary data after XML parsing and appends to verse list before chunk creation.

### Fix 2: Validation filter for test refs
- **Files**: `test_verse_lookup.py`, `evaluate_semantic.py`
- **What**: Added `_is_valid_ref(ref)`, `_filter_valid(refs)` functions using `CHAPTER_VERSES` (standard 700-verse counts) + `_SUPPLEMENTED_VERSES` set. Filters out refs where verse number exceeds chapter max and is not in supplement list.
- **Applied to**: All dataset loaders — `load_gita_guidance_qa`, `load_hf_gita_qa`, `load_kaggle_gita_qa`. Entries with no valid refs are skipped.
- **Impact**: 18 hallucinated refs filtered from gita_guidance_qa (no impact on sample size — still yields 50 samples from 693 valid entries). No invalid refs in hf_gita_qa (0/3500) or kaggle_gita_qa (0/12902).

### Results
- **Verse lookup**: Dir% = WithID-rerank R@1 = **100%** (all 4 datasets, up from 97% pre-supplement).
- **Semantic eval**: WithID Recall@1 = 100%, Sim@top3 = 0.113, Dir% = 100%. NoID R@1 = 28.85% (iskcon_vedabase-driven). All 700 verse nodes present.
- Graph stats: 700 Verse nodes (+73 vs pre-supplement), 5832 total chunks (+52), 5163 relationships.

### Files changed
- `src/preprocessing/supplement_verses.py` — external API fetcher, IAST normalizer, caching (new file)
- `main.py` — loads supplementary verses in `preprocess()`
- `test_verse_lookup.py` — `_is_valid_ref()`, `_filter_valid()`, `_SUPPLEMENTED_VERSES`, `_parse_ref()`
- `evaluate_semantic.py` — same validation functions added to loaders
