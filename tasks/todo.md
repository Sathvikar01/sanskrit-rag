# SansRAG Implementation Tasks

## Project Overview
- **Goal**: Embed Sanskrit datasets using BGE-M3 and retrieve with hybrid search
- **Embeddings**: Dense (1024-dim) + Sparse (lexical) + Multi-vec (Colbert)
- **Indices**: HNSW + IVF + BM25
- **Regularization**: L1/L2 with adaptive tuning

---

## Phase 1: Project Setup
- [x] Create project structure
- [x] Create `requirements.txt`
- [x] Create `config/settings.py`
- [x] Create Docker Compose for Milvus

## Phase 2: Data Processing
- [x] Implement `TEIXMLParser` class
- [x] Implement `SemanticChunker` with token counting
- [x] Add verse ID extraction
- [x] Support all three dataset types

## Phase 3: Embeddings
- [x] Implement `NVIDIAEmbeddingClient`
- [x] Dense embedding generation (1024-dim)
- [x] Sparse embedding generation (lexical features)
- [x] Colbert multi-vec generation (128-dim)
- [x] Embedding caching to disk
- [x] L1/L2 regularization functions

## Phase 4: Vector Storage
- [x] Implement `MilvusManager`
- [x] Create collection schema
- [x] HNSW index creation (M=16, efConstruction=256)
- [x] Sparse inverted index creation
- [x] IVF index for Colbert vectors
- [x] BM25 search support
- [x] Insert embeddings batch processing

## Phase 5: Hybrid Retrieval
- [x] Implement `HybridRetriever`
- [x] Dense search (HNSW)
- [x] Sparse search (inverted index)
- [x] Colbert search (IVF)
- [x] BM25 search
- [x] RRF (Reciprocal Rank Fusion)
- [x] L1 regularization in scoring
- [x] L2 regularization in scoring
- [x] Combined regularization
- [x] `RegularizedRetriever` with adaptive tuning

## Phase 6: Testing
- [x] `test_xml_parser.py`
- [x] `test_embedding_client.py` (L1/L2 tests)
- [x] `test_milvus_manager.py`
- [x] `test_retriever.py` (regularization tests)
- [x] `test_integration.py`

## Phase 7: Pipeline Orchestration
- [x] Implement `SansRAGPipeline` class
- [x] Milvus Docker startup
- [x] Dataset ingestion pipeline
- [x] Interactive REPL with commands
- [x] Result display (pretty-print)
- [x] Result save (JSON)
- [x] Test query runner
- [x] CLI argument parsing

---

## Verification Checklist
- [ ] Run `pip install -r requirements.txt`
- [ ] Run `docker-compose up -d` in `docker/`
- [ ] Run `python src/main.py --ingest --test`
- [ ] Verify test queries return results
- [ ] Check regularization parameters adapt

---

## Current Status
**COMPLETED** - All core components implemented.

## Next Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Start Milvus: `cd docker && docker-compose up -d`
3. Ingest data: `python src/main.py --ingest`
4. Interactive search: `python src/main.py`
