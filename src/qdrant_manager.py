"""Qdrant Vector Store Manager with HNSW, IVF, and real BM25 indices.

Enhanced for:
- Semantic embeddings with dense/sparse vectors
- Verse-wise embeddings for graph nodes
- Separate collections for commentaries
- Metadata-enriched payloads for filtering
"""
import math
import os
import time
import json
import re
import hashlib
import threading
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    QDRANT_HOST,
    QDRANT_PORT,
    DENSE_DIM,
    COLBERT_DIM,
    MAX_TEXT_LENGTH,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    COLLECTION_NAMES
)

BM25_INDEX_VERSION = 2

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        SparseVector,
        SparseIndexParams,
        SparseVectorParams,
        HnswConfigDiff,
        OptimizersConfigDiff,
        Filter,
        FieldCondition,
        MatchValue,
        MatchAny,
        SearchRequest,
        ScoredPoint,
        PayloadSchemaType,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: qdrant-client not installed. Run: pip install qdrant-client")


@dataclass
class SearchResult:
    """Container for search results."""
    id: str
    text: str
    score: float
    dataset_type: str
    verse_id: Optional[str]
    metadata: Dict[str, Any]


class QdrantManager:
    """Manager for Qdrant vector database operations.

    Provides semantic relevance search via dense embeddings,
    sparse lexical matching, and real BM25 term-frequency search.
    """

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        prefer_grpc: bool = False,
        path: str = None,
        collection_name: str = "sansr_seg_lemma"
    ):
        self.host = host
        self.port = port
        self.prefer_grpc = prefer_grpc
        env_local_path = os.getenv("QDRANT_LOCAL_PATH")
        self.local_path = path or env_local_path or str(Path(__file__).parent.parent / "qdrant_storage")
        self._local_requested = bool(path or env_local_path or os.getenv("QDRANT_USE_LOCAL", "").lower() in {"1", "true", "yes"})
        self._client: Optional[QdrantClient] = None
        self._connected = False
        self._collection_name = collection_name
        self._bm25_states: Dict[str, Dict[str, Any]] = {}

    def connect(self) -> bool:
        """Connect to Qdrant server (remote or local)."""
        if not QDRANT_AVAILABLE:
            print("Qdrant not available, using mock mode")
            return False

        def _try_local():
            try:
                Path(self.local_path).mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=self.local_path, timeout=5)
                client.get_collections()
                return client
            except Exception:
                return None

        def _try_remote():
            try:
                client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    prefer_grpc=self.prefer_grpc,
                    timeout=5
                )
                client.get_collections()
                return client
            except Exception:
                return None

        if self._local_requested:
            local_client = _try_local()
            if local_client is not None:
                self._client = local_client
                self._connected = True
                print(f"Connected to Qdrant (local) at {self.local_path}")
                return True

        remote_client = _try_remote()
        if remote_client is not None:
            self._client = remote_client
            self._connected = True
            print(f"Connected to Qdrant at {self.host}:{self.port}")
            return True

        if not self._local_requested:
            local_client = _try_local()
            if local_client is not None:
                self._client = local_client
                self._connected = True
                print(f"Connected to Qdrant (local fallback) at {self.local_path}")
                return True

        print("Qdrant connection failed")
        return False

    def disconnect(self):
        """Disconnect from Qdrant."""
        if self._client:
            self._client.close()
            self._connected = False

    @property
    def client(self) -> Optional[QdrantClient]:
        return self._client

    def _resolve_collection_name(self, collection_name: Optional[str] = None) -> str:
        return collection_name or self._collection_name

    def _get_bm25_index_file(self, collection_name: str) -> Path:
        if collection_name == self._collection_name:
            return Path(self.local_path) / "bm25_index.json"
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", collection_name)
        return Path(self.local_path) / f"bm25_index_{safe_name}.json"

    def _get_bm25_state(self, collection_name: str) -> Dict[str, Any]:
        if collection_name not in self._bm25_states:
            self._bm25_states[collection_name] = {
                "term_freq": {},
                "idf": {},
                "doc_lengths": {},
                "total_docs": 0,
                "avg_doc_len": 0.0,
                "loaded": False,
            }
        return self._bm25_states[collection_name]

    def _collection_point_count(self, collection_name: str) -> Optional[int]:
        """Return Qdrant's current point count for cache validation."""
        try:
            info = self._client.get_collection(collection_name)
            return int(info.points_count or 0)
        except Exception:
            return None

    def _load_bm25_cache(
        self,
        index_file: Path,
        collection_name: str,
        expected_points: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Load a BM25 cache only when it matches the current collection/index format."""
        if not index_file.exists():
            return None

        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Could not load BM25 cache: {e}")
            return None

        meta = data.get("_meta") or {}
        if meta.get("bm25_index_version") != BM25_INDEX_VERSION:
            print("Ignoring stale BM25 cache: index version mismatch")
            return None
        if meta.get("collection_name") != collection_name:
            print("Ignoring stale BM25 cache: collection mismatch")
            return None
        if not data.get("doc_lengths"):
            print("Ignoring stale BM25 cache: missing document lengths")
            return None
        if expected_points is not None and int(data.get("total_docs", 0) or 0) != expected_points:
            print("Ignoring stale BM25 cache: collection point count changed")
            return None

        return data

    def _matches_verse_filter(
        self,
        verse_id: Optional[str],
        verse_filter: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check whether a verse id satisfies a verse filter payload."""
        if not verse_filter:
            return True

        verse_ids = verse_filter.get("verse_ids") or []
        if verse_ids:
            return bool(verse_id and verse_id in set(verse_ids))

        chapter = verse_filter.get("chapter")
        if not chapter or not verse_id:
            return True

        prefix = f"BhG {chapter}."
        if not verse_id.startswith(prefix):
            return False

        try:
            verse_num = int(verse_id.split(".", 1)[1])
        except (IndexError, ValueError):
            return False

        verse_start = int(verse_filter.get("verse_start", 1) or 1)
        verse_end = int(verse_filter.get("verse_end", verse_start) or verse_start)
        return verse_start <= verse_num <= verse_end

    def _normalize_point_id(self, point_id: Any) -> str:
        """Return a stable Qdrant-safe UUID string for arbitrary chunk IDs."""
        text = str(point_id)
        try:
            uuid.UUID(text)
            return text
        except (ValueError, TypeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sansrag:{text}"))
    
    def create_collection(
        self,
        collection_name: str,
        dense_dim: int = DENSE_DIM,
        colbert_dim: int = COLBERT_DIM,
        drop_if_exists: bool = True
    ) -> bool:
        """Create a collection with dense and sparse vector support."""
        if not QDRANT_AVAILABLE or not self._connected:
            return False
        
        if self.collection_exists(collection_name):
            if not drop_if_exists:
                return True
            self._client.delete_collection(collection_name)
        
        hnsw_config = HnswConfigDiff(
            m=HNSW_M,
            ef_construct=HNSW_EF_CONSTRUCTION
        )
        
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dense_dim,
                distance=Distance.COSINE
            ),
            sparse_vectors_config={
                "text_sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False
                    )
                )
            },
            hnsw_config=hnsw_config,
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2
            )
        )
        
        self._client.create_payload_index(
            collection_name=collection_name,
            field_name="dataset_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        self._client.create_payload_index(
            collection_name=collection_name,
            field_name="verse_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        print(f"Created collection '{collection_name}' with dense and sparse indices")
        return True
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        if not QDRANT_AVAILABLE or not self._connected:
            return False
        collections = self._client.get_collections().collections
        return any(c.name == collection_name for c in collections)
    
    def insert_embeddings(
        self,
        collection_name: str,
        embeddings: List[Any],
        batch_size: int = 256,
        show_progress: bool = True
    ) -> int:
        """Insert embeddings into collection."""
        if not QDRANT_AVAILABLE or not self._connected:
            return 0
        
        if not self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' not found")
            return 0
        
        total_inserted = 0
        
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            
            points = []
            for e in batch:
                dense_vec = e.dense_vector.tolist() if hasattr(e.dense_vector, 'tolist') else list(e.dense_vector)
                
                sparse_dict = e.sparse_vector
                sparse_vec = SparseVector(
                    indices=list(sparse_dict.keys()),
                    values=list(sparse_dict.values())
                )
                
                colbert_arr = np.array(e.colbert_vectors)
                colbert_mean = colbert_arr.mean(axis=0).tolist() if len(colbert_arr.shape) > 1 else list(e.colbert_vectors)
                
                original_id = str(e.id)
                payload = {
                    "text": e.text[:MAX_TEXT_LENGTH - 1] if len(e.text) >= MAX_TEXT_LENGTH else e.text,
                    "dataset_type": e.metadata.get("dataset_type", "unknown"),
                    "verse_id": e.metadata.get("verse_id", "")[:99],
                    "line_number": e.metadata.get("line_number", 0),
                    "metadata": {**e.metadata, "original_id": original_id},
                    "colbert_vector": colbert_mean
                }
                
                points.append(PointStruct(
                    id=self._normalize_point_id(original_id),
                    vector={
                        "": dense_vec,
                        "text_sparse": sparse_vec
                    },
                    payload=payload
                ))
            
            self._client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            total_inserted += len(batch)
            
            if show_progress and total_inserted % 500 == 0:
                print(f"  Inserted {total_inserted}/{len(embeddings)} embeddings")
        
        print(f"Total inserted: {total_inserted}")
        return total_inserted
    
    def load_collection(self, collection_name: str):
        """Load collection into memory for search."""
        if not QDRANT_AVAILABLE or not self._connected:
            return
        if self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' ready for search")
    
    # ============================================================
    # DENSE VECTOR SEARCH (Semantic Relevance)
    # ============================================================
    
    def search_dense(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        ef_search: int = 128,
        verse_filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search using dense vectors for semantic relevance."""
        if not QDRANT_AVAILABLE or not self._connected:
            return []
        
        coll_name = self._resolve_collection_name(collection_name)
        if not self.collection_exists(coll_name):
            return []
        
        query_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
        
        filter_obj = None
        if verse_filter:
            filter_obj = self._build_verse_filter(verse_filter)
        
        results = self._client.query_points(
            collection_name=coll_name,
            query=query_list,
            query_filter=filter_obj,
            limit=top_k,
            with_payload=True
        ).points
        
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(SearchResult(
                id=hit.id,
                text=payload.get("text", ""),
                score=hit.score,
                dataset_type=payload.get("dataset_type", ""),
                verse_id=payload.get("verse_id", ""),
                metadata=payload.get("metadata", {})
            ))
        
        return search_results
    
    # ============================================================
    # SPARSE VECTOR SEARCH (Lexical Matching)
    # ============================================================
    
    def search_sparse(
        self,
        query_sparse: Dict[int, float],
        top_k: int = 10,
        verse_filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search using sparse vectors for lexical matching."""
        if not QDRANT_AVAILABLE or not self._connected:
            return []
        
        coll_name = self._resolve_collection_name(collection_name)
        if not self.collection_exists(coll_name):
            return []
        
        sparse_vec = SparseVector(
            indices=list(query_sparse.keys()),
            values=list(query_sparse.values())
        )
        
        filter_obj = None
        if verse_filter:
            filter_obj = self._build_verse_filter(verse_filter)
        
        results = self._client.query_points(
            collection_name=coll_name,
            query=sparse_vec,
            using="text_sparse",
            query_filter=filter_obj,
            limit=top_k,
            with_payload=True
        ).points
        
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(SearchResult(
                id=hit.id,
                text=payload.get("text", ""),
                score=hit.score,
                dataset_type=payload.get("dataset_type", ""),
                verse_id=payload.get("verse_id", ""),
                metadata=payload.get("metadata", {})
            ))
        
        return search_results

    def search_by_verse_ids(
        self,
        verse_ids: List[str],
        top_k: int = 10,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """Fetch exact verse-id matches from Qdrant for explicit verse queries."""
        if not QDRANT_AVAILABLE or not self._connected or not verse_ids:
            return []

        coll_name = self._resolve_collection_name(collection_name)
        if not self.collection_exists(coll_name):
            return []

        verse_filter = Filter(
            must=[FieldCondition(key="verse_id", match=MatchAny(any=list(dict.fromkeys(verse_ids))))]
        )

        points, _ = self._client.scroll(
            collection_name=coll_name,
            scroll_filter=verse_filter,
            limit=max(top_k * 4, len(verse_ids) * 4),
            with_payload=True,
        )

        ordered_points = sorted(
            points,
            key=lambda point: (
                ((point.payload or {}).get("verse_id") or ""),
                int(((point.payload or {}).get("metadata") or {}).get("line_number", 0) or 0),
                str(point.id),
            ),
        )

        results = []
        for rank, point in enumerate(ordered_points[:top_k], 1):
            payload = point.payload or {}
            results.append(SearchResult(
                id=str(point.id),
                text=payload.get("text", ""),
                score=max(0.0, 1.0 - (rank - 1) * 0.01),
                dataset_type=payload.get("dataset_type", ""),
                verse_id=payload.get("verse_id", ""),
                metadata={
                    **payload.get("metadata", {}),
                    "retrieval_mode": "verse_filter_exact",
                }
            ))

        return results
    
    # ============================================================
    # COLBERT SEARCH
    # ============================================================
    
    def search_colbert(
        self,
        query_colbert: np.ndarray,
        top_k: int = 10,
        nprobe: int = 16,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search using Colbert vectors (mean-pooled approximation)."""
        if not QDRANT_AVAILABLE or not self._connected:
            return []
        
        coll_name = self._resolve_collection_name(collection_name)
        if not self.collection_exists(coll_name):
            return []
        
        query_mean = query_colbert.mean(axis=0) if len(query_colbert.shape) > 1 else query_colbert
        
        results = self._client.query_points(
            collection_name=coll_name,
            query=query_mean.tolist(),
            limit=top_k,
            with_payload=True
        ).points
        
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(SearchResult(
                id=hit.id,
                text=payload.get("text", ""),
                score=hit.score,
                dataset_type=payload.get("dataset_type", ""),
                verse_id=payload.get("verse_id", ""),
                metadata=payload.get("metadata", {})
            ))
        
        return search_results
    
    # ============================================================
    # REAL BM25 SEARCH (Term Frequency - Inverse Document Frequency)
    # ============================================================
    
    def _build_bm25_index(
        self,
        top_n: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        """Build BM25 index by sampling documents from Qdrant."""
        if not QDRANT_AVAILABLE or not self._connected:
            return

        coll_name = self._resolve_collection_name(collection_name)
        if not self.collection_exists(coll_name):
            return

        state = self._get_bm25_state(coll_name)
        if state["loaded"]:
            return

        expected_points = self._collection_point_count(coll_name)
        index_file = self._get_bm25_index_file(coll_name)
        data = self._load_bm25_cache(index_file, coll_name, expected_points)
        if data:
            state["term_freq"] = data.get("term_freq", {})
            state["idf"] = data.get("idf", {})
            state["doc_lengths"] = data.get("doc_lengths", {})
            state["total_docs"] = data.get("total_docs", 0)
            state["avg_doc_len"] = data.get("avg_doc_len", 0.0)
            state["loaded"] = True
            print(f"Loaded BM25 index from cache ({state['total_docs']} docs)")
            return

        try:
            all_points = []
            next_offset = None
            while True:
                batch_limit = 1000
                if top_n is not None:
                    batch_limit = max(1, min(batch_limit, top_n - len(all_points)))
                points, next_offset = self._client.scroll(
                    collection_name=coll_name,
                    limit=batch_limit,
                    offset=next_offset,
                    with_payload=True
                )
                all_points.extend(points)
                if not next_offset or (top_n is not None and len(all_points) >= top_n):
                    break

            term_freq: Dict[str, Dict[str, int]] = {}
            doc_lengths: Dict[str, int] = {}

            for point in all_points:
                payload = point.payload or {}
                text = payload.get("text", "").lower()
                doc_id = str(point.id)
                tokens = re.findall(r'\w+', text)
                doc_lengths[doc_id] = len(tokens)

                for token in tokens:
                    if token not in term_freq:
                        term_freq[token] = {}
                    term_freq[token][doc_id] = term_freq[token].get(doc_id, 0) + 1

            total_docs = len(doc_lengths)
            idf = {}
            for term, doc_counts in term_freq.items():
                idf[term] = math.log((total_docs + 1) / (len(doc_counts) + 0.5))

            state["term_freq"] = term_freq
            state["idf"] = idf
            state["doc_lengths"] = doc_lengths
            state["total_docs"] = total_docs
            state["avg_doc_len"] = sum(doc_lengths.values()) / max(1, total_docs)
            state["loaded"] = True

            try:
                index_file.parent.mkdir(parents=True, exist_ok=True)
                with open(index_file, 'w') as f:
                    json.dump({
                        "_meta": {
                            "bm25_index_version": BM25_INDEX_VERSION,
                            "collection_name": coll_name,
                            "source_point_count": expected_points,
                            "built_at": time.time(),
                        },
                        "term_freq": term_freq,
                        "idf": idf,
                        "doc_lengths": doc_lengths,
                        "total_docs": total_docs,
                        "avg_doc_len": state["avg_doc_len"]
                    }, f)
                print(f"Built and cached BM25 index ({total_docs} docs)")
            except Exception as e:
                print(f"Could not save BM25 cache: {e}")
        except Exception as e:
            print(f"Warning: Could not build BM25 index: {e}")
    
    def bm25_search(
        self,
        query_terms: List[str],
        top_k: int = 10,
        verse_filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform real BM25 search with TF-IDF weighting."""
        if not QDRANT_AVAILABLE or not self._connected:
            return []
        
        coll_name = self._resolve_collection_name(collection_name)
        state = self._get_bm25_state(coll_name)

        if not state["term_freq"]:
            self._build_bm25_index(collection_name=coll_name)
            state = self._get_bm25_state(coll_name)
        
        if not state["term_freq"]:
            return self._bm25_fallback(query_terms, top_k, verse_filter, collection_name=coll_name)
        
        k1 = 1.5
        b = 0.75
        avg_doc_len = state["avg_doc_len"]
        
        doc_scores: Dict[str, float] = {}
        
        for term in query_terms:
            term_lower = term.lower()
            
            if term_lower not in state["term_freq"]:
                continue
            
            idf = state["idf"].get(term_lower, 0.0)
            if idf <= 0:
                continue
            
            for doc_id, tf in state["term_freq"][term_lower].items():
                doc_len = int((state.get("doc_lengths") or {}).get(doc_id, 100) or 100)
                bm25_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1)))
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + bm25_score
        
        candidate_limit = top_k * (6 if verse_filter else 2)
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_limit]
        
        if not sorted_docs:
            return self._bm25_fallback(query_terms, top_k, verse_filter, collection_name=coll_name)
        
        doc_ids = [d[0] for d in sorted_docs]
        results = []
        
        try:
            points = self._client.retrieve(
                collection_name=coll_name,
                ids=doc_ids,
                with_payload=True
            )
            
            score_map = dict(sorted_docs)
            for point in points:
                payload = point.payload or {}
                verse_id = payload.get("verse_id", "")
                if verse_filter and not self._matches_verse_filter(verse_id, verse_filter):
                    continue
                results.append(SearchResult(
                    id=str(point.id),
                    text=payload.get("text", ""),
                    score=score_map.get(str(point.id), 0.0),
                    dataset_type=payload.get("dataset_type", ""),
                    verse_id=verse_id,
                    metadata={
                        **payload.get("metadata", {}),
                        "retrieval_mode": "bm25"
                    }
                ))
            
            results.sort(key=lambda x: x.score, reverse=True)
            if verse_filter and not results:
                return self._bm25_fallback(query_terms, top_k, verse_filter, collection_name=coll_name)
            return results[:top_k]
        except Exception:
            return self._bm25_fallback(query_terms, top_k, verse_filter, collection_name=coll_name)
    
    def _bm25_fallback(
        self,
        query_terms: List[str],
        top_k: int = 10,
        verse_filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """Fallback BM25 using sparse vector search."""
        query_sparse = {}
        for term in query_terms:
            term_hash = int(hashlib.md5(f"42:{term.lower()}".encode()).hexdigest(), 16) % (2**31)
            query_sparse[term_hash] = query_sparse.get(term_hash, 0) + 1
        
        if query_sparse:
            max_freq = max(query_sparse.values())
            query_sparse = {k: v / max_freq for k, v in query_sparse.items()}
        
        return self.search_sparse(
            query_sparse,
            top_k,
            verse_filter,
            collection_name=collection_name,
        )
    
    # ============================================================
    # VERSE-LEVEL FILTERING
    # ============================================================
    
    def _build_verse_filter(self, verse_filter: Dict[str, Any]) -> Optional[Filter]:
        """Build a Qdrant Filter from verse filter parameters."""
        if not verse_filter:
            return None
        
        conditions = []
        
        verse_ids = verse_filter.get("verse_ids")
        if verse_ids:
            conditions.append(
                FieldCondition(key="verse_id", match=MatchAny(any=verse_ids))
            )
        
        chapter = verse_filter.get("chapter")
        if chapter:
            verse_start = verse_filter.get("verse_start", 1)
            verse_end = verse_filter.get("verse_end", 999)
            
            all_verse_ids = [f"BhG {chapter}.{v}" for v in range(verse_start, verse_end + 1)]
            conditions.append(
                FieldCondition(key="verse_id", match=MatchAny(any=all_verse_ids))
            )
        
        if not conditions:
            return None
        
        return Filter(must=conditions)
    
    def search_with_verse_filter(
        self,
        query_vector: np.ndarray,
        verse_filter: Dict[str, Any],
        top_k: int = 10,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search dense vectors with verse-level filtering applied."""
        return self.search_dense(
            query_vector,
            top_k,
            verse_filter=verse_filter,
            collection_name=collection_name,
        )
    
    # ============================================================
    # STATS
    # ============================================================
    
    def get_collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """Get statistics about a collection."""
        if not QDRANT_AVAILABLE or not self._connected:
            return {}
        
        coll_name = collection_name or self._collection_name
        if not self.collection_exists(coll_name):
            return {}
        
        info = self._client.get_collection(coll_name)
        return {
            "name": coll_name,
            "row_count": info.points_count or 0,
            "index_info": str(info.config.params)
        }


def create_all_collections(manager: QdrantManager) -> Dict[str, str]:
    """Create all dataset collections."""
    collections = {}
    for dtype, coll_name in COLLECTION_NAMES.items():
        if manager.create_collection(coll_name):
            collections[dtype] = coll_name
    return collections


if __name__ == "__main__":
    manager = QdrantManager()
    
    if manager.connect():
        collections = create_all_collections(manager)
        
        for dtype, coll_name in collections.items():
            print(f"\n{dtype}: {coll_name}")
            stats = manager.get_collection_stats(coll_name)
            print(f"  Row count: {stats.get('row_count', 0)}")
        
        manager.disconnect()
