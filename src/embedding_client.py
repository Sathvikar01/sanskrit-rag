"""NVIDIA BGE-M3 Embedding Client with dense, sparse, and multi-vec support."""
import re
import time
import json
import hashlib
import tiktoken
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    NVIDEA_API_KEY,
    NVIDIA_API_URL,
    EMBEDDING_MODEL,
    BATCH_SIZE,
    MAX_RETRIES,
    DENSE_DIM,
    COLBERT_DIM
)
from src.evidence_cache import EvidenceCache

MAX_TOKENS = 512

_DETERMINISTIC_HASH_SEED = 42
_WORD_TO_SPARSE_IDX: Dict[str, int] = {}


def _get_deterministic_sparse_idx(token: str) -> int:
    """Get a deterministic sparse index for a token."""
    global _WORD_TO_SPARSE_IDX
    if token not in _WORD_TO_SPARSE_IDX:
        hash_val = int(hashlib.md5(f"{_DETERMINISTIC_HASH_SEED}:{token}".encode()).hexdigest(), 16)
        _WORD_TO_SPARSE_IDX[token] = hash_val % (2**31)
    return _WORD_TO_SPARSE_IDX[token]


@dataclass
class EmbeddingResult:
    """Container for all embedding types from BGE-M3."""
    id: str
    text: str
    dense_vector: np.ndarray
    sparse_vector: Dict[int, float]
    colbert_vectors: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "dense_vector": self.dense_vector.tolist(),
            "sparse_vector": self.sparse_vector,
            "colbert_vectors": self.colbert_vectors.tolist(),
            "metadata": self.metadata
        }


class NVIDIAEmbeddingClient:
    """Client for NVIDIA NIM BGE-M3 embedding API."""
    
    def __init__(
        self,
        api_key: str = None,
        api_url: str = NVIDIA_API_URL,
        model: str = EMBEDDING_MODEL,
        batch_size: int = BATCH_SIZE,
        max_retries: int = MAX_RETRIES,
        cache_dir: Optional[str] = None
    ):
        self.api_key = api_key or NVIDEA_API_KEY
        self.api_url = api_url
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.session = self._create_session()
        self._embedding_cache: Dict[str, EmbeddingResult] = {}
        self.evidence_cache = EvidenceCache()
        
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self._tokenizer = None
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _get_cache_key(self, text: str, input_type: str = "passage") -> str:
        key = {
            "model": self.model,
            "input_type": input_type,
            "dense_dim": DENSE_DIM,
            "text": text,
        }
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[EmbeddingResult]:
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        cached_json = self.evidence_cache.get("embedding", cache_key)
        if cached_json:
            result = self._embedding_from_dict(cached_json)
            if len(result.dense_vector) == DENSE_DIM:
                self._embedding_cache[cache_key] = result
                return result

        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            data = np.load(str(cache_file), allow_pickle=True).item()
            result = self._embedding_from_dict(data)
            if len(result.dense_vector) == DENSE_DIM:
                self._embedding_cache[cache_key] = result
                return result
        return None
    
    def _save_to_cache(self, result: EmbeddingResult, cache_key: str):
        self._embedding_cache[cache_key] = result
        self.evidence_cache.set("embedding", cache_key, result.to_dict())
        cache_file = self.cache_dir / f"{cache_key}.npy"
        np.save(str(cache_file), result.to_dict())

    def _embedding_from_dict(self, data: Dict[str, Any]) -> EmbeddingResult:
        sparse_vector = {
            int(key): float(value)
            for key, value in (data.get("sparse_vector") or {}).items()
        }
        return EmbeddingResult(
            id=data.get("id", ""),
            text=data.get("text", ""),
            dense_vector=np.array(data.get("dense_vector", []), dtype=np.float32),
            sparse_vector=sparse_vector,
            colbert_vectors=np.array(data.get("colbert_vectors", []), dtype=np.float32),
            metadata=data.get("metadata", {}) or {},
        )
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        ids: List[str] = None,
        metadata_list: List[Dict] = None,
        input_type: str = "passage",
    ) -> List[EmbeddingResult]:
        """Get dense, sparse, and multi-vec embeddings for a batch of texts."""
        if ids is None:
            ids = [self._get_cache_key(t, input_type=input_type)[:16] for t in texts]
        if metadata_list is None:
            metadata_list = [{} for _ in texts]
        
        results: List[Optional[EmbeddingResult]] = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for idx, (text, text_id) in enumerate(zip(texts, ids)):
            cache_key = self._get_cache_key(text, input_type=input_type)
            cached = self._load_from_cache(cache_key)
            if cached:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            batch_results = self._call_api_batch(uncached_texts, input_type=input_type)
            for i, (idx, emb_result) in enumerate(zip(uncached_indices, batch_results)):
                emb_result.id = ids[idx]
                emb_result.text = uncached_texts[i]
                emb_result.metadata = {
                    **(metadata_list[idx] or {}),
                    "embedding_model": self.model,
                    "embedding_input_type": input_type,
                    "dense_dim": DENSE_DIM,
                }
                cache_key = self._get_cache_key(uncached_texts[i], input_type=input_type)
                self._save_to_cache(emb_result, cache_key)
                results[idx] = emb_result

        return [r for r in results if r is not None]
    
    def _call_api_batch(self, texts: List[str], input_type: str = "passage") -> List[EmbeddingResult]:
        """Call NVIDIA API for embeddings."""
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            dense_embeddings = self._get_dense_embeddings(batch, input_type=input_type)
            sparse_embeddings = self._get_sparse_embeddings(batch)
            colbert_embeddings = self._get_colbert_embeddings(batch)
            
            for j, (dense, sparse, colbert) in enumerate(
                zip(dense_embeddings, sparse_embeddings, colbert_embeddings)
            ):
                results.append(EmbeddingResult(
                    id="",
                    text=batch[j],
                    dense_vector=np.array(dense, dtype=np.float32),
                    sparse_vector=sparse,
                    colbert_vectors=np.array(colbert, dtype=np.float32),
                    metadata={}
                ))
            
            time.sleep(0.1)
        
        return results
    
    def _get_dense_embeddings(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        """Get dense embeddings from NVIDIA API with retry logic and token truncation."""
        cleaned_texts = []
        for t in texts:
            t = re.sub(r'_Case=[^_]+', '', t)
            t = re.sub(r'--', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            
            if not t:
                cleaned_texts.append("unknown")
                continue
            
            if self._tokenizer:
                tokens = self._tokenizer.encode(t)
                if len(tokens) > MAX_TOKENS - 10:
                    tokens = tokens[:MAX_TOKENS - 10]
                    t = self._tokenizer.decode(tokens)
            
            words = t.split()
            max_words = 350
            if len(words) > max_words:
                t = ' '.join(words[:max_words])
            
            if not t.strip():
                t = "unknown"
            cleaned_texts.append(t)

        if not cleaned_texts:
            return []

        payload = {
            "model": self.model,
            "input": cleaned_texts,
            "input_type": input_type,
            "encoding_format": "float"
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    error_detail = ""
                    try:
                        error_detail = e.response.json().get("error", str(e.response.text[:200]))
                    except:
                        error_detail = e.response.text[:200] if e.response.text else "Unknown error"
                    print(f"API 400 Error (attempt {attempt+1}/{self.max_retries}): {error_detail}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise
                elif e.response.status_code in [429, 500, 502, 503, 504]:
                    print(f"API {e.response.status_code} Error (attempt {attempt+1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise
                else:
                    raise
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        return []
    
    def _get_sparse_embeddings(self, texts: List[str]) -> List[Dict[int, float]]:
        """Generate sparse embeddings (BM25-style lexical features) with deterministic hashing."""
        sparse_vectors = []

        for text in texts:
            tokens = text.lower().split()
            token_freq = {}
            for token in tokens:
                token_hash = _get_deterministic_sparse_idx(token)
                token_freq[token_hash] = token_freq.get(token_hash, 0) + 1

            max_freq = max(token_freq.values()) if token_freq else 1
            sparse_vec = {k: v / max_freq * np.log(1 + len(tokens) / (1 + v))
                          for k, v in token_freq.items()}
            sparse_vectors.append(sparse_vec)

        return sparse_vectors

    def _get_colbert_embeddings(self, texts: List[str]) -> List[List[List[float]]]:
        """Get Colbert-style multi-vector embeddings using dense embedding per token."""
        colbert_results = []

        for text in texts:
            words = text.split()[:128]
            if not words:
                colbert_results.append([[0.0] * COLBERT_DIM] * 128)
                continue

            word_embeddings = []
            for word in words:
                word_emb = self._get_single_word_embedding(word)
                word_embeddings.append(word_emb)

            while len(word_embeddings) < 128:
                word_embeddings.append([0.0] * COLBERT_DIM)

            colbert_results.append(word_embeddings[:128])

        return colbert_results

    def _get_single_word_embedding(self, word: str) -> List[float]:
        """Get a dense embedding for a single word, reduced to COLBERT_DIM."""
        try:
            payload = {
                "model": self.model,
                "input": [word],
                "input_type": "passage",
                "encoding_format": "float"
            }
            response = self.session.post(
                self.api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            dense_emb = data["data"][0]["embedding"]
            reduced = np.array(dense_emb).reshape(-1, 8).mean(axis=1).tolist()
            while len(reduced) < COLBERT_DIM:
                reduced.append(0.0)
            return reduced[:COLBERT_DIM]
        except Exception:
            np.random.seed(hash(word) % (2**31))
            emb = np.random.randn(COLBERT_DIM).astype(np.float32).tolist()
            return emb

    def _build_query_fallback_embedding(self, query: str, reason: str = "") -> EmbeddingResult:
        """Build a local-only query embedding when the remote dense API is unavailable.

        This is intentionally used only for live query embedding, not corpus ingestion.
        Sparse features still provide lexical retrieval, while dense-only paths are skipped.
        """
        sparse_vector = self._get_sparse_embeddings([query])[0]
        return EmbeddingResult(
            id=self._get_cache_key(query, input_type="query")[:16],
            text=query,
            dense_vector=np.zeros(DENSE_DIM, dtype=np.float32),
            sparse_vector=sparse_vector,
            colbert_vectors=np.zeros((128, COLBERT_DIM), dtype=np.float32),
            metadata={
                "dense_available": False,
                "embedding_fallback": "local_sparse_only",
                "embedding_input_type": "query",
                "embedding_model": self.model,
                "fallback_reason": reason[:300],
            },
        )
    
    def embed_query(self, query: str, allow_fallback: bool = True) -> EmbeddingResult:
        """Embed a single query for retrieval."""
        failure_reason = ""
        try:
            results = self.get_embeddings_batch([query], input_type="query")
            if results:
                result = results[0]
                result.metadata = {
                    **(result.metadata or {}),
                    "dense_available": True,
                }
                return result
            failure_reason = "embedding API returned no query embedding"
        except Exception as exc:
            failure_reason = str(exc)
            if not allow_fallback:
                raise
            print(f"Query embedding fallback activated: {exc}")

        if not allow_fallback:
            raise RuntimeError(failure_reason or "query embedding failed")

        return self._build_query_fallback_embedding(query, failure_reason)
    
    def embed_chunks(
        self,
        chunks: List[Any],
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """Embed a list of TextChunk objects."""
        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
        metadata = [c.to_dict() for c in chunks]
        
        results = []
        total = len(texts)
        
        for i in range(0, total, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]
            batch_meta = metadata[i:i + self.batch_size]
            
            batch_results = self.get_embeddings_batch(
                batch_texts, batch_ids, batch_meta, input_type="passage"
            )
            results.extend(batch_results)
            
            if show_progress:
                print(f"  Embedded {min(i + self.batch_size, total)}/{total} chunks")
        
        return results


def compute_l1_regularization(weights: np.ndarray) -> float:
    """Compute L1 regularization term."""
    return np.sum(np.abs(weights))


def compute_l2_regularization(weights: np.ndarray) -> float:
    """Compute L2 regularization term."""
    return np.sum(weights ** 2)


def apply_regularization(
    scores: np.ndarray,
    weights: np.ndarray,
    l1_lambda: float = 0.01,
    l2_lambda: float = 0.001
) -> np.ndarray:
    """Apply L1/L2 regularization to scores."""
    l1_penalty = l1_lambda * compute_l1_regularization(weights)
    l2_penalty = l2_lambda * compute_l2_regularization(weights)
    regularized_scores = scores - (l1_penalty + l2_penalty)
    return regularized_scores


if __name__ == "__main__":
    client = NVIDIAEmbeddingClient()
    
    test_texts = [
        "dharma-kṣetre kuru-kṣetre",
        "arjuna uvāca",
        "kṛṣṇaṃ praṇamya"
    ]
    
    results = client.get_embeddings_batch(test_texts)
    
    for r in results:
        print(f"\nID: {r.id}")
        print(f"Text: {r.text[:50]}...")
        print(f"Dense shape: {r.dense_vector.shape}")
        print(f"Sparse keys: {len(r.sparse_vector)}")
        print(f"Colbert shape: {len(r.colbert_vectors)}x{len(r.colbert_vectors[0])}")
