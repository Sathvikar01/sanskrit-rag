"""Vector store using FAISS with BGE-M3-SanskritFT embeddings."""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.preprocessing.chunker import Chunk
from src.utils.config import Config
from src.utils.logger import logger


class VectorStore:
    """FAISS-based vector store for Sanskrit text retrieval."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        self.model_name = config.embedding_model
        self.device = config.embedding_device
        self.batch_size = config.get("embedding.batch_size", 32)
        self.normalize = config.get("embedding.normalize", True)

        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_ids: list[str] = []
        self.dimension: int = 1024

    def load_model(self):
        """Load the embedding model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Model loaded on device: {self.device}")

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of texts to encode.

        Returns:
            Numpy array of shape (len(texts), dimension).
        """
        if self.model is None:
            self.load_model()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        return np.array(embeddings, dtype=np.float32)

    def build_index(self, chunks: list[Chunk], use_devanagari: bool = True):
        """Build FAISS index from chunks.

        Args:
            chunks: List of Chunk objects to index.
            use_devanagari: Whether to use Devanagari text (True) or IAST (False).
        """
        logger.info(f"Building FAISS index from {len(chunks)} chunks")

        if use_devanagari:
            texts = [c.text_devanagari for c in chunks]
        else:
            texts = [c.text_iast for c in chunks]

        self.chunk_ids = [c.chunk_id for c in chunks]

        embeddings = self.encode_texts(texts)
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        logger.info(
            f"FAISS index built: {self.index.ntotal} vectors, "
            f"dimension={self.dimension}"
        )

    def search(
        self,
        query: str,
        top_k: int = 50,
        use_devanagari: bool = True,
    ) -> list[dict]:
        """Search for similar chunks.

        Args:
            query: Query text to search for.
            top_k: Number of results to return.
            use_devanagari: Whether query is in Devanagari (True) or IAST (False).

        Returns:
            List of dicts with chunk_id, score, and rank.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        if self.model is None:
            self.load_model()

        query_embedding = self.encode_texts([query])
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx < 0:
                continue
            results.append(
                {
                    "chunk_id": self.chunk_ids[idx],
                    "score": float(score),
                    "rank": rank,
                }
            )

        return results

    def save(self, index_path: str | Path, metadata_path: str | Path):
        """Save FAISS index and metadata to disk.

        Args:
            index_path: Path to save the FAISS index.
            metadata_path: Path to save chunk ID metadata.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved FAISS index to {index_path}")

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunk_ids": self.chunk_ids,
                    "dimension": self.dimension,
                    "model_name": self.model_name,
                    "normalize": self.normalize,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Saved metadata to {metadata_path}")

    def load(self, index_path: str | Path, metadata_path: str | Path):
        """Load FAISS index and metadata from disk.

        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the metadata file.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.chunk_ids = metadata["chunk_ids"]
        self.dimension = metadata["dimension"]
        logger.info(f"Loaded metadata: {len(self.chunk_ids)} chunk IDs")


class VectorStoreSingleton:
    """Singleton wrapper for VectorStore."""

    _instance: Optional[VectorStore] = None

    @classmethod
    def get_instance(cls, config: Config = None) -> VectorStore:
        if cls._instance is None:
            cls._instance = VectorStore(config)
        return cls._instance
