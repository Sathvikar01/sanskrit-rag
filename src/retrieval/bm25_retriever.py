"""BM25 retriever for Sanskrit text using lemma-normalized content."""

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.preprocessing.chunker import Chunk
from src.utils.logger import logger


class BM25Retriever:
    """BM25-based retriever for Sanskrit text."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25.

        Simple whitespace + punctuation tokenization.
        """
        text = text.lower()
        text = text.replace("।", " ").replace("||", " ")
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]

    def _get_lemma_tokens(self, chunk: Chunk) -> list[str]:
        """Get lemma-based tokens from a chunk.

        Uses the lemmatized forms for better matching.
        """
        if chunk.lemmas:
            return [l.lower() for l in chunk.lemmas if l]
        return self._tokenize(chunk.text_iast)

    def build_index(self, chunks: list[Chunk], use_lemmas: bool = True):
        """Build BM25 index from chunks.

        Args:
            chunks: List of Chunk objects to index.
            use_lemmas: Whether to use lemma tokens (True) or raw text (False).
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks")

        self.chunk_ids = [c.chunk_id for c in chunks]

        if use_lemmas:
            self.tokenized_corpus = [self._get_lemma_tokens(c) for c in chunks]
        else:
            self.tokenized_corpus = [self._tokenize(c.text_iast) for c in chunks]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

        logger.info(f"BM25 index built with {len(chunks)} documents")

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        """Search for relevant chunks using BM25.

        Args:
            query: Query text to search for.
            top_k: Number of results to return.

        Returns:
            List of dicts with chunk_id, score, and rank.
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index first.")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.warning(f"No tokens extracted from query: {query}")
            return []

        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] <= 0:
                continue
            results.append(
                {
                    "chunk_id": self.chunk_ids[idx],
                    "score": float(scores[idx]),
                    "rank": rank,
                }
            )

        return results

    def search_with_lemma_expansion(
        self,
        query: str,
        expanded_lemmas: list[str],
        top_k: int = 50,
    ) -> list[dict]:
        """Search with expanded lemma query.

        Args:
            query: Original query text.
            expanded_lemmas: Additional lemmas to expand the query.
            top_k: Number of results to return.

        Returns:
            List of dicts with chunk_id, score, and rank.
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index first.")

        query_tokens = self._tokenize(query)
        expanded_tokens = [l.lower() for l in expanded_lemmas]
        all_tokens = list(set(query_tokens + expanded_tokens))

        if not all_tokens:
            return []

        scores = self.bm25.get_scores(all_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] <= 0:
                continue
            results.append(
                {
                    "chunk_id": self.chunk_ids[idx],
                    "score": float(scores[idx]),
                    "rank": rank,
                }
            )

        return results
