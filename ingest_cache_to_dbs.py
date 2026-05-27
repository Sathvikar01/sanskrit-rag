"""Ingest cached embeddings into Qdrant (seg_lemma) and Neo4j (lemma_morph)."""
import sys
import os
import uuid
import time
from pathlib import Path
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.qdrant_manager import QdrantManager
from src.neo4j_manager import Neo4jManager
from src.embedding_client import EmbeddingResult
from config.settings import COLLECTION_NAMES


def hash_to_uuid(hex_str: str) -> str:
    """Convert a hex hash string to a valid UUID."""
    hex_str = hex_str[:32]
    return str(uuid.UUID(hex_str))


def load_cache_embeddings(cache_dir: Path) -> list:
    """Load all cached embeddings from .npy files."""
    embeddings = []
    files = sorted(cache_dir.glob("*.npy"))
    total = len(files)
    
    for i, f in enumerate(files):
        data = np.load(f, allow_pickle=True).item()
        dense = np.array(data["dense_vector"], dtype=np.float32)
        sparse = data["sparse_vector"]
        colbert = np.array(data["colbert_vectors"], dtype=np.float32)
        
        qdrant_id = hash_to_uuid(f.stem)
        
        emb = EmbeddingResult(
            id=qdrant_id,
            text=data["text"],
            dense_vector=dense,
            sparse_vector=sparse,
            colbert_vectors=colbert,
            metadata={**data.get("metadata", {}), "original_id": f.stem}
        )
        embeddings.append(emb)
        
        if (i + 1) % 5000 == 0:
            print(f"  Loaded {i+1}/{total}...")
    
    return embeddings


def detect_dataset_type(text: str) -> str:
    """Detect which dataset a chunk belongs to based on content."""
    text_lower = text.lower()
    if "case=" in text_lower or "gender=" in text_lower or "number=" in text_lower:
        return "seg_lemma"
    return "raw"


def main():
    cache_dir = Path(__file__).parent / "cache"
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    print(f"Loading cached embeddings from {cache_dir}...")
    embeddings = load_cache_embeddings(cache_dir)
    print(f"Loaded {len(embeddings)} embeddings from cache")
    
    if not embeddings:
        print("No embeddings found in cache")
        return
    
    # Split: seg_lemma -> Qdrant, lemma_morph -> Neo4j
    seg_lemma_embs = []
    lemma_morph_embs = []
    for emb in embeddings:
        dtype = emb.metadata.get("dataset_type", detect_dataset_type(emb.text))
        if dtype == "seg_lemma":
            seg_lemma_embs.append(emb)
        elif dtype == "lemma_morph":
            lemma_morph_embs.append(emb)
    
    print(f"\nDistribution:")
    print(f"  seg_lemma (-> Qdrant): {len(seg_lemma_embs)}")
    print(f"  lemma_morph (-> Neo4j): {len(lemma_morph_embs)}")
    print(f"  raw (-> cache only): {len(embeddings) - len(seg_lemma_embs) - len(lemma_morph_embs)}")
    
    # === Qdrant (seg_lemma) ===
    qdrant = QdrantManager()
    if qdrant.connect():
        print(f"\n--- Qdrant (seg_lemma) ---")
        coll_name = COLLECTION_NAMES["seg_lemma"]
        qdrant.create_collection(coll_name)
        
        if seg_lemma_embs:
            start = time.time()
            count = qdrant.insert_embeddings(coll_name, seg_lemma_embs, batch_size=512)
            elapsed = time.time() - start
            print(f"  Inserted {count} in {elapsed:.1f}s")
        
        stats = qdrant.get_collection_stats(coll_name)
        print(f"  Total points: {stats.get('row_count', 0):,}")
        qdrant.disconnect()
    else:
        print("\nFailed to connect to Qdrant")
    
    # === Neo4j (lemma_morph) ===
    neo4j = Neo4jManager()
    if neo4j.connect():
        print(f"\n--- Neo4j (lemma_morph) ---")
        neo4j.create_schema()
        
        if lemma_morph_embs:
            start = time.time()
            count = neo4j.insert_embeddings(lemma_morph_embs, batch_size=50)
            elapsed = time.time() - start
            print(f"  Inserted {count} in {elapsed:.1f}s")
        
        stats = neo4j.get_collection_stats()
        print(f"  Chunks: {stats.get('chunk_count', 0):,}")
        print(f"  Words: {stats.get('word_count', 0):,}")
        print(f"  Lemmas: {stats.get('lemma_count', 0):,}")
        neo4j.disconnect()
    else:
        print("\nFailed to connect to Neo4j (Docker required)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
