"""Ingest cached embeddings into Qdrant."""
import sys
import os
import uuid
import time
from pathlib import Path
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.qdrant_manager import QdrantManager, create_all_collections
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
    
    # Only ingest seg_lemma into Qdrant, keep raw and lemma_morph in cache
    seg_lemma_embs = []
    for emb in embeddings:
        dtype = emb.metadata.get("dataset_type", detect_dataset_type(emb.text))
        if dtype == "seg_lemma":
            seg_lemma_embs.append(emb)
    
    print(f"\nIngesting only seg_lemma into Qdrant: {len(seg_lemma_embs)} embeddings")
    print(f"Keeping raw and lemma_morph in cache only")
    
    manager = QdrantManager()
    if not manager.connect():
        print("Failed to connect to Qdrant.")
        return
    
    print("\nCreating Qdrant collection for seg_lemma...")
    coll_name = COLLECTION_NAMES["seg_lemma"]
    manager.create_collection(coll_name)
    
    print(f"\nInserting {len(seg_lemma_embs)} embeddings into {coll_name}...")
    start = time.time()
    count = manager.insert_embeddings(coll_name, seg_lemma_embs, batch_size=512)
    elapsed = time.time() - start
    print(f"  Inserted {count} in {elapsed:.1f}s")
    
    print("\nCollection stats:")
    stats = manager.get_collection_stats(coll_name)
    print(f"  {coll_name}: {stats.get('row_count', 0):,} points")
    
    manager.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
