"""Rebuild a Qdrant collection with the configured local embedding backend."""
import argparse
import io
import sys
import time
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import COLLECTION_NAMES
from src.embedding_client import NVIDIAEmbeddingClient
from src.qdrant_manager import QdrantManager
from src.xml_parser import TEIXMLParser, TextChunk


def configure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Qdrant with local SanskritFT embeddings.")
    parser.add_argument(
        "--dataset",
        choices=sorted(COLLECTION_NAMES.keys()),
        default="seg_lemma",
        help="Source dataset to parse and embed.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name. Defaults to the configured dataset collection.",
    )
    parser.add_argument("--chapter", type=int, default=None, help="Optional chapter-only rebuild.")
    parser.add_argument("--limit", type=int, default=None, help="Optional chunk limit for smoke runs.")
    parser.add_argument("--force", action="store_true", help="Drop and recreate the collection first.")
    parser.add_argument("--insert-batch-size", type=int, default=128)
    return parser.parse_args()


def filter_chunks(chunks: List[TextChunk], chapter: int | None, limit: int | None) -> List[TextChunk]:
    if chapter is not None:
        prefix = f"BhG {chapter}."
        chunks = [chunk for chunk in chunks if (chunk.verse_id or "").startswith(prefix)]
    if limit is not None:
        chunks = chunks[:limit]
    return chunks


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    collection_name = args.collection or COLLECTION_NAMES[args.dataset]

    embedder = NVIDIAEmbeddingClient()
    if embedder.backend != "local":
        raise RuntimeError(
            "This rebuild script is intended for EMBEDDING_BACKEND=local. "
            f"Current backend: {embedder.backend}"
        )

    parser = TEIXMLParser()
    chunks_by_dataset = parser.parse_all_datasets(str(ROOT_DIR))
    chunks = filter_chunks(chunks_by_dataset.get(args.dataset, []), args.chapter, args.limit)
    if not chunks:
        raise RuntimeError(f"No chunks found for dataset={args.dataset}, chapter={args.chapter}")

    print("=" * 72)
    print("Qdrant Local Embedding Rebuild")
    print("=" * 72)
    print(f"Dataset: {args.dataset}")
    print(f"Collection: {collection_name}")
    print(f"Embedding backend: {embedder.backend}")
    print(f"Embedding model: {embedder.model}")
    print(f"Chunks: {len(chunks)}")

    qdrant = QdrantManager(collection_name=collection_name)
    if not qdrant.connect():
        raise RuntimeError("Could not connect to Qdrant")

    qdrant.create_collection(collection_name, drop_if_exists=args.force)

    start = time.time()
    embeddings = embedder.embed_chunks(chunks, show_progress=True)
    embed_seconds = time.time() - start
    print(f"Embedding completed in {embed_seconds:.1f}s")

    start = time.time()
    inserted = qdrant.insert_embeddings(
        collection_name,
        embeddings,
        batch_size=args.insert_batch_size,
        show_progress=True,
    )
    insert_seconds = time.time() - start

    stats = qdrant.get_collection_stats(collection_name)
    print(f"Inserted: {inserted}")
    print(f"Insert seconds: {insert_seconds:.1f}")
    print(f"Collection points: {stats.get('row_count', 0)}")
    qdrant.disconnect()


if __name__ == "__main__":
    main()
