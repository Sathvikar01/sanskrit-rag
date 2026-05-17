"""Migrate commentaries from chunks.jsonl to SQLite."""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import json

from src.storage.commentary_store import CommentaryStore


def migrate(chunks_path: str = "data/processed/chunks.jsonl", db_path: str = "data/storage/commentaries.db"):
    """Populate SQLite commentary store from chunks.jsonl."""
    store = CommentaryStore(db_path)
    store.connect()

    verse_count = 0
    commentary_count = 0

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunk_type = chunk.get("chunk_type", "")
            verse_ref = chunk.get("verse_ref", "")

            if chunk_type == "verse":
                store.insert_verse(
                    verse_ref=verse_ref,
                    chapter_num=int(chunk.get("chapter_num", 0)),
                    verse_num=int(chunk.get("verse_num", 0)),
                    text_iast=chunk.get("text_iast", ""),
                    text_devanagari=chunk.get("text_devanagari", ""),
                    speaker=chunk.get("speaker", ""),
                    word_count=int(chunk.get("word_count", 0)),
                )
                verse_count += 1

            elif chunk_type == "commentary":
                store.insert_commentary(
                    chunk_id=chunk.get("chunk_id", ""),
                    verse_ref=verse_ref,
                    commentator=chunk.get("commentator", ""),
                    text_iast=chunk.get("text_iast", ""),
                    text_devanagari=chunk.get("text_devanagari", ""),
                    word_count=int(chunk.get("word_count", 0)),
                )
                commentary_count += 1

    store.conn.commit()
    stats = store.get_stats()
    store.close()

    print("Migration complete:")
    print(f"  Verses inserted: {verse_count}")
    print(f"  Commentaries inserted: {commentary_count}")
    print(f"  DB stats: {stats}")
    print(f"  DB path: {db_path}")


if __name__ == "__main__":
    migrate()
