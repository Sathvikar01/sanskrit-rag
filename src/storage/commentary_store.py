"""SQLite store for Bhagavad Gita commentaries.

Stores verse-commentary pairs in a lightweight SQLite database.
Separate from Neo4j/FAISS/BM25 — used only for commentary lookup.
"""

import sqlite3
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = Path("data/storage/commentaries.db")


class CommentaryStore:
    """SQLite-backed commentary store for Bhagavad Gita."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """Open database connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _ensure_connected(self):
        """Raise if not connected."""
        if self.conn is None:
            raise RuntimeError("CommentaryStore not connected. Call connect() first.")

    def _create_tables(self):
        """Create tables if they don't exist."""
        self._ensure_connected()
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS verses (
                verse_ref TEXT PRIMARY KEY,
                chapter_num INTEGER,
                verse_num INTEGER,
                text_iast TEXT NOT NULL,
                text_devanagari TEXT,
                speaker TEXT,
                word_count INTEGER
            );

            CREATE TABLE IF NOT EXISTS commentaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                verse_ref TEXT NOT NULL,
                commentator TEXT NOT NULL,
                text_iast TEXT NOT NULL,
                text_devanagari TEXT,
                word_count INTEGER,
                FOREIGN KEY (verse_ref) REFERENCES verses(verse_ref)
            );

            CREATE INDEX IF NOT EXISTS idx_commentaries_verse_ref
                ON commentaries(verse_ref);
            CREATE INDEX IF NOT EXISTS idx_commentaries_commentator
                ON commentaries(commentator);
        """)
        self.conn.commit()

    def insert_verse(
        self,
        verse_ref: str,
        chapter_num: int,
        verse_num: int,
        text_iast: str,
        text_devanagari: str = "",
        speaker: str = "",
        word_count: int = 0,
    ):
        """Insert or replace a verse."""
        self._ensure_connected()
        self.conn.execute(
            """INSERT OR REPLACE INTO verses
               (verse_ref, chapter_num, verse_num, text_iast, text_devanagari, speaker, word_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (verse_ref, chapter_num, verse_num, text_iast, text_devanagari, speaker, word_count),
        )

    def insert_commentary(
        self,
        chunk_id: str,
        verse_ref: str,
        commentator: str,
        text_iast: str,
        text_devanagari: str = "",
        word_count: int = 0,
    ):
        """Insert or replace a commentary."""
        self._ensure_connected()
        self.conn.execute(
            """INSERT OR REPLACE INTO commentaries
               (chunk_id, verse_ref, commentator, text_iast, text_devanagari, word_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (chunk_id, verse_ref, commentator, text_iast, text_devanagari, word_count),
        )

    def get_commentaries_for_verse(self, verse_ref: str) -> list[dict]:
        """Get all commentaries for a specific verse.

        Args:
            verse_ref: Verse reference (e.g., 'BhG 2.47').

        Returns:
            List of commentary dicts with commentator, text, etc.
        """
        self._ensure_connected()
        cursor = self.conn.execute(
            """SELECT chunk_id, verse_ref, commentator, text_iast, text_devanagari, word_count
               FROM commentaries
               WHERE verse_ref = ?
               ORDER BY commentator""",
            (verse_ref,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_commentaries_for_verses(self, verse_refs: list[str]) -> dict[str, list[dict]]:
        """Get commentaries for multiple verses at once.

        Args:
            verse_refs: List of verse references.

        Returns:
            Dict mapping verse_ref to list of commentary dicts.
        """
        if not verse_refs:
            return {}

        self._ensure_connected()
        placeholders = ",".join("?" * len(verse_refs))
        cursor = self.conn.execute(
            f"""SELECT chunk_id, verse_ref, commentator, text_iast, text_devanagari, word_count
                FROM commentaries
                WHERE verse_ref IN ({placeholders})
                ORDER BY verse_ref, commentator""",
            verse_refs,
        )
        result: dict[str, list[dict]] = {}
        for row in cursor.fetchall():
            row_dict = dict(row)
            ref = row_dict["verse_ref"]
            result.setdefault(ref, []).append(row_dict)
        return result

    def get_verse(self, verse_ref: str) -> Optional[dict]:
        """Get verse data by reference."""
        self._ensure_connected()
        cursor = self.conn.execute(
            "SELECT * FROM verses WHERE verse_ref = ?", (verse_ref,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_stats(self) -> dict:
        """Get database statistics."""
        self._ensure_connected()
        verse_count = self.conn.execute("SELECT COUNT(*) FROM verses").fetchone()[0]
        commentary_count = self.conn.execute("SELECT COUNT(*) FROM commentaries").fetchone()[0]
        commentators = self.conn.execute(
            "SELECT commentator, COUNT(*) FROM commentaries GROUP BY commentator"
        ).fetchall()
        return {
            "verses": verse_count,
            "commentaries": commentary_count,
            "commentators": {row[0]: row[1] for row in commentators},
        }
