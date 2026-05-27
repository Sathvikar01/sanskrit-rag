"""SQLite database manager for storing raw Bhagavad Gita verses."""
import sqlite3
import re
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from lxml import etree

from src.commentary_manager import get_author_display_name, normalize_author_key
from src.text_quality import clean_text, is_usable_text


# The bundled dataset.xml expands to 677 distinct verse IDs.
EXPECTED_BHAGAVAD_GITA_VERSE_COUNT = 677
VERSE_ID_RE = re.compile(r"^BhG\s+(\d+)\.(\d+)$")
VERSE_MARKER_RE = re.compile(r"^BhG\s*(\d+)\.(\d+)(?:-(\d+))?")


def parse_verse_id(verse_id: str) -> Optional[Tuple[int, int]]:
    """Return chapter and verse number for a canonical BhG verse id."""
    match = VERSE_ID_RE.match((verse_id or "").strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def expand_verse_marker(text: str) -> List[str]:
    """Expand markers like 'BhG 1.15-18' into concrete verse ids."""
    match = VERSE_MARKER_RE.match((text or "").strip())
    if not match:
        return []
    chapter = int(match.group(1))
    verse_start = int(match.group(2))
    verse_end = int(match.group(3)) if match.group(3) else verse_start
    if verse_end < verse_start:
        verse_end = verse_start
    return [f"BhG {chapter}.{verse_num}" for verse_num in range(verse_start, verse_end + 1)]


class VerseDatabase:
    """SQLite database for storing and retrieving raw Sanskrit verses."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "verses.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._local = threading.local()

    @property
    def _conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
            self._ensure_tables(conn)
        return self._local.conn

    def _ensure_tables(self, conn):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS verses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verse_id TEXT NOT NULL UNIQUE,
                chapter INTEGER NOT NULL,
                verse_num INTEGER NOT NULL,
                speaker TEXT,
                sanskrit_text TEXT NOT NULL,
                transliteration TEXT,
                word_count INTEGER DEFAULT 0,
                source_file TEXT DEFAULT 'dataset.xml'
            );

            CREATE TABLE IF NOT EXISTS commentaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verse_id TEXT NOT NULL,
                commentator TEXT NOT NULL,
                sanskrit_text TEXT NOT NULL,
                FOREIGN KEY (verse_id) REFERENCES verses(verse_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS verse_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verse_id TEXT NOT NULL,
                line_num INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (verse_id) REFERENCES verses(verse_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_verses_chapter ON verses(chapter);
            CREATE INDEX IF NOT EXISTS idx_verses_verse_id ON verses(verse_id);
            CREATE INDEX IF NOT EXISTS idx_commentaries_verse ON commentaries(verse_id);
            CREATE INDEX IF NOT EXISTS idx_commentaries_commentator ON commentaries(commentator);
        """)
        conn.commit()

    def _create_tables(self):
        pass

    def connect(self):
        _ = self._conn

    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def insert_verse(
        self,
        verse_id: str,
        chapter: int,
        verse_num: int,
        speaker: str,
        sanskrit_text: str,
        lines: List[str],
        commentaries: List[Tuple[str, str]] = None,
    ) -> bool:
        """Insert a verse with its lines and optional commentaries."""
        try:
            conn = self._conn
            conn.execute(
                """INSERT OR REPLACE INTO verses
                (verse_id, chapter, verse_num, speaker, sanskrit_text, word_count)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    verse_id,
                    chapter,
                    verse_num,
                    speaker or "",
                    sanskrit_text,
                    len(sanskrit_text.split()),
                ),
            )

            conn.execute(
                "DELETE FROM verse_lines WHERE verse_id = ?", (verse_id,)
            )
            for i, line in enumerate(lines, 1):
                conn.execute(
                    "INSERT INTO verse_lines (verse_id, line_num, text) VALUES (?, ?, ?)",
                    (verse_id, i, line),
                )

            if commentaries:
                conn.execute(
                    "DELETE FROM commentaries WHERE verse_id = ?", (verse_id,)
                )
                for commentator, text in commentaries:
                    conn.execute(
                        "INSERT INTO commentaries (verse_id, commentator, sanskrit_text) VALUES (?, ?, ?)",
                        (verse_id, commentator, text),
                    )

            conn.commit()
            return True
        except Exception as e:
            print(f"Error inserting verse {verse_id}: {e}")
            self._conn.rollback()
            return False

    def _row_to_verse(self, row, lookup_verse_id: Optional[str] = None) -> Dict:
        """Convert a SQLite verse row into the public verse dictionary."""
        conn = self._conn
        verse_id = lookup_verse_id or row["verse_id"]
        parsed = parse_verse_id(verse_id)
        chapter = parsed[0] if parsed else row["chapter"]
        verse_num = parsed[1] if parsed else row["verse_num"]

        lines = [
            r["text"]
            for r in conn.execute(
                "SELECT text FROM verse_lines WHERE verse_id = ? ORDER BY line_num",
                (row["verse_id"],),
            ).fetchall()
        ]

        commentaries = [
            {"commentator": r["commentator"], "text": r["sanskrit_text"]}
            for r in conn.execute(
                "SELECT commentator, sanskrit_text FROM commentaries WHERE verse_id = ?",
                (row["verse_id"],),
            ).fetchall()
        ]

        return {
            "verse_id": verse_id,
            "chapter": chapter,
            "verse_num": verse_num,
            "speaker": row["speaker"],
            "sanskrit_text": row["sanskrit_text"],
            "lines": lines,
            "commentaries": commentaries,
            "word_count": row["word_count"],
        }

    def _find_range_container_row(self, verse_id: str):
        """Find a grouped legacy row that probably contains a missing verse."""
        parsed = parse_verse_id(verse_id)
        if not parsed:
            return None

        chapter, verse_num = parsed
        conn = self._conn
        previous = conn.execute(
            """SELECT * FROM verses
            WHERE chapter = ? AND verse_num < ?
            ORDER BY verse_num DESC LIMIT 1""",
            (chapter, verse_num),
        ).fetchone()
        if not previous:
            return None

        next_row = conn.execute(
            """SELECT verse_num FROM verses
            WHERE chapter = ? AND verse_num > ?
            ORDER BY verse_num ASC LIMIT 1""",
            (chapter, previous["verse_num"]),
        ).fetchone()
        next_verse_num = next_row["verse_num"] if next_row else previous["verse_num"] + 1
        if previous["verse_num"] < verse_num < next_verse_num and next_verse_num - previous["verse_num"] <= 10:
            return previous
        return None

    def get_verse(self, verse_id: str) -> Optional[Dict]:
        """Get a single verse by ID with legacy range fallback."""
        row = self._conn.execute(
            "SELECT * FROM verses WHERE verse_id = ?", (verse_id,)
        ).fetchone()
        if row:
            return self._row_to_verse(row)

        container = self._find_range_container_row(verse_id)
        if container:
            return self._row_to_verse(container, lookup_verse_id=verse_id)
        return None

    def get_verses_by_ids(self, verse_ids: List[str]) -> List[Dict]:
        """Get multiple verses by their IDs."""
        results = []
        for vid in verse_ids:
            verse = self.get_verse(vid)
            if verse:
                results.append(verse)
        return results

    def get_verses_by_chapter(self, chapter: int) -> List[Dict]:
        """Get all verses from a chapter."""
        conn = self._conn
        rows = conn.execute(
            "SELECT * FROM verses WHERE chapter = ? ORDER BY verse_num",
            (chapter,),
        ).fetchall()

        results = []
        for row in rows:
            lines = [
                r["text"]
                for r in conn.execute(
                    "SELECT text FROM verse_lines WHERE verse_id = ? ORDER BY line_num",
                    (row["verse_id"],),
                ).fetchall()
            ]
            commentaries = [
                {"commentator": r["commentator"], "text": r["sanskrit_text"]}
                for r in conn.execute(
                    "SELECT commentator, sanskrit_text FROM commentaries WHERE verse_id = ?",
                    (row["verse_id"],),
                ).fetchall()
            ]
            results.append(
                {
                    "verse_id": row["verse_id"],
                    "chapter": row["chapter"],
                    "verse_num": row["verse_num"],
                    "speaker": row["speaker"],
                    "sanskrit_text": row["sanskrit_text"],
                    "lines": lines,
                    "commentaries": commentaries,
                    "word_count": row["word_count"],
                }
            )
        return results

    def search_verses(self, query: str) -> List[Dict]:
        """Search verses by keyword in sanskrit text."""
        conn = self._conn
        terms = query.lower().split()
        like_patterns = [f"%{t}%" for t in terms]

        conditions = " AND ".join(
            ["sanskrit_text LIKE ?" for _ in like_patterns]
        )
        rows = conn.execute(
            f"SELECT * FROM verses WHERE {conditions} ORDER BY chapter, verse_num",
            like_patterns,
        ).fetchall()

        results = []
        for row in rows:
            lines = [
                r["text"]
                for r in conn.execute(
                    "SELECT text FROM verse_lines WHERE verse_id = ? ORDER BY line_num",
                    (row["verse_id"],),
                ).fetchall()
            ]
            results.append(
                {
                    "verse_id": row["verse_id"],
                    "chapter": row["chapter"],
                    "verse_num": row["verse_num"],
                    "speaker": row["speaker"],
                    "sanskrit_text": row["sanskrit_text"],
                    "lines": lines,
                    "word_count": row["word_count"],
                }
            )
        return results

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = self._conn
        verse_count = conn.execute("SELECT COUNT(*) as c FROM verses").fetchone()["c"]
        commentary_count = conn.execute("SELECT COUNT(*) as c FROM commentaries").fetchone()["c"]
        chapters = conn.execute(
            "SELECT COUNT(DISTINCT chapter) as c FROM verses"
        ).fetchone()["c"]

        return {
            "total_verses": verse_count,
            "total_commentaries": commentary_count,
            "chapters": chapters,
        }


def _legacy_ingest_xml_to_sqlite(xml_path: str, db_path: str = None) -> VerseDatabase:
    """Legacy parser retained for comparison; use ingest_xml_to_sqlite instead."""
    db = VerseDatabase(db_path)
    db.connect()

    tree = etree.parse(xml_path, etree.XMLParser(remove_blank_text=True))
    root = tree.getroot()
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    verses_ingested = 0
    current_verse_id = None
    current_chapter = 0
    current_verse_num = 0
    current_speaker = ""
    current_lines = []
    current_commentaries = []
    current_commentator = None
    current_commentary_texts = []

    def save_current_verse():
        nonlocal verses_ingested
        if current_verse_id and current_lines:
            sanskrit_text = " ".join(current_lines)
            commentaries = []
            if current_commentator and current_commentary_texts:
                commentaries.append(
                    (current_commentator, " ".join(current_commentary_texts))
                )

            db.insert_verse(
                verse_id=current_verse_id,
                chapter=current_chapter,
                verse_num=current_verse_num,
                speaker=current_speaker,
                sanskrit_text=sanskrit_text,
                lines=current_lines,
                commentaries=commentaries if commentaries else None,
            )
            verses_ingested += 1

    for element in root.iter():
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            text = (element.text or "").strip()
            if not text:
                continue

            verse_ids = expand_verse_marker(text)
            if verse_ids:
                save_current_verse()
                current_verse_ids = verse_ids
                current_verse_id = verse_ids[0]
                current_chapter, current_verse_num = parse_verse_id(current_verse_id) or (0, 0)
                current_lines = []
                current_commentaries = []
                current_commentator = None
                current_commentary_texts = []

                speaker_match = re.match(r"^BhG\s*\d+\.\d+\s+(.+)$", text)
                current_speaker = speaker_match.group(1).strip() if speaker_match else ""
                continue

            commentator_match = re.match(r"^(.+?)\s*[-–—]\s*$", text)
            if commentator_match and current_verse_id:
                if current_commentator and current_commentary_texts:
                    current_commentaries.append(
                        (current_commentator, " ".join(current_commentary_texts))
                    )
                current_commentator = commentator_match.group(1).strip()
                current_commentary_texts = []
                continue

        elif tag == "l" and current_verse_id:
            text = (element.text or "").strip()
            if text:
                current_lines.append(text)

    save_current_verse()
    db.close()
    print(f"Ingested {verses_ingested} verses into {db.db_path}")
    return db


def ingest_xml_to_sqlite(xml_path: str, db_path: str = None) -> VerseDatabase:
    """Parse dataset.xml and ingest verses plus commentary into SQLite."""
    db = VerseDatabase(db_path)
    db.connect()

    tree = etree.parse(xml_path, etree.XMLParser(remove_blank_text=True))
    root = tree.getroot()

    verses_ingested = 0
    current_verse_id = None
    current_verse_ids = []
    current_chapter = 0
    current_verse_num = 0
    current_speaker = ""
    current_lines = []
    current_commentaries = []
    current_commentator = None
    current_commentary_texts = []

    def save_current_verse():
        nonlocal verses_ingested, current_commentary_texts
        if not current_verse_id or not current_lines:
            return

        commentaries = list(current_commentaries)
        if current_commentator and current_commentary_texts:
            commentaries.append((current_commentator, clean_text(" ".join(current_commentary_texts))))
            current_commentary_texts = []

        cleaned_commentaries = []
        for commentator, text in commentaries:
            cleaned = clean_text(text)
            if cleaned and is_usable_text(cleaned, min_score=0.25):
                cleaned_commentaries.append((commentator, cleaned))

        lines = [clean_text(line) for line in current_lines if clean_text(line)]
        text = clean_text(" ".join(current_lines))
        for verse_id in (current_verse_ids or [current_verse_id]):
            chapter, verse_num = parse_verse_id(verse_id) or (current_chapter, current_verse_num)
            db.insert_verse(
                verse_id=verse_id,
                chapter=chapter,
                verse_num=verse_num,
                speaker=current_speaker,
                sanskrit_text=text,
                lines=lines,
                commentaries=cleaned_commentaries if cleaned_commentaries else None,
            )
            verses_ingested += 1

    def flush_commentary_buffer():
        nonlocal current_commentary_texts
        if current_commentator and current_commentary_texts:
            current_commentaries.append((current_commentator, clean_text(" ".join(current_commentary_texts))))
        current_commentary_texts = []

    def split_commentary_marker(text: str):
        marker_match = re.match(r"^\s*(.+?)(?:\s*[:\-–—]{1,2}\s*)(.*)$", text)
        if not marker_match:
            return None, text
        author_key = normalize_author_key(marker_match.group(1))
        if not author_key:
            return None, text
        return get_author_display_name(author_key), marker_match.group(2).strip()

    for element in root.iter():
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            text = clean_text(" ".join(element.itertext()))
            if not text:
                continue

            verse_ids = expand_verse_marker(text)
            if verse_ids:
                save_current_verse()
                current_verse_ids = verse_ids
                current_verse_id = verse_ids[0]
                current_chapter, current_verse_num = parse_verse_id(current_verse_id) or (0, 0)
                current_lines = []
                current_commentaries = []
                current_commentator = None
                current_commentary_texts = []

                speaker_match = re.match(r"^BhG\s*\d+\.\d+\s+(.+)$", text)
                current_speaker = speaker_match.group(1).strip() if speaker_match else ""
                continue

            commentator, remainder = split_commentary_marker(text)
            if commentator and current_verse_id:
                flush_commentary_buffer()
                current_commentator = commentator
                current_commentary_texts = []
                if remainder:
                    current_commentary_texts.append(remainder)
                continue

            if current_verse_id and current_commentator:
                current_commentary_texts.append(text)
                continue

        elif tag == "l" and current_verse_id:
            text = clean_text(" ".join(element.itertext()))
            if not text:
                continue
            if current_commentator:
                current_commentary_texts.append(text)
            else:
                current_lines.append(text)

    save_current_verse()
    db.close()
    print(f"Ingested {verses_ingested} verses into {db.db_path}")
    return db


if __name__ == "__main__":
    import sys

    xml_path = Path(__file__).parent.parent / "dataset.xml"
    if not xml_path.exists():
        print(f"XML file not found: {xml_path}")
        sys.exit(1)

    db = ingest_xml_to_sqlite(str(xml_path))
    db.connect()
    stats = db.get_stats()
    print(f"Database stats: {stats}")

    sample = db.get_verse("BhG 1.1")
    if sample:
        print(f"\nSample verse: {sample['verse_id']}")
        print(f"Speaker: {sample['speaker']}")
        print(f"Lines: {sample['lines']}")
        print(f"Commentaries: {len(sample['commentaries'])}")

    db.close()
