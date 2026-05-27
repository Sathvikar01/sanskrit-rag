"""SQLite-backed cache for query expansion, reranked evidence, and LLM answers."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = ROOT_DIR / "cache" / "evidence_cache.sqlite"


class EvidenceCache:
    """Small JSON cache with per-namespace TTLs."""

    DEFAULT_TTLS = {
        "query_expansion": 60 * 60 * 24 * 30,
        "rerank_bundle": 60 * 60 * 12,
        "llm_answer": 60 * 60 * 6,
        "embedding": 60 * 60 * 24 * 90,
    }

    def __init__(self, db_path: str | Path = DEFAULT_CACHE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY (namespace, cache_key)
            )
            """
        )
        self._conn.commit()

    def build_key(self, payload: Dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT value_json, expires_at FROM cache_entries WHERE namespace = ? AND cache_key = ?",
            (namespace, key),
        ).fetchone()
        if not row:
            return None
        if float(row["expires_at"]) < time.time():
            self._conn.execute(
                "DELETE FROM cache_entries WHERE namespace = ? AND cache_key = ?",
                (namespace, key),
            )
            self._conn.commit()
            return None
        return json.loads(row["value_json"])

    def set(self, namespace: str, key: str, value: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.DEFAULT_TTLS.get(namespace, 3600)
        now = time.time()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cache_entries
            (namespace, cache_key, value_json, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (namespace, key, json.dumps(value, ensure_ascii=False, default=str), now, now + ttl),
        )
        self._conn.commit()

    def status(self) -> Dict[str, Any]:
        rows = self._conn.execute(
            "SELECT namespace, COUNT(*) AS c FROM cache_entries WHERE expires_at >= ? GROUP BY namespace",
            (time.time(),),
        ).fetchall()
        return {
            "path": str(self.db_path),
            "entries": {row["namespace"]: row["c"] for row in rows},
        }

    def close(self) -> None:
        self._conn.close()
