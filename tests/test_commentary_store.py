"""Tests for SQLite commentary store."""

import tempfile


from src.storage.commentary_store import CommentaryStore


class TestCommentaryStore:
    """Test CommentaryStore SQLite operations."""

    def _create_store(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        return CommentaryStore(tmp.name)

    def test_create_tables(self):
        store = self._create_store()
        store.connect()
        stats = store.get_stats()
        assert stats["verses"] == 0
        assert stats["commentaries"] == 0
        store.close()

    def test_insert_verse(self):
        store = self._create_store()
        store.connect()
        store.insert_verse("BhG 2.47", 2, 47, "karmaṇy evādhikāras te", "कर्मण्येवाधिकारस्ते", "kṛṣṇa", 14)
        store.conn.commit()
        verse = store.get_verse("BhG 2.47")
        assert verse is not None
        assert verse["verse_ref"] == "BhG 2.47"
        assert verse["chapter_num"] == 2
        assert "karmaṇy" in verse["text_iast"]
        store.close()

    def test_insert_commentary(self):
        store = self._create_store()
        store.connect()
        store.insert_verse("BhG 2.47", 2, 47, "karmaṇy evādhikāras te", "कर्मण्येवाधिकारस्ते")
        store.insert_commentary("BhG_2.47_sridhara", "BhG 2.47", "sridhara", "Sridhara commentary text")
        store.conn.commit()
        comms = store.get_commentaries_for_verse("BhG 2.47")
        assert len(comms) == 1
        assert comms[0]["commentator"] == "sridhara"
        assert comms[0]["text_iast"] == "Sridhara commentary text"
        store.close()

    def test_multiple_commentaries(self):
        store = self._create_store()
        store.connect()
        store.insert_verse("BhG 2.47", 2, 47, "verse text", "")
        store.insert_commentary("c1", "BhG 2.47", "sridhara", "Sridhara says...")
        store.insert_commentary("c2", "BhG 2.47", "visvanatha", "Visvanatha says...")
        store.insert_commentary("c3", "BhG 2.47", "baladeva", "Baladeva says...")
        store.conn.commit()
        comms = store.get_commentaries_for_verse("BhG 2.47")
        assert len(comms) == 3
        commentators = [c["commentator"] for c in comms]
        assert "sridhara" in commentators
        assert "visvanatha" in commentators
        assert "baladeva" in commentators
        store.close()

    def test_get_commentaries_for_verses(self):
        store = self._create_store()
        store.connect()
        store.insert_verse("BhG 2.47", 2, 47, "verse 2.47", "")
        store.insert_verse("BhG 3.35", 3, 35, "verse 3.35", "")
        store.insert_commentary("c1", "BhG 2.47", "sridhara", "Commentary on 2.47")
        store.insert_commentary("c2", "BhG 3.35", "sridhara", "Commentary on 3.35")
        store.conn.commit()
        result = store.get_commentaries_for_verses(["BhG 2.47", "BhG 3.35"])
        assert "BhG 2.47" in result
        assert "BhG 3.35" in result
        assert len(result["BhG 2.47"]) == 1
        assert len(result["BhG 3.35"]) == 1
        store.close()

    def test_empty_verse_ref(self):
        store = self._create_store()
        store.connect()
        result = store.get_commentaries_for_verse("BhG 99.99")
        assert result == []
        store.close()

    def test_get_commentaries_for_empty_list(self):
        store = self._create_store()
        store.connect()
        result = store.get_commentaries_for_verses([])
        assert result == {}
        store.close()

    def test_stats(self):
        store = self._create_store()
        store.connect()
        store.insert_verse("BhG 1.1", 1, 1, "verse text", "")
        store.insert_commentary("c1", "BhG 1.1", "sridhara", "comm 1")
        store.insert_commentary("c2", "BhG 1.1", "visvanatha", "comm 2")
        store.conn.commit()
        stats = store.get_stats()
        assert stats["verses"] == 1
        assert stats["commentaries"] == 2
        assert stats["commentators"]["sridhara"] == 1
        assert stats["commentators"]["visvanatha"] == 1
        store.close()
