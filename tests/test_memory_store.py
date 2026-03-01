import tempfile
import unittest
from pathlib import Path

import memory_store


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        memory_store.MEMORY_DIR = root / "memory_data"
        memory_store.MEMORY_INDEX_DIR = memory_store.MEMORY_DIR / "index"
        memory_store.MEMORY_RECORDS_FILE = memory_store.MEMORY_DIR / "memory_records.jsonl"
        memory_store._memory_embeddings = memory_store.HashEmbeddings()
        memory_store.init_memory_index()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_session_isolated_search_and_clear(self):
        memory_store.write_memory("preferred release timezone is pst", "session-a")
        memory_store.write_memory("preferred release timezone is utc", "session-b")

        result_a = memory_store.search_memory("preferred release timezone is pst", "session-a", top_k=3)
        result_b = memory_store.search_memory("preferred release timezone is utc", "session-b", top_k=3)

        self.assertEqual(len(result_a.get("items", [])), 1)
        self.assertEqual(len(result_b.get("items", [])), 1)
        self.assertIn("pst", result_a["items"][0]["summary"].lower())
        self.assertIn("utc", result_b["items"][0]["summary"].lower())

        cleared = memory_store.clear_memory("session-a")
        self.assertGreaterEqual(cleared.get("cleared_count", 0), 1)

        result_after_clear = memory_store.search_memory("preferred release timezone is pst", "session-a", top_k=3)
        self.assertEqual(result_after_clear.get("items", []), [])

    def test_ttl_expiry(self):
        write_result = memory_store.write_memory("remember my timezone is pst", "session-ttl", ttl_days=1)
        self.assertTrue(write_result.get("written", False))

        records = memory_store._load_all_records()
        for record in records:
            if record.get("memory_id") == write_result.get("memory_id"):
                record["expires_at"] = "2000-01-01T00:00:00+00:00"
        memory_store._write_all_records(records)

        expired_result = memory_store.search_memory("timezone", "session-ttl", top_k=3)
        self.assertEqual(expired_result.get("items", []), [])

    def test_duplicate_guard(self):
        first = memory_store.write_memory("team timezone is pst", "session-dedup")
        second = memory_store.write_memory("team timezone is pst", "session-dedup")
        self.assertTrue(first.get("written", False))
        self.assertFalse(second.get("written", True))
        self.assertEqual(second.get("reason"), "duplicate_memory")


if __name__ == "__main__":
    unittest.main()
