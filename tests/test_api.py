"""Tests for the FastAPI service contract."""
import unittest

try:
    from fastapi.testclient import TestClient

    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.api import app, get_service
except ModuleNotFoundError:
    TestClient = None
    app = None
    get_service = None


class FakeService:
    def health(self):
        return {"ok": True, "components": {"qdrant": True, "neo4j": False, "sqlite": True, "llm": True}}

    def stats(self):
        return {"components": self.health()["components"], "sqlite": {"total_verses": 700}}

    def ask(self, **kwargs):
        return {
            "ok": True,
            "query": kwargs["query"],
            "answer": "Answer",
            "has_evidence": True,
            "evidence": {"canonical_verses": [{"verse_id": "BhG 2.47"}]},
        }

    def search(self, **kwargs):
        return {
            "ok": True,
            "query": kwargs["query"],
            "method": "cross_db_rrf",
            "total_results": 1,
            "results": [{"id": "chunk-a", "verse_id": "BhG 2.47"}],
        }

    def docker(self, action):
        return {"ok": True, "action": action, "output": "No containers running."}


@unittest.skipIf(TestClient is None, "fastapi is not installed")
class TestAPI(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_service] = lambda: FakeService()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["components"]["qdrant"])

    def test_stats_endpoint(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sqlite"]["total_verses"], 700)

    def test_ask_endpoint(self):
        response = self.client.post("/api/ask", json={"query": "Explain BG 2.47"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["query"], "Explain BG 2.47")
        self.assertTrue(response.json()["has_evidence"])

    def test_search_endpoint(self):
        response = self.client.post("/api/search", json={"query": "karma yoga"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["method"], "cross_db_rrf")
        self.assertEqual(response.json()["total_results"], 1)

    def test_docker_endpoint(self):
        response = self.client.post("/api/docker", json={"action": "status"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "status")


if __name__ == "__main__":
    unittest.main()
