from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.runtime import Runtime


class FakeVectorStore:
    def __init__(self):
        self.documents = {}

    def get(self, where=None, include=None):
        documents = self.documents
        if where:
            documents = {
                key: value
                for key, value in documents.items()
                if all(value.metadata.get(field) == expected for field, expected in where.items())
            }
        return {
            "ids": list(documents),
            "metadatas": [document.metadata for document in documents.values()],
        }

    def delete(self, ids):
        for document_id in ids:
            self.documents.pop(document_id, None)

    def add_documents(self, documents, ids):
        self.documents.update(zip(ids, documents))
        return ids


class FakeGraph:
    def invoke(self, state):
        return {
            **state,
            "answer": "Use Depends. [1]",
            "sources": [
                {
                    "citation": 1,
                    "file": "fastapi.md",
                    "section": "Dependencies",
                    "chunk_id": "0-demo",
                    "snippet": "Use Depends.",
                }
            ],
            "status": "answered",
            "retry_count": 1,
            "grounding_check": "passed",
        }


def test_query_feedback_documents_and_health(monkeypatch, tmp_path):
    settings = Settings(feedback_db=str(tmp_path / "feedback.sqlite3"))
    runtime = Runtime(FakeGraph(), FakeVectorStore(), settings)
    monkeypatch.setattr(main, "build_runtime", lambda: runtime)

    with TestClient(main.app) as client:
        query_response = client.post(
            "/query", json={"question": "How do I declare a dependency?"}
        )
        assert query_response.status_code == 200
        query_body = query_response.json()
        assert query_body["retries_used"] == 1
        assert query_body["sources"][0]["citation"] == 1
        assert query_body["grounding_check"] == "passed"

        feedback_response = client.post(
            "/feedback",
            json={"query_id": query_body["query_id"], "rating": "up"},
        )
        assert feedback_response.status_code == 200
        assert client.post(
            "/feedback",
            json={"query_id": "80e2b30c-19db-4c8f-bd8b-06e1bd60dc94", "rating": "up"},
        ).status_code == 404

        assert client.get("/documents").json() == {"documents": []}
        assert client.get("/health").json() == {"status": "ok"}


def test_root_serves_rag_interface():
    with TestClient(main.app, follow_redirects=False) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Upload your documents" in response.text
    assert "Ask a question about your documents" in response.text
    assert 'id="file-picker"' in response.text


def test_ingest_upload_indexes_markdown(monkeypatch, tmp_path):
    settings = Settings(feedback_db=str(tmp_path / "feedback.sqlite3"))
    runtime = Runtime(FakeGraph(), FakeVectorStore(), settings)
    monkeypatch.setattr(main, "build_runtime", lambda: runtime)

    with TestClient(main.app) as client:
        response = client.post(
            "/ingest",
            files={"files": ("notes.md", "# Notes\n\n## Setup\n\nInstall the package.", "text/markdown")},
        )

        assert response.status_code == 200
        assert response.json()["files_added"] == ["notes.md"]
        assert response.json()["chunks_added"] == 1
        assert client.get("/documents").json()["documents"][0]["source"] == "notes.md"


def test_ingest_accepts_document_urls(monkeypatch, tmp_path):
    settings = Settings(feedback_db=str(tmp_path / "feedback.sqlite3"))
    runtime = Runtime(FakeGraph(), FakeVectorStore(), settings)

    async def fetch_document(url, _max_bytes):
        return url, "# External guide\n\nUseful documentation."

    monkeypatch.setattr(main, "build_runtime", lambda: runtime)
    monkeypatch.setattr(main, "_fetch_url", fetch_document)

    with TestClient(main.app) as client:
        response = client.post(
            "/ingest",
            json={"urls": ["https://docs.example.org/guide.md"]},
        )

        assert response.status_code == 200
        assert response.json()["files_added"] == ["https://docs.example.org/guide.md"]
        assert response.json()["chunks_added"] == 1


def test_blank_query_is_rejected(monkeypatch, tmp_path):
    runtime = Runtime(
        FakeGraph(),
        FakeVectorStore(),
        Settings(feedback_db=str(tmp_path / "db.sqlite3")),
    )
    monkeypatch.setattr(main, "build_runtime", lambda: runtime)

    with TestClient(main.app) as client:
        assert client.post("/query", json={"question": "   "}).status_code == 422
