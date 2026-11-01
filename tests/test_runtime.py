import pytest

from app import runtime
from app.config import Settings


def test_runtime_requires_gemini_api_key():
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        runtime.build_runtime(Settings(_env_file=None, google_api_key=None))


def test_runtime_builds_gemini_chat_and_embeddings(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None,
        google_api_key="test-key",
        docs_dir=str(tmp_path / "docs"),
        chroma_dir=str(tmp_path / "chroma"),
    )
    calls = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            calls["embeddings"] = kwargs

    class FakeChat:
        def __init__(self, **kwargs):
            calls["chat"] = kwargs

    class FakeVectorStore:
        def __init__(self, **kwargs):
            calls["vectorstore"] = kwargs

    graph_marker = object()
    monkeypatch.setattr(runtime, "GoogleGenerativeAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(runtime, "ChatGoogleGenerativeAI", FakeChat)
    monkeypatch.setattr(runtime, "Chroma", FakeVectorStore)
    monkeypatch.setattr(runtime, "ingest_local_docs", lambda *_args: None)
    monkeypatch.setattr(runtime, "build_graph", lambda llm, store, configured: graph_marker)

    result = runtime.build_runtime(settings)

    assert result.graph is graph_marker
    assert calls["embeddings"] == {
        "model": "gemini-embedding-001",
        "google_api_key": "test-key",
    }
    assert calls["chat"] == {
        "model": "gemini-2.5-flash",
        "google_api_key": "test-key",
        "temperature": 0,
    }
    assert calls["vectorstore"]["collection_name"] == "technical_docs_gemini"
