from types import SimpleNamespace

from langchain_core.documents import Document

from app.config import Settings
from app.graph import Analysis, Grade, GroundingGrade, build_graph


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response

    def invoke(self, _prompt):
        return self.response


class FakeLLM:
    def __init__(
        self,
        grades,
        answer="A dependency is declared with Depends. [1]",
        grounding_verdict="supported",
    ):
        self.grades = iter(grades)
        self.answer = answer
        self.grounding_verdict = grounding_verdict

    def with_structured_output(self, model):
        if model is Analysis:
            return FakeStructuredModel(
                Analysis(rewritten_query="FastAPI dependency injection", query_type="how-to")
            )
        if model is GroundingGrade:
            return FakeStructuredModel(GroundingGrade(verdict=self.grounding_verdict))
        return FakeStructuredModel(next(self.grades))

    def invoke(self, prompt):
        if "Write a concise, meaningfully different search query" in prompt:
            return SimpleNamespace(content="alternate search terms")
        return SimpleNamespace(content=self.answer)


class FakeVectorStore:
    def __init__(self, results):
        self.results = iter(results)

    def similarity_search(self, _query, k):
        return next(self.results)[:k]


def _initial_state():
    return {
        "question": "How do I declare a dependency?",
        "rewritten_query": "",
        "query_type": "",
        "documents": [],
        "relevant_docs": [],
        "retry_count": 0,
        "max_retries": 2,
        "answer": "",
        "sources": [],
        "status": "",
        "grounding_check": "not_run",
    }


def test_graph_answers_with_citations():
    document = Document(
        page_content="Use Depends to declare a dependency.",
        metadata={"source": "fastapi.md", "section": "Dependencies", "chunk_id": "0-test"},
    )
    graph = build_graph(
        FakeLLM([Grade(verdict="relevant")]),
        FakeVectorStore([[document]]),
        Settings(),
    )

    result = graph.invoke(_initial_state())

    assert result["status"] == "answered"
    assert result["sources"][0]["citation"] == 1
    assert result["sources"][0]["file"] == "fastapi.md"
    assert result["retry_count"] == 0
    assert result["grounding_check"] == "passed"


def test_graph_retries_then_answers():
    first = Document(page_content="Unrelated chunk", metadata={"source": "x.md", "chunk_id": "0"})
    second = Document(
        page_content="Depends declares dependencies.",
        metadata={"source": "fastapi.md", "section": "Dependencies", "chunk_id": "1"},
    )
    graph = build_graph(
        FakeLLM([Grade(verdict="irrelevant"), Grade(verdict="relevant")]),
        FakeVectorStore([[first], [second]]),
        Settings(),
    )

    result = graph.invoke(_initial_state())

    assert result["status"] == "answered"
    assert result["retry_count"] == 1
    assert result["sources"][0]["file"] == "fastapi.md"
    assert result["grounding_check"] == "passed"


def test_graph_stops_after_retry_limit():
    document = Document(page_content="No help here", metadata={"source": "x.md", "chunk_id": "0"})
    graph = build_graph(
        FakeLLM([Grade(verdict="irrelevant")] * 3),
        FakeVectorStore([[document], [document], [document]]),
        Settings(),
    )

    result = graph.invoke(_initial_state())

    assert result["status"] == "no_answer"
    assert result["sources"] == []
    assert result["retry_count"] == 2
    assert result["grounding_check"] == "not_run"


def test_graph_rejects_answer_not_supported_by_context():
    document = Document(
        page_content="Use Depends to declare a dependency.",
        metadata={"source": "fastapi.md", "section": "Dependencies", "chunk_id": "0-test"},
    )
    graph = build_graph(
        FakeLLM([Grade(verdict="relevant")], grounding_verdict="unsupported"),
        FakeVectorStore([[document]]),
        Settings(),
    )

    result = graph.invoke(_initial_state())

    assert result["status"] == "no_answer"
    assert result["grounding_check"] == "failed"
    assert result["sources"] == []
    assert "couldn't verify" in result["answer"]
