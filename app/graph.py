from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.config import Settings
from app.vector_store import VectorStore


class GraphState(TypedDict):
    question: str
    rewritten_query: str
    query_type: str
    documents: list[Document]
    relevant_docs: list[Document]
    retry_count: int
    max_retries: int
    answer: str
    sources: list[dict[str, str | int]]
    status: str
    grounding_check: Literal["passed", "failed", "not_run"]


class Analysis(BaseModel):
    rewritten_query: str
    query_type: Literal["conceptual", "how-to", "troubleshooting", "api-reference"]


class Grade(BaseModel):
    verdict: Literal["relevant", "irrelevant"]


class GroundingGrade(BaseModel):
    verdict: Literal["supported", "unsupported"]


def build_graph(llm: BaseChatModel, vectorstore: VectorStore, settings: Settings):
    def analyze(state: GraphState) -> dict[str, object]:
        result = llm.with_structured_output(Analysis).invoke(
            "Rewrite the question for semantic search over technical documentation. "
            "Expand abbreviations and include useful technical terms. Preserve the user's intent.\n"
            f"Question: {state['question']}"
        )
        return {
            "rewritten_query": result.rewritten_query,
            "query_type": result.query_type,
            "retry_count": 0,
        }

    def retrieve(state: GraphState) -> dict[str, object]:
        documents = vectorstore.similarity_search(state["rewritten_query"], k=settings.retrieval_k)
        return {"documents": documents}

    def grade(state: GraphState) -> dict[str, object]:
        grader = llm.with_structured_output(Grade)
        relevant = []
        for document in state["documents"]:
            result = grader.invoke(
                "Decide whether this document chunk contains information that helps answer "
                "the user's question. Mark it relevant only when it provides direct evidence.\n"
                f"Question: {state['question']}\nChunk:\n{document.page_content}"
            )
            if result.verdict == "relevant":
                relevant.append(document)
        return {"relevant_docs": relevant}

    def route(state: GraphState) -> str:
        if state["relevant_docs"]:
            return "generate"
        if state["retry_count"] < state["max_retries"]:
            return "rewrite"
        return "no_answer"

    def rewrite(state: GraphState) -> dict[str, object]:
        result = llm.invoke(
            "Write a concise, meaningfully different search query for the user's original "
            "question. Return only the query, without quotation marks or explanation.\n"
            f"Previous search query: {state['rewritten_query']}\n"
            f"Original question: {state['question']}"
        )
        query = result.content
        if not isinstance(query, str) or not query.strip():
            raise ValueError("the query rewrite model returned an empty query")
        return {
            "rewritten_query": query.strip(),
            "retry_count": state["retry_count"] + 1,
        }

    def generate(state: GraphState) -> dict[str, object]:
        context_parts = []
        sources = []
        for index, document in enumerate(state["relevant_docs"], start=1):
            source = str(document.metadata["source"])
            section = str(document.metadata.get("section", "Document"))
            chunk_id = str(document.metadata["chunk_id"])
            context_parts.append(
                f"[{index}] ({source}, {section})\n{document.page_content}"
            )
            sources.append(
                {
                    "citation": index,
                    "file": source,
                    "section": section,
                    "chunk_id": chunk_id,
                    "snippet": document.page_content[:200],
                }
            )

        context = "\n\n".join(context_parts)
        result = llm.invoke(
            "Answer the question using only the supplied context. Cite factual claims with "
            "the matching [n] references. If the context does not support an answer, say so "
            "rather than guessing.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}"
        )
        if not isinstance(result.content, str) or not result.content.strip():
            raise ValueError("the answer model returned an empty response")
        return {
            "answer": result.content.strip(),
            "sources": sources,
            "status": "answered",
        }

    def no_answer(_: GraphState) -> dict[str, object]:
        return {
            "answer": "I couldn't find this in the indexed documents.",
            "sources": [],
            "status": "no_answer",
        }

    def verify(state: GraphState) -> dict[str, object]:
        context = "\n\n".join(
            f"[{index}] {document.page_content}"
            for index, document in enumerate(state["relevant_docs"], start=1)
        )
        result = llm.with_structured_output(GroundingGrade).invoke(
            "Check whether every factual claim in the proposed answer is supported by "
            "the supplied document context. Citations must refer to the supplied context. "
            "Return supported only if all claims are grounded; otherwise return unsupported.\n"
            f"Question: {state['question']}\n"
            f"Context:\n{context}\n"
            f"Proposed answer:\n{state['answer']}"
        )
        if result.verdict == "supported":
            return {"grounding_check": "passed"}
        return {
            "answer": "I couldn't verify a sufficiently grounded answer from the indexed documents.",
            "sources": [],
            "status": "no_answer",
            "grounding_check": "failed",
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("analyze", analyze)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("verify", verify)
    workflow.add_node("no_answer", no_answer)
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        route,
        {"generate": "generate", "rewrite": "rewrite", "no_answer": "no_answer"},
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", END)
    workflow.add_edge("no_answer", END)
    return workflow.compile()
