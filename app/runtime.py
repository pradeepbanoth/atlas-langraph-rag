from pathlib import Path
from typing import NamedTuple, Protocol

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import Settings, get_settings
from app.graph import GraphState, build_graph
from app.ingest import ingest_local_docs


class RunnableGraph(Protocol):
    def invoke(self, state: GraphState) -> GraphState: ...


class Runtime(NamedTuple):
    graph: RunnableGraph
    vectorstore: Chroma
    settings: Settings


def build_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or get_settings()
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is required. Copy .env.example to .env and configure your Gemini key."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )
    docs_dir = Path(settings.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    ingest_local_docs(sorted(docs_dir.iterdir()), vectorstore, settings)

    llm = ChatGoogleGenerativeAI(
        model=settings.google_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )
    return Runtime(build_graph(llm, vectorstore, settings), vectorstore, settings)
