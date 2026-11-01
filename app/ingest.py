from hashlib import sha256
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import Settings
from app.vector_store import VectorStore


def split_text(content: str, source: str, settings: Settings) -> list[Document]:
    if not content.strip():
        raise ValueError("document content must not be empty")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=min(settings.chunk_overlap, settings.chunk_size - 1),
        separators=["\n```", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(sections)

    documents: list[Document] = []
    for index, chunk in enumerate(chunks):
        section = " / ".join(
            chunk.metadata[key] for key in ("h1", "h2", "h3") if chunk.metadata.get(key)
        )
        chunk.metadata.update(
            source=source,
            section=section or "Document",
            chunk_id=f"{index}-{sha256(chunk.page_content.encode('utf-8')).hexdigest()[:12]}",
        )
        documents.append(chunk)
    return documents


def add_text(
    content: str,
    source: str,
    vectorstore: VectorStore,
    settings: Settings,
) -> int:
    documents = split_text(content, source, settings)
    existing = vectorstore.get(where={"source": source})
    ids = [f"{source}:{document.metadata['chunk_id']}" for document in documents]
    if set(existing["ids"]) == set(ids):
        return len(documents)
    if existing["ids"]:
        vectorstore.delete(ids=existing["ids"])
    vectorstore.add_documents(
        documents=documents,
        ids=ids,
    )
    return len(documents)


def ingest_local_docs(
    paths: Iterable[Path],
    vectorstore: VectorStore,
    settings: Settings,
) -> None:
    for path in paths:
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            add_text(path.read_text(encoding="utf-8"), path.name, vectorstore, settings)
