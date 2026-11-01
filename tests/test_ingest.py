from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.ingest import add_text, split_text


class FakeVectorStore:
    def __init__(self):
        self.documents = {}

    def get(self, where=None):
        source = where["source"]
        matches = {
            chunk_id: document
            for chunk_id, document in self.documents.items()
            if document.metadata["source"] == source
        }
        return {"ids": list(matches)}

    def delete(self, ids):
        for chunk_id in ids:
            self.documents.pop(chunk_id, None)

    def add_documents(self, documents, ids):
        self.documents.update(zip(ids, documents))
        return ids


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        seed = sum((index + 1) * ord(char) for index, char in enumerate(text))
        return [float((seed + index * 17) % 97 + 1) for index in range(8)]


def test_split_text_includes_source_section_and_stable_chunk_ids():
    settings = Settings(chunk_size=120, chunk_overlap=20)
    content = "# Guide\n\n## Setup\n\nInstall the package and configure the application."

    chunks = split_text(content, "guide.md", settings)

    assert chunks
    assert all(chunk.metadata["source"] == "guide.md" for chunk in chunks)
    assert all(chunk.metadata["section"] == "Guide / Setup" for chunk in chunks)
    assert all(chunk.metadata["chunk_id"] for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        chunk.metadata["chunk_id"] for chunk in split_text(content, "guide.md", settings)
    ]


def test_add_text_replaces_outdated_chunks_without_duplicates():
    store = FakeVectorStore()
    settings = Settings(chunk_size=120, chunk_overlap=20)

    first_count = add_text("# Guide\n\n## Setup\n\nInstall version one.", "guide.md", store, settings)
    same_count = add_text("# Guide\n\n## Setup\n\nInstall version one.", "guide.md", store, settings)
    updated_count = add_text("# Guide\n\n## Setup\n\nInstall version two.", "guide.md", store, settings)

    assert first_count == same_count == updated_count == 1
    assert len(store.documents) == 1
    assert "version two" in next(iter(store.documents.values())).page_content


def test_add_text_works_with_persistent_chroma(tmp_path):
    store = Chroma(
        collection_name="test_docs",
        embedding_function=TinyEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    settings = Settings(chunk_size=120, chunk_overlap=20)

    first_count = add_text("# Guide\n\n## Setup\n\nInstall version one.", "guide.md", store, settings)
    updated_count = add_text("# Guide\n\n## Setup\n\nInstall version two.", "guide.md", store, settings)
    reopened_store = Chroma(
        collection_name="test_docs",
        embedding_function=TinyEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    stored = reopened_store.get(include=["metadatas", "documents"])

    assert first_count == updated_count == 1
    assert len(stored["ids"]) == 1
    assert stored["metadatas"][0]["source"] == "guide.md"
    assert "version two" in stored["documents"][0]
