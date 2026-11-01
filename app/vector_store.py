from typing import Any, Protocol

from langchain_core.documents import Document


class VectorStore(Protocol):
    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]: ...

    def delete(self, ids: list[str]) -> Any: ...

    def get(
        self,
        where: dict[str, str] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def similarity_search(self, query: str, k: int = 4) -> list[Document]: ...
