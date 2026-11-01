from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class Source(BaseModel):
    citation: int
    file: str
    section: str
    chunk_id: str
    snippet: str


class QueryResponse(BaseModel):
    query_id: UUID
    answer: str
    sources: list[Source]
    status: Literal["answered", "no_answer"]
    retries_used: int
    grounding_check: Literal["passed", "failed", "not_run"]


class IngestUrlsRequest(BaseModel):
    urls: list[Annotated[str, Field(min_length=1, max_length=2048)]] = Field(
        min_length=1,
        max_length=10,
    )


class IngestResponse(BaseModel):
    files_added: list[str]
    chunks_added: int


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class FeedbackRequest(BaseModel):
    query_id: UUID
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    accepted: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
