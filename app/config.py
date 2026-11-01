from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str | None = None
    google_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    chroma_dir: str = "data/chroma"
    collection_name: str = "technical_docs_gemini"
    docs_dir: str = "data/docs"
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    retrieval_k: int = Field(default=2, ge=1, le=20)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_upload_bytes: int = Field(default=2_000_000, ge=1024)
    feedback_db: str = "data/feedback.sqlite3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
