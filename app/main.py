import json
import ipaddress
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile
from starlette.responses import FileResponse

from app.feedback import FeedbackStore
from app.ingest import add_text
from app.runtime import Runtime, build_runtime
from app.schemas import (
    DocumentsResponse,
    DocumentInfo,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestResponse,
    IngestUrlsRequest,
    QueryRequest,
    QueryResponse,
)

SUPPORTED_CONTENT_TYPES = {
    "text/html",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = build_runtime()
    app.state.runtime = runtime
    app.state.feedback = FeedbackStore(runtime.settings.feedback_db)
    yield


app = FastAPI(
    title="Self-corrective RAG assistant",
    description="A technical-docs question-answering API with graded retrieval and bounded retries.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


def get_runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Application runtime is not initialized")
    return runtime


def get_feedback_store(request: Request) -> FeedbackStore:
    store = getattr(request.app.state, "feedback", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Feedback store is not initialized")
    return store


def _validate_public_https_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise ValueError("only absolute HTTPS URLs are accepted")
        if parsed.username or parsed.password:
            raise ValueError("URLs containing credentials are not accepted")
        host = parsed.hostname
        addresses = {
            item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        }
        if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise ValueError("URL host must resolve only to public IP addresses")
    except (ValueError, OSError) as exc:
        raise ValueError(f"invalid or unsafe URL: {exc}") from exc

    source = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return source


async def _fetch_url(url: str, max_bytes: int) -> tuple[str, str]:
    try:
        source = _validate_public_https_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "technical-docs-rag/1.0"},
        ) as client:
            async with client.stream("GET", url) as response:
                if response.is_redirect:
                    raise HTTPException(status_code=400, detail="redirected URLs are not accepted")
                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=400,
                        detail=f"URL returned HTTP {response.status_code}",
                    )
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if content_type not in SUPPORTED_CONTENT_TYPES:
                    raise HTTPException(
                        status_code=415,
                        detail=f"unsupported URL content type: {content_type or 'unknown'}",
                    )
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > max_bytes:
                        raise HTTPException(status_code=413, detail="URL document exceeds size limit")
                encoding = response.encoding or "utf-8"
                text = bytes(chunks).decode(encoding, errors="replace")
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"could not fetch URL: {exc}") from exc

    if content_type == "text/html":
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()
        plain_text = soup.get_text("\n", strip=True)
        text = f"# {title}\n\n{plain_text}" if title else plain_text
    return source, text


async def _parse_multipart(form: FormData, max_bytes: int) -> list[tuple[str, str]]:
    documents = []
    for key, value in form.multi_items():
        if key != "files" or not isinstance(value, UploadFile):
            continue
        filename = (value.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename.lower().endswith((".md", ".txt")):
            raise HTTPException(status_code=415, detail=f"unsupported file type: {filename}")
        raw = await value.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise HTTPException(status_code=413, detail=f"{filename} exceeds the upload size limit")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"{filename} must be UTF-8 text") from exc
        documents.append((filename, text))
    if not documents:
        raise HTTPException(status_code=422, detail="provide at least one file")
    return documents


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    runtime = get_runtime(request)
    query_id = uuid4()
    result = runtime.graph.invoke(
        {
            "question": body.question,
            "rewritten_query": "",
            "query_type": "",
            "documents": [],
            "relevant_docs": [],
            "retry_count": 0,
            "max_retries": runtime.settings.max_retries,
            "answer": "",
            "sources": [],
            "status": "",
            "grounding_check": "not_run",
        }
    )
    response = QueryResponse(
        query_id=query_id,
        answer=result["answer"],
        sources=result["sources"],
        status=result["status"],
        retries_used=result["retry_count"],
        grounding_check=result["grounding_check"],
    )
    get_feedback_store(request).save_query(
        query_id, body.question, response.answer, response.status
    )
    return response


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request) -> IngestResponse:
    runtime = get_runtime(request)
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type == "application/json":
        try:
            body = IngestUrlsRequest.model_validate(await request.json())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=422, detail="request body must be valid JSON") from exc
        documents = [
            await _fetch_url(url, runtime.settings.max_upload_bytes) for url in body.urls
        ]
    elif content_type == "multipart/form-data":
        form = await request.form()
        documents = await _parse_multipart(form, runtime.settings.max_upload_bytes)
    else:
        raise HTTPException(
            status_code=415,
            detail="use application/json with URLs or multipart/form-data with Markdown/text files",
        )

    files_added = []
    chunks_added = 0
    for source, content in documents:
        try:
            chunks_added += add_text(
                content,
                source,
                runtime.vectorstore,
                runtime.settings,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"{source}: {exc}") from exc
        files_added.append(source)
    return IngestResponse(files_added=files_added, chunks_added=chunks_added)


@app.get("/documents", response_model=DocumentsResponse)
def documents(request: Request) -> DocumentsResponse:
    runtime = get_runtime(request)
    stored = runtime.vectorstore.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for metadata in stored["metadatas"] or []:
        source = metadata.get("source")
        if source:
            counts[source] = counts.get(source, 0) + 1
    return DocumentsResponse(
        documents=[
            DocumentInfo(source=source, chunk_count=count)
            for source, count in sorted(counts.items())
        ]
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    if not get_feedback_store(request).save_feedback(body.query_id, body.rating, body.comment):
        raise HTTPException(status_code=404, detail="unknown query_id")
    return FeedbackResponse(accepted=True)
