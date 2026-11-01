# Self-corrective RAG assistant

A FastAPI service that answers technical questions from a small document
collection. LangGraph analyzes the question, retrieves ChromaDB chunks, grades
each chunk, and either answers with numbered citations, rewrites and retries,
or returns an explicit no-answer response.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key with access to the configured chat and embedding models
- Internet access for the first dependency install and model API calls

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GOOGLE_API_KEY` in `.env` using a key from Google AI Studio. The included
3-document corpus is indexed automatically at startup. ChromaDB and feedback
are stored locally under `data/`.
Do not commit `.env` or API keys.

## Run

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the RAG web interface. Use the knowledge-base
panel to select and index Markdown or text documents, then ask questions in the
assistant panel and review the cited sources. The API documentation remains
available at `http://127.0.0.1:8000/docs`.

## API examples

Query the included FastAPI notes:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/query `
  -ContentType 'application/json' `
  -Body '{"question":"How do I declare a dependency in FastAPI?"}'
```

The response contains `query_id`, `answer`, `sources`, `status`, `retries_used`,
and `grounding_check` (`passed`, `failed`, or `not_run`). Successful answers cite
context as `[1]`, `[2]`, etc., and the
`sources` list maps each citation to a file, section, chunk ID, and snippet.
When no evidence passes grading within the retry limit, the API returns
`status: "no_answer"` and an empty source list.

Illustrative response shape (the generated wording can vary):

```json
{
  "query_id": "7ce925ce-1390-4ed8-b225-4c819710e50a",
  "answer": "Declare the dependency as a callable parameter using Depends. [1]",
  "sources": [
    {
      "citation": 1,
      "file": "fastapi.md",
      "section": "FastAPI / Dependencies",
      "chunk_id": "0-0123456789ab",
      "snippet": "FastAPI dependencies are callables declared with Depends..."
    }
  ],
  "status": "answered",
  "retries_used": 0,
  "grounding_check": "passed"
}
```

Upload Markdown or UTF-8 text files (maximum 2 MB per file):

```powershell
curl.exe -X POST http://127.0.0.1:8000/ingest `
  -F "files=@./notes.md"
```

Ingest public HTTPS documentation URLs:

```powershell
curl.exe -X POST http://127.0.0.1:8000/ingest `
  -H "Content-Type: application/json" `
  -d '{"urls":["https://example.org/guide.md"]}'
```

URL ingestion accepts Markdown, plain text, or HTML, rejects redirects and
non-public hosts, and enforces the same 2 MB limit. `GET /documents` reports the
indexed sources and chunk counts. Send feedback for a query:

```powershell
curl.exe -X POST http://127.0.0.1:8000/feedback `
  -H "Content-Type: application/json" `
  -d '{"query_id":"<query_id-from-query-response>","rating":"up","comment":"Helpful"}'
```

`GET /health` checks that the API process is responsive. Invalid requests return
validation errors; unknown feedback query IDs return 404; model/provider failures
are surfaced as server errors rather than fabricated answers.

## Architecture

```text
POST /query
  -> analyze original question
  -> retrieve the top 2 Chroma chunks by default
  -> grade each chunk against the original question
       -> relevant chunks: generate grounded answer and citations
       -> verify every answer claim against the retrieved context
       -> no relevant chunks: rewrite query -> retrieve again (at most 2 retries)
       -> retry limit reached: explicit no-answer response
```

The graph stores the original question separately from the evolving retrieval
query. The retry counter is part of graph state, so the loop is bounded and the
number of retries is returned to the API caller.

Ingestion first splits Markdown at headings, then applies a recursive
character-based splitter. The default chunk size is 800 characters with 120
characters of overlap. Each chunk stores its source, heading path, and stable
chunk ID. The same Gemini embedding model is used for indexing and search. A
separate default Chroma collection (`technical_docs_gemini`) prevents vectors
created with a previous provider from being mixed with Gemini vectors.
After generation, a structured-output verification node checks the answer's
claims against the retrieved passages. Unsupported answers are replaced by an
explicit no-answer response. Two retrieved chunks limit ordinary successful
queries to five Gemini generation requests: analysis, two grading decisions,
answer generation, and verification.

## Design choices and tradeoffs

- Assumptions: documents are UTF-8 Markdown or plain text; each query is
  independent rather than part of a persistent chat session; and only claims
  supported by retrieved passages should be presented as an answer.
- Each retrieved chunk receives its own structured grading call. This keeps
  relevance decisions explicit but adds latency and model cost.
- A grounding-verification node checks the generated answer against retrieved
  passages. Unsupported answers are replaced with a no-answer response; the
  extra model call improves reliability but adds latency and provider usage.
- Two retries bound model calls while giving query rewrites a chance to improve
  retrieval.
- Gemini hosted embeddings and chat keep the runtime setup simple; they require
  network access and incur provider usage costs.
- Feedback is persisted in SQLite and associated with a query UUID. A later
  feedback submission for the same query updates its existing rating.
- URL ingestion accepts public HTTPS hosts only, rejects redirects, and limits
  input size. This avoids common unsafe URL-fetching behavior.

## Tests

Install dependencies, then run:

```powershell
python -m pytest
```

Graph tests use fake models and a fake vector store, so they do not need API
credentials. They cover citation mapping, successful retry, and the bounded
no-answer path, including rejection when the generated answer is not supported
by the retrieved context.

## Future improvements

Hybrid BM25/vector retrieval, reranking, a small evaluation set, streaming,
parallel grading, authentication, and conversation memory would be useful next
steps. Production deployment should also add request-level rate limits,
observability, and stronger resource controls.
