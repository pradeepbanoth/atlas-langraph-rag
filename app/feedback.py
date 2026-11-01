import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import UUID


class FeedbackStore:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS queries (
                    query_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    query_id TEXT PRIMARY KEY REFERENCES queries(query_id),
                    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                    comment TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def save_query(self, query_id: UUID, question: str, answer: str, status: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO queries (query_id, question, answer, status) VALUES (?, ?, ?, ?)",
                (str(query_id), question, answer, status),
            )

    def save_feedback(self, query_id: UUID, rating: str, comment: str | None) -> bool:
        with closing(self._connect()) as connection, connection:
            exists = connection.execute(
                "SELECT 1 FROM queries WHERE query_id = ?", (str(query_id),)
            ).fetchone()
            if exists is None:
                return False
            connection.execute(
                """
                INSERT INTO feedback (query_id, rating, comment) VALUES (?, ?, ?)
                ON CONFLICT(query_id) DO UPDATE SET
                    rating = excluded.rating,
                    comment = excluded.comment,
                    created_at = CURRENT_TIMESTAMP
                """,
                (str(query_id), rating, comment),
            )
        return True
