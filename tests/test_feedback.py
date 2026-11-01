from uuid import uuid4

from app.feedback import FeedbackStore


def test_feedback_requires_known_query_and_persists(tmp_path):
    store = FeedbackStore(str(tmp_path / "feedback.sqlite3"))
    query_id = uuid4()

    assert store.save_feedback(query_id, "up", None) is False
    store.save_query(query_id, "question", "answer", "answered")
    assert store.save_feedback(query_id, "up", "helpful") is True
    assert store.save_feedback(query_id, "down", "updated") is True
