"""Authenticated API test fixtures with no password or network dependency."""

from app.api import app, get_current_user
from app.learning_data import UserRecord


TEST_USER = UserRecord(
    user_id="22222222-2222-4222-8222-222222222222",
    email="learner@example.com",
    password_hash="test-only-not-a-real-password-hash",
    display_name="测试学习者",
    updated_at="2026-08-22T00:00:00.000Z",
    created_at="2026-08-22T00:00:00.000Z",
)


def install_authenticated_user() -> None:
    app.dependency_overrides[get_current_user] = lambda: TEST_USER
    app.state.rag_services.clear()
    app.state.agent_services.clear()
    app.state.tutor_services.clear()
    app.state.material_managers.clear()


def clear_user_services() -> None:
    app.state.rag_services.clear()
    app.state.agent_services.clear()
    app.state.material_managers.clear()
