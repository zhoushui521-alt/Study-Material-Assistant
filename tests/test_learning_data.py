import tempfile
import unittest
from pathlib import Path

from app.learning_data import (
    LEARNING_SCHEMA_VERSION,
    LearningDataNotFoundError,
    LearningDataStore,
)


class LearningDataStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "learning.sqlite3"
        )
        self.store = await LearningDataStore.open(self.database_path)

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self.temporary_directory.cleanup()

    async def test_create_and_query_user_with_versioned_schema(self) -> None:
        created = await self.store.create_user()

        loaded = await self.store.get_user(created.user_id)

        self.assertEqual(loaded, created)
        self.assertEqual(
            await self.store.schema_version(),
            LEARNING_SCHEMA_VERSION,
        )

    async def test_create_session_and_enforce_user_ownership(self) -> None:
        first_user = await self.store.create_user()
        second_user = await self.store.create_user()
        session = await self.store.create_session(
            first_user.user_id,
            "Embedding",
        )

        first_sessions = await self.store.list_sessions(first_user.user_id)
        second_sessions = await self.store.list_sessions(second_user.user_id)

        self.assertEqual(first_sessions, (session,))
        self.assertEqual(second_sessions, ())
        with self.assertRaises(LearningDataNotFoundError):
            await self.store.get_session(second_user.user_id, session.session_id)

    async def test_save_and_query_conversation_and_learning_record(self) -> None:
        user = await self.store.create_user()
        other_user = await self.store.create_user()
        session = await self.store.create_session(user.user_id, "RAG")
        await self.store.record_tutor_exchange(
            user_id=user.user_id,
            session_id=session.session_id,
            topic="Embedding",
            intent="quiz",
            user_content="帮我出题",
            tutor_content="练习：Embedding 的作用是什么？",
            activity_type="practice_quiz",
            metadata={"tools_used": ["knowledge_retrieval", "quiz_generator"]},
        )

        messages = await self.store.list_messages(
            user.user_id,
            session.session_id,
        )
        self.assertTrue(all(item.user_id == user.user_id for item in messages))
        records = await self.store.list_learning_records(user.user_id)
        updated_session = await self.store.get_session(
            user.user_id,
            session.session_id,
        )

        self.assertEqual([item.role for item in messages], ["user", "tutor"])
        self.assertEqual(messages[0].content, "帮我出题")
        self.assertEqual(records[0].activity_type, "practice_quiz")
        self.assertEqual(
            records[0].metadata["tools_used"],
            ["knowledge_retrieval", "quiz_generator"],
        )
        self.assertEqual(updated_session.topic, "Embedding")
        self.assertEqual(await self.store.list_user_messages(other_user.user_id), ())
        self.assertEqual(
            await self.store.list_learning_records(other_user.user_id),
            (),
        )
        with self.assertRaises(LearningDataNotFoundError):
            await self.store.list_messages(
                other_user.user_id,
                session.session_id,
            )

    async def test_data_and_migration_survive_reopen(self) -> None:
        user = await self.store.create_user()
        session = await self.store.create_session(user.user_id, "LangGraph")
        await self.store.record_tutor_exchange(
            user_id=user.user_id,
            session_id=session.session_id,
            topic="LangGraph",
            intent="summary",
            user_content="总结本次学习",
            tutor_content="本次学习了持久化。",
            activity_type="summarize_learning",
        )
        await self.store.close()

        self.store = await LearningDataStore.open(self.database_path)
        messages = await self.store.list_messages(user.user_id, session.session_id)
        records = await self.store.list_learning_records(
            user.user_id,
            session_id=session.session_id,
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(len(records), 1)
        self.assertEqual(await self.store.schema_version(), LEARNING_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
