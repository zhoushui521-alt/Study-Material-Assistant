"""Stage 5.1：本地用户身份、学习会话、对话与学习记录持久化。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


LEARNING_DB_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "learning" / "learning.sqlite3"
)
LEARNING_SCHEMA_VERSION = 2
MAX_SESSION_TOPIC_LENGTH = 200
MAX_HISTORY_ITEMS = 100


class LearningDataError(RuntimeError):
    """学习数据无法按持久化契约完成读写。"""


class LearningDataNotFoundError(LearningDataError):
    """用户或学习会话不存在，或不属于当前用户。"""


class LearningDataConflictError(LearningDataError):
    """学习数据违反唯一性或当前状态约束。"""


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    email: str | None
    password_hash: str | None
    display_name: str | None
    updated_at: str | None
    created_at: str


@dataclass(frozen=True)
class LearningSessionRecord:
    session_id: str
    user_id: str
    topic: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ConversationMessageRecord:
    message_id: str
    session_id: str
    user_id: str
    role: str
    content: str
    intent: str | None
    created_at: str


@dataclass(frozen=True)
class LearningActivityRecord:
    record_id: str
    user_id: str
    session_id: str
    topic: str
    activity_type: str
    created_at: str
    metadata: dict[str, Any]


MIGRATION_1 = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS learning_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    updated_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_learning_sessions_user_updated
    ON learning_sessions(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_order INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'tutor')),
    content TEXT NOT NULL,
    intent TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    FOREIGN KEY (session_id) REFERENCES learning_sessions(session_id)
        ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_messages_session_order
    ON conversation_messages(session_id, message_order);

CREATE TABLE IF NOT EXISTS learning_records (
    record_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES learning_sessions(session_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_learning_records_user_created
    ON learning_records(user_id, created_at DESC, record_id DESC);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
COMMIT;
"""

MIGRATION_2 = """
BEGIN IMMEDIATE;

ALTER TABLE users ADD COLUMN email TEXT;
ALTER TABLE users ADD COLUMN password_hash TEXT;
ALTER TABLE users ADD COLUMN display_name TEXT;
ALTER TABLE users ADD COLUMN updated_at TEXT;

UPDATE users SET updated_at = created_at WHERE updated_at IS NULL;

CREATE UNIQUE INDEX idx_users_email_normalized
    ON users(lower(email))
    WHERE email IS NOT NULL;

CREATE UNIQUE INDEX idx_learning_sessions_session_user
    ON learning_sessions(session_id, user_id);

ALTER TABLE conversation_messages RENAME TO conversation_messages_v1;
DROP INDEX idx_conversation_messages_session_order;

CREATE TABLE conversation_messages (
    message_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_order INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'tutor')),
    content TEXT NOT NULL,
    intent TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id, user_id)
        REFERENCES learning_sessions(session_id, user_id) ON DELETE CASCADE
);

INSERT INTO conversation_messages(
    message_id, user_id, session_id, message_order,
    role, content, intent, created_at
)
SELECT messages.message_id, sessions.user_id, messages.session_id,
       messages.message_order, messages.role, messages.content,
       messages.intent, messages.created_at
FROM conversation_messages_v1 AS messages
INNER JOIN learning_sessions AS sessions
    ON sessions.session_id = messages.session_id;

DROP TABLE conversation_messages_v1;

CREATE UNIQUE INDEX idx_conversation_messages_session_order
    ON conversation_messages(session_id, message_order);
CREATE INDEX idx_conversation_messages_user_created
    ON conversation_messages(user_id, created_at, message_id);

ALTER TABLE learning_records RENAME TO learning_records_v1;
DROP INDEX idx_learning_records_user_created;

CREATE TABLE learning_records (
    record_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id, user_id)
        REFERENCES learning_sessions(session_id, user_id) ON DELETE CASCADE
);

INSERT INTO learning_records(
    record_id, user_id, session_id, topic,
    activity_type, metadata_json, created_at
)
SELECT record_id, user_id, session_id, topic,
       activity_type, metadata_json, created_at
FROM learning_records_v1;

DROP TABLE learning_records_v1;

CREATE INDEX idx_learning_records_user_created
    ON learning_records(user_id, created_at DESC, record_id DESC);

CREATE TABLE auth_sessions (
    session_token_hash TEXT PRIMARY KEY CHECK (length(session_token_hash) = 64),
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_auth_sessions_user_created
    ON auth_sessions(user_id, created_at DESC);
CREATE INDEX idx_auth_sessions_expires
    ON auth_sessions(expires_at);

INSERT INTO schema_migrations(version) VALUES (2);
COMMIT;
"""

def _canonical_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("ID 必须是有效 UUID。") from error


def _bounded_limit(limit: int) -> int:
    if limit < 1 or limit > MAX_HISTORY_ITEMS:
        raise ValueError(f"limit 必须介于 1 和 {MAX_HISTORY_ITEMS} 之间。")
    return limit


def _normalize_topic(topic: str) -> str:
    normalized = topic.strip()
    if not normalized or len(normalized) > MAX_SESSION_TOPIC_LENGTH:
        raise ValueError(
            f"学习主题不能为空且不能超过 {MAX_SESSION_TOPIC_LENGTH} 个字符。"
        )
    return normalized


class LearningDataStore:
    """显式 SQL 数据层；业务记录与 LangGraph checkpoint 共用数据库文件。"""

    def __init__(
        self,
        *,
        connection: aiosqlite.Connection,
        checkpoint_connection: aiosqlite.Connection,
        checkpointer: AsyncSqliteSaver,
    ) -> None:
        self._connection = connection
        self._checkpoint_connection = checkpoint_connection
        self.checkpointer = checkpointer
        self._write_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def open(
        cls,
        database_path: Path = LEARNING_DB_PATH,
    ) -> "LearningDataStore":
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(database_path)
        checkpoint_connection: aiosqlite.Connection | None = None
        try:
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.execute("PRAGMA journal_mode = WAL")
            await cls._apply_migrations(connection)

            checkpoint_connection = await aiosqlite.connect(database_path)
            await checkpoint_connection.execute("PRAGMA busy_timeout = 5000")
            await checkpoint_connection.execute("PRAGMA journal_mode = WAL")
            serializer = JsonPlusSerializer(
                pickle_fallback=False,
                allowed_json_modules=(),
                allowed_msgpack_modules=(),
            )
            checkpointer = AsyncSqliteSaver(
                checkpoint_connection,
                serde=serializer,
            )
            await checkpointer.setup()
            return cls(
                connection=connection,
                checkpoint_connection=checkpoint_connection,
                checkpointer=checkpointer,
            )
        except Exception:
            try:
                if checkpoint_connection is not None:
                    await checkpoint_connection.close()
            finally:
                await connection.close()
            raise

    @staticmethod
    async def _apply_migrations(connection: aiosqlite.Connection) -> None:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            )
            """
        )
        cursor = await connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        )
        row = await cursor.fetchone()
        current_version = int(row[0]) if row is not None else 0
        if current_version > LEARNING_SCHEMA_VERSION:
            raise LearningDataError("学习数据库版本高于当前程序支持范围。")
        if current_version < 1:
            try:
                await connection.executescript(MIGRATION_1)
            except Exception:
                try:
                    await connection.rollback()
                except Exception:
                    pass
                raise
        if current_version < 2:
            try:
                await connection.executescript(MIGRATION_2)
            except Exception:
                try:
                    await connection.rollback()
                except Exception:
                    pass
                raise
        await connection.commit()

    async def schema_version(self) -> int:
        cursor = await self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        )
        row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def create_user(
        self,
        *,
        email: str | None = None,
        password_hash: str | None = None,
        display_name: str | None = None,
    ) -> UserRecord:
        user_id = str(uuid.uuid4())
        async with self._write_lock:
            try:
                await self._connection.execute(
                    """
                    INSERT INTO users(
                        user_id, email, password_hash, display_name, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    )
                    """,
                    (user_id, email, password_hash, display_name),
                )
                await self._connection.commit()
            except sqlite3.IntegrityError as error:
                await self._connection.rollback()
                raise LearningDataConflictError("用户身份已经存在。") from error
            except Exception as error:
                await self._connection.rollback()
                raise LearningDataError("用户创建失败。") from error
        return await self.get_user(user_id)

    async def get_user(self, user_id: str) -> UserRecord:
        normalized_user_id = _canonical_uuid(user_id)
        cursor = await self._connection.execute(
            """
            SELECT user_id, email, password_hash, display_name,
                   updated_at, created_at
            FROM users WHERE user_id = ?
            """,
            (normalized_user_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise LearningDataNotFoundError("用户不存在。")
        return self._user_record(row)

    async def get_user_by_email(self, email: str) -> UserRecord:
        cursor = await self._connection.execute(
            """
            SELECT user_id, email, password_hash, display_name,
                   updated_at, created_at
            FROM users
            WHERE lower(email) = lower(?)
            """,
            (email,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise LearningDataNotFoundError("用户不存在。")
        return self._user_record(row)

    @staticmethod
    def _user_record(row: aiosqlite.Row) -> UserRecord:
        return UserRecord(
            user_id=row["user_id"],
            email=row["email"],
            password_hash=row["password_hash"],
            display_name=row["display_name"],
            updated_at=row["updated_at"],
            created_at=row["created_at"],
        )

    async def create_auth_session(
        self,
        user_id: str,
        token_hash: str,
        expires_at: str,
    ) -> None:
        normalized_user_id = _canonical_uuid(user_id)
        async with self._write_lock:
            try:
                await self._connection.execute(
                    """
                    INSERT INTO auth_sessions(
                        session_token_hash, user_id, expires_at
                    ) VALUES (?, ?, ?)
                    """,
                    (token_hash, normalized_user_id, expires_at),
                )
                await self._connection.commit()
            except Exception as error:
                await self._connection.rollback()
                raise LearningDataError("登录状态创建失败。") from error

    async def get_user_by_auth_session(
        self,
        token_hash: str,
        *,
        now: str,
    ) -> UserRecord:
        cursor = await self._connection.execute(
            """
            SELECT users.user_id, users.email, users.password_hash,
                   users.display_name, users.updated_at, users.created_at
            FROM auth_sessions
            INNER JOIN users ON users.user_id = auth_sessions.user_id
            WHERE auth_sessions.session_token_hash = ?
              AND auth_sessions.expires_at > ?
              AND users.email IS NOT NULL
              AND users.password_hash IS NOT NULL
            """,
            (token_hash, now),
        )
        row = await cursor.fetchone()
        if row is None:
            raise LearningDataNotFoundError("登录状态不存在。")
        return self._user_record(row)

    async def delete_auth_session(self, token_hash: str) -> None:
        async with self._write_lock:
            try:
                await self._connection.execute(
                    "DELETE FROM auth_sessions WHERE session_token_hash = ?",
                    (token_hash,),
                )
                await self._connection.commit()
            except Exception as error:
                await self._connection.rollback()
                raise LearningDataError("登录状态删除失败。") from error

    async def create_session(
        self,
        user_id: str,
        topic: str,
    ) -> LearningSessionRecord:
        normalized_user_id = _canonical_uuid(user_id)
        normalized_topic = _normalize_topic(topic)
        session_id = str(uuid.uuid4())
        async with self._write_lock:
            try:
                cursor = await self._connection.execute(
                    "SELECT 1 FROM users WHERE user_id = ?",
                    (normalized_user_id,),
                )
                if await cursor.fetchone() is None:
                    raise LearningDataNotFoundError("用户不存在。")
                await self._connection.execute(
                    """
                    INSERT INTO learning_sessions(session_id, user_id, topic)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, normalized_user_id, normalized_topic),
                )
                await self._connection.commit()
            except LearningDataNotFoundError:
                await self._connection.rollback()
                raise
            except Exception as error:
                await self._connection.rollback()
                raise LearningDataError("学习会话创建失败。") from error
        return await self.get_session(normalized_user_id, session_id)

    async def get_session(
        self,
        user_id: str,
        session_id: str,
    ) -> LearningSessionRecord:
        normalized_user_id = _canonical_uuid(user_id)
        normalized_session_id = _canonical_uuid(session_id)
        cursor = await self._connection.execute(
            """
            SELECT session_id, user_id, topic, created_at, updated_at
            FROM learning_sessions
            WHERE session_id = ? AND user_id = ?
            """,
            (normalized_session_id, normalized_user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise LearningDataNotFoundError("学习会话不存在。")
        return LearningSessionRecord(
            session_id=row["session_id"],
            user_id=row["user_id"],
            topic=row["topic"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def list_sessions(
        self,
        user_id: str,
        *,
        limit: int = MAX_HISTORY_ITEMS,
    ) -> tuple[LearningSessionRecord, ...]:
        normalized_user_id = _canonical_uuid(user_id)
        bounded_limit = _bounded_limit(limit)
        await self.get_user(normalized_user_id)
        cursor = await self._connection.execute(
            """
            SELECT session_id, user_id, topic, created_at, updated_at
            FROM learning_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC, session_id DESC
            LIMIT ?
            """,
            (normalized_user_id, bounded_limit),
        )
        rows = await cursor.fetchall()
        return tuple(
            LearningSessionRecord(
                session_id=row["session_id"],
                user_id=row["user_id"],
                topic=row["topic"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        )

    async def list_messages(
        self,
        user_id: str,
        session_id: str,
        *,
        limit: int = MAX_HISTORY_ITEMS,
    ) -> tuple[ConversationMessageRecord, ...]:
        session = await self.get_session(user_id, session_id)
        bounded_limit = _bounded_limit(limit)
        cursor = await self._connection.execute(
            """
            SELECT message_id, user_id, session_id, role, content, intent, created_at
            FROM (
                SELECT message_id, user_id, session_id, message_order, role, content,
                       intent, created_at
                FROM conversation_messages
                WHERE session_id = ? AND user_id = ?
                ORDER BY message_order DESC
                LIMIT ?
            )
            ORDER BY message_order
            """,
            (session.session_id, session.user_id, bounded_limit),
        )
        rows = await cursor.fetchall()
        return tuple(
            ConversationMessageRecord(
                message_id=row["message_id"],
                session_id=row["session_id"],
                user_id=row["user_id"],
                role=row["role"],
                content=row["content"],
                intent=row["intent"],
                created_at=row["created_at"],
            )
            for row in rows
        )

    async def list_user_messages(
        self,
        user_id: str,
        *,
        limit: int = MAX_HISTORY_ITEMS,
    ) -> tuple[ConversationMessageRecord, ...]:
        normalized_user_id = _canonical_uuid(user_id)
        bounded_limit = _bounded_limit(limit)
        await self.get_user(normalized_user_id)
        cursor = await self._connection.execute(
            """
            SELECT message_id, user_id, session_id, role, content, intent, created_at
            FROM (
                SELECT messages.message_id, messages.user_id, messages.session_id,
                       messages.message_order, messages.role, messages.content,
                       messages.intent, messages.created_at
                FROM conversation_messages AS messages
                INNER JOIN learning_sessions AS sessions
                    ON sessions.session_id = messages.session_id
                WHERE messages.user_id = ?
                ORDER BY messages.created_at DESC, messages.session_id DESC,
                         messages.message_order DESC
                LIMIT ?
            )
            ORDER BY created_at, session_id, message_order
            """,
            (normalized_user_id, bounded_limit),
        )
        rows = await cursor.fetchall()
        return tuple(
            ConversationMessageRecord(
                message_id=row["message_id"],
                session_id=row["session_id"],
                user_id=row["user_id"],
                role=row["role"],
                content=row["content"],
                intent=row["intent"],
                created_at=row["created_at"],
            )
            for row in rows
        )

    async def list_learning_records(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        limit: int = MAX_HISTORY_ITEMS,
    ) -> tuple[LearningActivityRecord, ...]:
        normalized_user_id = _canonical_uuid(user_id)
        bounded_limit = _bounded_limit(limit)
        parameters: list[Any] = [normalized_user_id]
        session_clause = ""
        if session_id is not None:
            session = await self.get_session(normalized_user_id, session_id)
            session_clause = " AND session_id = ?"
            parameters.append(session.session_id)
        else:
            await self.get_user(normalized_user_id)
        parameters.append(bounded_limit)
        cursor = await self._connection.execute(
            f"""
            SELECT record_id, user_id, session_id, topic, activity_type,
                   metadata_json, created_at
            FROM learning_records
            WHERE user_id = ?{session_clause}
            ORDER BY created_at DESC, record_id DESC
            LIMIT ?
            """,
            tuple(parameters),
        )
        rows = await cursor.fetchall()
        return tuple(
            LearningActivityRecord(
                record_id=row["record_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                topic=row["topic"],
                activity_type=row["activity_type"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        )

    async def record_tutor_exchange(
        self,
        *,
        user_id: str,
        session_id: str,
        topic: str,
        intent: str,
        user_content: str,
        tutor_content: str,
        activity_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_user_id = _canonical_uuid(user_id)
        normalized_session_id = _canonical_uuid(session_id)
        normalized_topic = _normalize_topic(topic)
        metadata_json = json.dumps(
            dict(metadata or {}),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        async with self._write_lock:
            try:
                cursor = await self._connection.execute(
                    """
                    SELECT 1 FROM learning_sessions
                    WHERE session_id = ? AND user_id = ?
                    """,
                    (normalized_session_id, normalized_user_id),
                )
                if await cursor.fetchone() is None:
                    raise LearningDataNotFoundError("学习会话不存在。")
                cursor = await self._connection.execute(
                    """
                    SELECT COALESCE(MAX(message_order), 0)
                    FROM conversation_messages
                    WHERE session_id = ?
                    """,
                    (normalized_session_id,),
                )
                row = await cursor.fetchone()
                next_message_order = int(row[0]) + 1
                await self._connection.execute(
                    """
                    INSERT INTO conversation_messages(
                        message_id, user_id, session_id, message_order,
                        role, content, intent
                    ) VALUES (?, ?, ?, ?, 'user', ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        normalized_user_id,
                        normalized_session_id,
                        next_message_order,
                        user_content,
                        intent,
                    ),
                )
                await self._connection.execute(
                    """
                    INSERT INTO conversation_messages(
                        message_id, user_id, session_id, message_order,
                        role, content, intent
                    ) VALUES (?, ?, ?, ?, 'tutor', ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        normalized_user_id,
                        normalized_session_id,
                        next_message_order + 1,
                        tutor_content,
                        intent,
                    ),
                )
                await self._connection.execute(
                    """
                    INSERT INTO learning_records(
                        record_id, user_id, session_id, topic,
                        activity_type, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        normalized_user_id,
                        normalized_session_id,
                        normalized_topic,
                        activity_type,
                        metadata_json,
                    ),
                )
                await self._connection.execute(
                    """
                    UPDATE learning_sessions
                    SET topic = ?, updated_at = strftime(
                        '%Y-%m-%dT%H:%M:%fZ', 'now'
                    )
                    WHERE session_id = ? AND user_id = ?
                    """,
                    (normalized_topic, normalized_session_id, normalized_user_id),
                )
                await self._connection.commit()
            except LearningDataNotFoundError:
                await self._connection.rollback()
                raise
            except Exception as error:
                await self._connection.rollback()
                raise LearningDataError("Tutor 学习记录保存失败。") from error

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._checkpoint_connection.close()
        finally:
            await self._connection.close()
