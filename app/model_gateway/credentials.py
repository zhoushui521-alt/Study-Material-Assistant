"""用户 BYOK 凭证的认证加密与独立 SQLite 持久化。"""

from __future__ import annotations

import sqlite3
import uuid
from binascii import Error as BinasciiError
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from cryptography.fernet import Fernet, InvalidToken

from app.model_gateway.errors import (
    ModelCredentialDecryptionError,
    ModelCredentialEncryptionUnavailableError,
    ModelCredentialStoreError,
    ModelGatewayConfigurationError,
)


MODEL_CREDENTIAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelCredentialMetadata:
    user_id: str
    provider: str
    model_name: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StoredModelCredential(ModelCredentialMetadata):
    encrypted_api_key: bytes = field(repr=False)


class CredentialCipher:
    """使用 Fernet 提供机密性与完整性校验；主密钥只来自运行环境。"""

    def __init__(self, key: str) -> None:
        try:
            self._fernet = Fernet(key.encode("ascii"))
        except (UnicodeEncodeError, ValueError, BinasciiError) as error:
            raise ModelGatewayConfigurationError(
                "MODEL_CREDENTIAL_ENCRYPTION_KEY 必须是有效的 Fernet Key。"
            ) from error

    def encrypt(self, api_key: str) -> bytes:
        return self._fernet.encrypt(api_key.encode("utf-8"))

    def decrypt(self, encrypted_api_key: bytes) -> str:
        try:
            return self._fernet.decrypt(encrypted_api_key).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError) as error:
            raise ModelCredentialDecryptionError(
                "模型凭证无法使用当前加密主密钥解密。"
            ) from error


class ModelCredentialStore:
    """每次操作使用短连接；适配当前单实例 SQLite 与线程池调用。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._lock = RLock()

    @staticmethod
    def _canonical_user_id(user_id: str) -> str:
        try:
            return str(uuid.UUID(user_id))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("user_id 必须是有效 UUID。") from error

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=5)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = WAL")
            self._apply_schema(connection)
            return connection
        except Exception as error:
            if connection is not None:
                connection.close()
            raise ModelCredentialStoreError("模型凭证数据库不可用。") from error

    @staticmethod
    def _apply_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_gateway_schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            );

            CREATE TABLE IF NOT EXISTS user_model_credentials (
                user_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                encrypted_api_key BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                ),
                updated_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            );

            INSERT OR IGNORE INTO model_gateway_schema_migrations(version)
            VALUES (1);
            """
        )
        row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM model_gateway_schema_migrations"
        ).fetchone()
        version = int(row[0]) if row is not None else 0
        if version > MODEL_CREDENTIAL_SCHEMA_VERSION:
            raise ModelCredentialStoreError(
                "模型凭证数据库版本高于当前程序支持范围。"
            )
        connection.commit()

    @staticmethod
    def _metadata(row: sqlite3.Row) -> ModelCredentialMetadata:
        return ModelCredentialMetadata(
            user_id=row["user_id"],
            provider=row["provider"],
            model_name=row["model_name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, user_id: str) -> StoredModelCredential | None:
        normalized_user_id = self._canonical_user_id(user_id)
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT user_id, provider, model_name, encrypted_api_key,
                           created_at, updated_at
                    FROM user_model_credentials
                    WHERE user_id = ?
                    """,
                    (normalized_user_id,),
                ).fetchone()
            except Exception as error:
                raise ModelCredentialStoreError("模型凭证读取失败。") from error
            finally:
                connection.close()
        if row is None:
            return None
        return StoredModelCredential(
            **self._metadata(row).__dict__,
            encrypted_api_key=bytes(row["encrypted_api_key"]),
        )

    def metadata(self, user_id: str) -> ModelCredentialMetadata | None:
        record = self.get(user_id)
        if record is None:
            return None
        return ModelCredentialMetadata(
            user_id=record.user_id,
            provider=record.provider,
            model_name=record.model_name,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def upsert(
        self,
        *,
        user_id: str,
        provider: str,
        model_name: str,
        encrypted_api_key: bytes,
    ) -> ModelCredentialMetadata:
        normalized_user_id = self._canonical_user_id(user_id)
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT INTO user_model_credentials(
                        user_id, provider, model_name, encrypted_api_key
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        provider = excluded.provider,
                        model_name = excluded.model_name,
                        encrypted_api_key = excluded.encrypted_api_key,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    """,
                    (
                        normalized_user_id,
                        provider,
                        model_name,
                        sqlite3.Binary(encrypted_api_key),
                    ),
                )
                connection.commit()
            except Exception as error:
                connection.rollback()
                raise ModelCredentialStoreError("模型凭证保存失败。") from error
            finally:
                connection.close()
        metadata = self.metadata(normalized_user_id)
        if metadata is None:
            raise ModelCredentialStoreError("模型凭证保存后无法读取。")
        return metadata

    def delete(self, user_id: str) -> bool:
        normalized_user_id = self._canonical_user_id(user_id)
        with self._lock:
            connection = self._connect()
            try:
                cursor = connection.execute(
                    "DELETE FROM user_model_credentials WHERE user_id = ?",
                    (normalized_user_id,),
                )
                connection.commit()
                return cursor.rowcount > 0
            except Exception as error:
                connection.rollback()
                raise ModelCredentialStoreError("模型凭证删除失败。") from error
            finally:
                connection.close()


def require_cipher(encryption_key: str) -> CredentialCipher:
    if not encryption_key.strip():
        raise ModelCredentialEncryptionUnavailableError(
            "服务端未配置模型凭证加密主密钥。"
        )
    return CredentialCipher(encryption_key.strip())
