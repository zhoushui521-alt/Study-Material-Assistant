"""Minimal password authentication and revocable server-side sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.learning_data import (
    LearningDataConflictError,
    LearningDataNotFoundError,
    LearningDataStore,
    UserRecord,
)


AUTH_COOKIE_NAME = "zhixing_session"
AUTH_SESSION_TTL = timedelta(days=7)
MAX_EMAIL_LENGTH = 254
MAX_DISPLAY_NAME_LENGTH = 80
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 128
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthenticationError(RuntimeError):
    """Authentication could not be completed safely."""


class AuthenticationValidationError(AuthenticationError):
    """Registration or login input is invalid."""


class AuthenticationConflictError(AuthenticationError):
    """A unique identity already exists."""


class AuthenticationCredentialsError(AuthenticationError):
    """Credentials or a session token are invalid."""


@dataclass(frozen=True)
class AuthenticatedSession:
    user: UserRecord
    token: str
    expires_at: str


def normalize_email(value: str) -> str:
    normalized = value.strip().casefold()
    if (
        not normalized
        or len(normalized) > MAX_EMAIL_LENGTH
        or _EMAIL_PATTERN.fullmatch(normalized) is None
    ):
        raise AuthenticationValidationError("请输入有效邮箱地址。")
    return normalized


def normalize_display_name(value: str) -> str:
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > MAX_DISPLAY_NAME_LENGTH:
        raise AuthenticationValidationError(
            f"显示名称不能为空且不能超过 {MAX_DISPLAY_NAME_LENGTH} 个字符。"
        )
    return normalized


def validate_password(value: str) -> str:
    if not isinstance(value, str):
        raise AuthenticationValidationError("密码格式无效。")
    if len(value) < MIN_PASSWORD_LENGTH or len(value) > MAX_PASSWORD_LENGTH:
        raise AuthenticationValidationError(
            f"密码长度必须介于 {MIN_PASSWORD_LENGTH} 和 {MAX_PASSWORD_LENGTH} 个字符。"
        )
    if len(value.encode("utf-8")) > MAX_PASSWORD_LENGTH * 4:
        raise AuthenticationValidationError("密码格式无效。")
    return value


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")


def hash_password(password: str) -> str:
    validated = validate_password(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        validated.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
    )
    return (
        f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}$"
        f"{_b64encode(salt)}${_b64encode(digest)}"
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n_value, r_value, p_value, salt_value, digest_value = encoded.split(
            "$", maxsplit=5
        )
        if scheme != "scrypt":
            return False
        n = int(n_value)
        r = int(r_value)
        p = int(p_value)
        if (n, r, p) != (SCRYPT_N, SCRYPT_R, SCRYPT_P):
            return False
        salt = _b64decode(salt_value)
        expected = _b64decode(digest_value)
        if len(salt) != 16 or len(expected) != SCRYPT_DKLEN:
            return False
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
        )
    except (TypeError, ValueError, UnicodeError):
        return False
    return hmac.compare_digest(candidate, expected)


_DUMMY_PASSWORD_HASH = hash_password("invalid-password-value")


def session_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def utc_now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AuthenticationService:
    """Coordinates password verification with the SQLite identity store."""

    def __init__(self, store: LearningDataStore) -> None:
        self._store = store

    async def _issue_session(self, user: UserRecord) -> AuthenticatedSession:
        token = secrets.token_urlsafe(32)
        expires_at = (
            datetime.now(UTC) + AUTH_SESSION_TTL
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        await self._store.create_auth_session(
            user.user_id,
            session_token_hash(token),
            expires_at,
        )
        return AuthenticatedSession(user=user, token=token, expires_at=expires_at)

    async def register(
        self,
        *,
        email: str,
        password: str,
        display_name: str,
    ) -> AuthenticatedSession:
        normalized_email = normalize_email(email)
        normalized_name = normalize_display_name(display_name)
        password_hash = hash_password(password)
        try:
            user = await self._store.create_user(
                email=normalized_email,
                password_hash=password_hash,
                display_name=normalized_name,
            )
        except LearningDataConflictError as error:
            raise AuthenticationConflictError("该邮箱已经注册。") from error
        return await self._issue_session(user)

    async def login(self, *, email: str, password: str) -> AuthenticatedSession:
        normalized_email = normalize_email(email)
        try:
            candidate_password = validate_password(password)
        except AuthenticationValidationError:
            candidate_password = ""
        try:
            user = await self._store.get_user_by_email(normalized_email)
        except LearningDataNotFoundError:
            user = None
        stored_hash = (
            user.password_hash if user is not None else _DUMMY_PASSWORD_HASH
        )
        password_matches = verify_password(candidate_password, stored_hash or "")
        if user is None or user.password_hash is None or not password_matches:
            raise AuthenticationCredentialsError("邮箱或密码错误。")
        return await self._issue_session(user)

    async def authenticate(self, token: str | None) -> UserRecord:
        if not isinstance(token, str) or not 32 <= len(token) <= 256:
            raise AuthenticationCredentialsError("登录状态无效或已过期。")
        try:
            return await self._store.get_user_by_auth_session(
                session_token_hash(token),
                now=utc_now_text(),
            )
        except LearningDataNotFoundError as error:
            raise AuthenticationCredentialsError("登录状态无效或已过期。") from error

    async def logout(self, token: str | None) -> None:
        if not isinstance(token, str) or not token:
            return
        await self._store.delete_auth_session(session_token_hash(token))
