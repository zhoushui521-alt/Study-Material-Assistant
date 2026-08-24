"""Model Gateway 门面、基础路由与 BYOK 优先策略。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings
from app.model_gateway.credentials import (
    CredentialCipher,
    ModelCredentialMetadata,
    ModelCredentialStore,
    require_cipher,
)
from app.model_gateway.errors import (
    ModelAuthenticationError,
    ModelCredentialDecryptionError,
    ModelGatewayConfigurationError,
    ModelGatewayError,
    UnsupportedModelProviderError,
)
from app.model_gateway.providers import (
    ModelRoute,
    ProviderAdapterRegistry,
)


PROVIDER_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}$")
PROVIDER_DEFAULT_BASE_URLS: dict[str, str | None] = {
    "openai": None,
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai_compatible": None,
}


@dataclass(frozen=True)
class ModelSelection:
    provider: str
    model_name: str
    credential_source: Literal["system", "byok"]
    byok_configured: bool
    created_at: str | None = None
    updated_at: str | None = None


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if not PROVIDER_NAME_PATTERN.fullmatch(normalized):
        raise UnsupportedModelProviderError("模型 Provider 名称无效。")
    if normalized not in PROVIDER_DEFAULT_BASE_URLS:
        raise UnsupportedModelProviderError("模型 Provider 未受支持。")
    return normalized


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    if not MODEL_NAME_PATTERN.fullmatch(normalized):
        raise ValueError("模型名称格式无效。")
    return normalized


def normalize_api_key(api_key: str) -> str:
    normalized = api_key.strip()
    if not normalized or len(normalized) > 4096:
        raise ValueError("API Key 不能为空且不能超过 4096 个字符。")
    if any(character.isspace() for character in normalized):
        raise ValueError("API Key 不能包含空白字符。")
    return normalized


def is_model_authentication_error(error: BaseException) -> bool:
    """只按稳定类型/状态判断，不读取或传播 Provider 错误正文。"""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ModelAuthenticationError):
            return True
        name = type(current).__name__.casefold()
        status_code = getattr(current, "status_code", None)
        if status_code in {401, 403}:
            return True
        if name in {"authenticationerror", "permissiondeniederror"}:
            return True
        current = current.__cause__ or current.__context__
    return False


class ModelGateway:
    """统一处理用户凭证优先、系统回退、Provider 选择与客户端构造。"""

    def __init__(
        self,
        *,
        settings: Settings,
        credential_store: ModelCredentialStore | None = None,
        adapter_registry: ProviderAdapterRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.credential_store = credential_store or ModelCredentialStore(
            settings.model_credentials_database_path
        )
        self.adapter_registry = adapter_registry or ProviderAdapterRegistry()
        self._cipher: CredentialCipher | None = None
        if settings.model_credential_encryption_key:
            self._cipher = require_cipher(settings.model_credential_encryption_key)
        self._default_provider = normalize_provider(settings.default_model_provider)
        self._default_model = normalize_model_name(settings.default_model_name)
        if self._default_provider not in self.adapter_registry.providers:
            raise UnsupportedModelProviderError("默认模型 Provider 未注册。")

    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelGateway":
        return cls(settings=settings)

    @property
    def supported_providers(self) -> tuple[str, ...]:
        return self.adapter_registry.providers

    def _base_url(self, provider: str) -> str | None:
        normalized = normalize_provider(provider)
        if (
            normalized == self._default_provider
            and self.settings.model_base_url
        ):
            return self.settings.model_base_url
        base_url = PROVIDER_DEFAULT_BASE_URLS[normalized]
        if normalized == "openai_compatible" and base_url is None:
            raise ModelGatewayConfigurationError(
                "openai_compatible Provider 必须由服务端配置 MODEL_BASE_URL。"
            )
        return base_url

    def credential_metadata(self, user_id: str) -> ModelCredentialMetadata | None:
        return self.credential_store.metadata(user_id)

    def current_selection(self, user_id: str) -> ModelSelection:
        metadata = self.credential_metadata(user_id)
        if metadata is None:
            return ModelSelection(
                provider=self._default_provider,
                model_name=self._default_model,
                credential_source="system",
                byok_configured=False,
            )
        return ModelSelection(
            provider=metadata.provider,
            model_name=metadata.model_name,
            credential_source="byok",
            byok_configured=True,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
        )

    def save_credential(
        self,
        *,
        user_id: str,
        provider: str,
        model_name: str,
        api_key: str,
    ) -> ModelCredentialMetadata:
        normalized_provider = normalize_provider(provider)
        normalized_model = normalize_model_name(model_name)
        normalized_key = normalize_api_key(api_key)
        self._base_url(normalized_provider)
        cipher = self._cipher or require_cipher(
            self.settings.model_credential_encryption_key
        )
        return self.credential_store.upsert(
            user_id=user_id,
            provider=normalized_provider,
            model_name=normalized_model,
            encrypted_api_key=cipher.encrypt(normalized_key),
        )

    def delete_credential(self, user_id: str) -> bool:
        return self.credential_store.delete(user_id)

    def route(self, user_id: str | None = None) -> ModelRoute:
        record = self.credential_store.get(user_id) if user_id is not None else None
        if record is not None:
            if self._cipher is None:
                raise ModelCredentialDecryptionError(
                    "已保存 BYOK 凭证，但服务端没有可用的加密主密钥。"
                )
            api_key = self._cipher.decrypt(record.encrypted_api_key)
            provider = normalize_provider(record.provider)
            model_name = normalize_model_name(record.model_name)
            source: Literal["system", "byok"] = "byok"
        else:
            api_key = normalize_api_key(self.settings.model_api_key)
            provider = self._default_provider
            model_name = self._default_model
            source = "system"
        return ModelRoute(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=self._base_url(provider),
            temperature=self.settings.model_temperature,
            timeout_seconds=self.settings.model_timeout_seconds,
            max_tokens=self.settings.model_max_tokens,
            credential_source=source,
        )

    def create_chat_model(self, user_id: str | None = None) -> BaseChatModel:
        return self.adapter_registry.create_chat_model(self.route(user_id))


__all__ = [
    "ModelAuthenticationError",
    "ModelGateway",
    "ModelGatewayConfigurationError",
    "ModelGatewayError",
    "ModelSelection",
    "UnsupportedModelProviderError",
    "is_model_authentication_error",
    "normalize_api_key",
    "normalize_model_name",
    "normalize_provider",
]
