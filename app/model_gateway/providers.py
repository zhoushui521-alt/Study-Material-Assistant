"""Provider Adapter：把统一模型路由转换成具体 LangChain ChatModel。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.model_gateway.errors import UnsupportedModelProviderError


CredentialSource = Literal["system", "byok"]


@dataclass(frozen=True)
class ModelRoute:
    provider: str
    model_name: str
    api_key: str = field(repr=False)
    base_url: str | None
    temperature: float
    timeout_seconds: int
    max_tokens: int | None
    credential_source: CredentialSource


class ProviderAdapter(Protocol):
    def create_chat_model(self, route: ModelRoute) -> BaseChatModel:
        """创建符合业务层 BaseChatModel 契约的客户端。"""


class OpenAICompatibleProviderAdapter:
    """复用 LangChain ChatOpenAI 适配 OpenAI-compatible Chat API。"""

    def create_chat_model(self, route: ModelRoute) -> BaseChatModel:
        parameters: dict[str, object] = {
            "api_key": route.api_key,
            "model": route.model_name,
            "temperature": route.temperature,
            "timeout": route.timeout_seconds,
            "max_retries": 2,
        }
        if route.base_url is not None:
            parameters["base_url"] = route.base_url
        if route.max_tokens is not None:
            parameters["max_completion_tokens"] = route.max_tokens
        model = ChatOpenAI(**parameters)
        object.__setattr__(model, "_model_gateway_provider", route.provider)
        object.__setattr__(
            model,
            "_model_gateway_credential_source",
            route.credential_source,
        )
        return model


class ProviderAdapterRegistry:
    """显式 Provider 注册表；业务层不依赖具体 SDK 或 Provider 名称。"""

    def __init__(
        self,
        adapters: Mapping[str, ProviderAdapter] | None = None,
    ) -> None:
        if adapters is None:
            openai_compatible = OpenAICompatibleProviderAdapter()
            adapters = {
                "openai": openai_compatible,
                "deepseek": openai_compatible,
                "qwen": openai_compatible,
                "openai_compatible": openai_compatible,
            }
        self._adapters = dict(adapters)

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def create_chat_model(self, route: ModelRoute) -> BaseChatModel:
        adapter = self._adapters.get(route.provider)
        if adapter is None:
            raise UnsupportedModelProviderError("模型 Provider 未注册。")
        return adapter.create_chat_model(route)
