"""调用百炼 OpenAI 兼容 Embedding API，并返回文本向量。"""

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__:
    from app.config import load_local_env
else:
    from config import load_local_env


class EmbeddingAPIError(RuntimeError):
    """Embedding API 配置或调用失败时抛出的可读错误。"""


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    base_url: str
    model: str
    dimensions: int

    @classmethod
    def from_environment(cls) -> "EmbeddingConfig":
        load_local_env()
        api_key = os.getenv("BAILIAN_API_KEY", "").strip()
        base_url = os.getenv("BAILIAN_BASE_URL", "").strip().rstrip("/")
        model = os.getenv("BAILIAN_EMBEDDING_MODEL", "text-embedding-v4").strip()
        dimensions_text = os.getenv("BAILIAN_EMBEDDING_DIMENSIONS", "1024").strip()

        if not api_key:
            raise EmbeddingAPIError("缺少 BAILIAN_API_KEY：请复制 .env.example 为 .env 后填写。")
        if not base_url:
            raise EmbeddingAPIError("缺少 BAILIAN_BASE_URL：请从百炼控制台 API 文档复制兼容模式地址。")
        try:
            dimensions = int(dimensions_text)
        except ValueError as error:
            raise EmbeddingAPIError("BAILIAN_EMBEDDING_DIMENSIONS 必须是整数。") from error

        return cls(api_key=api_key, base_url=base_url, model=model, dimensions=dimensions)


def embed_texts(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    """将文本列表发送至 Embedding API，并按输入顺序返回向量列表。"""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("texts 必须至少包含一条非空文本。")

    payload = json.dumps(
        {"model": config.model, "input": texts, "dimensions": config.dimensions}
    ).encode("utf-8")
    request = Request(
        f"{config.base_url}/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise EmbeddingAPIError(f"Embedding API 请求失败（HTTP {error.code}）：{detail}") from error
    except URLError as error:
        raise EmbeddingAPIError(f"无法连接 Embedding API：{error.reason}") from error

    data = body.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise EmbeddingAPIError("Embedding API 返回格式异常：缺少与输入对应的 data。")

    try:
        ordered_items = sorted(data, key=lambda item: item["index"])
        return [[float(number) for number in item["embedding"]] for item in ordered_items]
    except (KeyError, TypeError, ValueError) as error:
        raise EmbeddingAPIError("Embedding API 返回格式异常：无法读取向量。") from error
