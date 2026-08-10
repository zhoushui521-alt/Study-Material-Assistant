"""调用百炼 OpenAI 兼容 Chat API，基于检索资料生成回答。"""

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__:
    from app.config import load_local_env
    from app.chunk_documents import DocumentChunk
else:
    from config import load_local_env
    from chunk_documents import DocumentChunk


class ChatAPIError(RuntimeError):
    """Chat API 配置或调用失败时抛出的可读错误。"""


@dataclass(frozen=True)
class ChatConfig:
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_environment(cls) -> "ChatConfig":
        load_local_env()
        api_key = os.getenv("BAILIAN_API_KEY", "").strip()
        base_url = os.getenv("BAILIAN_BASE_URL", "").strip().rstrip("/")
        model = os.getenv("BAILIAN_CHAT_MODEL", "qwen-plus").strip()
        if not api_key:
            raise ChatAPIError("缺少 BAILIAN_API_KEY。")
        if not base_url:
            raise ChatAPIError("缺少 BAILIAN_BASE_URL。")
        if not model:
            raise ChatAPIError("缺少 BAILIAN_CHAT_MODEL。")
        return cls(api_key=api_key, base_url=base_url, model=model)


def build_messages(question: str, sources: list[DocumentChunk]) -> list[dict[str, str]]:
    """构造受资料约束的回答提示词，减少无依据回答。"""
    source_text = "\n\n".join(
        f"[{chunk.source} · 第 {chunk.index} 段]\n{chunk.content}" for chunk in sources
    )
    return [
        {
            "role": "system",
            "content": "你是学习资料助手。只能依据提供的资料回答；资料不足时明确说明。"
            "回答末尾必须列出你使用的来源标签。",
        },
        {
            "role": "user",
            "content": f"问题：{question}\n\n资料：\n{source_text}",
        },
    ]


def generate_answer(question: str, sources: list[DocumentChunk], config: ChatConfig) -> str:
    """调用 Chat API，以检索到的资料作为上下文生成回答。"""
    if not question.strip():
        raise ValueError("问题不能为空。")
    if not sources:
        raise ValueError("没有检索到资料，不能生成基于资料的回答。")

    payload = json.dumps(
        {"model": config.model, "messages": build_messages(question, sources), "temperature": 0.2}
    ).encode("utf-8")
    request = Request(
        f"{config.base_url}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ChatAPIError(f"Chat API 请求失败（HTTP {error.code}）：{detail}") from error
    except URLError as error:
        raise ChatAPIError(f"无法连接 Chat API：{error.reason}") from error

    try:
        return str(body["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as error:
        raise ChatAPIError("Chat API 返回格式异常：无法读取回答内容。") from error
