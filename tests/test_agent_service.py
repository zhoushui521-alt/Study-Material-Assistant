import asyncio
import unittest
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, Mock

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from app.agent_service import (
    AGENT_TOOL_NAMES,
    ANSWER_TOOL_NAME,
    LIST_TOOL_NAME,
    PREVIEW_TOOL_NAME,
    AgentExecutionError,
    AgentService,
    AgentTimeoutError,
    AgentToolAuthorizationError,
    answer_from_materials,
    create_agent_service,
    list_available_materials,
    preview_web_material,
)
from app.langchain_rag import RAGAnswer
from app.material_ingestion import MaterialFile, MaterialManager
from app.rag_service import RAGService
from app.web_materials import WebMaterialPreview, WebMaterialService


def tool_call(name: str, arguments: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


class ScriptedToolCallingModel(BaseChatModel):
    """只按测试脚本返回 ToolCall 或最终消息，不访问真实模型。"""

    _responses: deque[AIMessage] = PrivateAttr()
    _seen_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tool_names: tuple[str, ...] = PrivateAttr(default=())
    _call_count: int = PrivateAttr(default=0)

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        self._responses = deque(responses)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling-model"

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def seen_messages(self) -> list[list[BaseMessage]]:
        return self._seen_messages

    @property
    def bound_tool_names(self) -> tuple[str, ...]:
        return self._bound_tool_names

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolCallingModel":
        del kwargs
        self._bound_tool_names = tuple(tool.name for tool in tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self._call_count += 1
        self._seen_messages.append(list(messages))
        message = self._responses.popleft()
        return ChatResult(generations=[ChatGeneration(message=message)])


def build_dependencies(
    model: BaseChatModel,
) -> tuple[Mock, Mock, Mock]:
    rag_service = Mock(spec=RAGService)
    rag_service.chat_model = model
    material_manager = Mock(spec=MaterialManager)
    web_material_service = Mock(spec=WebMaterialService)
    web_material_service.preview = AsyncMock()
    return rag_service, material_manager, web_material_service


class AgentServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_exposes_only_three_bounded_tools(self) -> None:
        self.assertEqual(
            AGENT_TOOL_NAMES,
            (ANSWER_TOOL_NAME, LIST_TOOL_NAME, PREVIEW_TOOL_NAME),
        )
        self.assertEqual(answer_from_materials.name, ANSWER_TOOL_NAME)
        self.assertEqual(list_available_materials.name, LIST_TOOL_NAME)
        self.assertEqual(preview_web_material.name, PREVIEW_TOOL_NAME)
        self.assertNotIn("index", " ".join(AGENT_TOOL_NAMES))
        self.assertNotIn("delete", " ".join(AGENT_TOOL_NAMES))

    async def test_agent_selects_material_list_without_opening_rag(self) -> None:
        model = ScriptedToolCallingModel(
            [
                tool_call(LIST_TOOL_NAME, {}, "list-1"),
                AIMessage(content="当前有一份学习资料。"),
            ]
        )
        rag_service, manager, web_service = build_dependencies(model)
        manager.list_materials.return_value = (
            MaterialFile(filename="rag.md", size_bytes=128),
        )
        service = create_agent_service(
            rag_service,
            manager,
            web_service,
            model=model,
        )

        result = await service.ask("现在有哪些资料？")

        self.assertEqual(result.answer, "当前有一份学习资料。")
        self.assertEqual(result.sources, ())
        self.assertEqual(result.tools_used, (LIST_TOOL_NAME,))
        self.assertEqual(model.bound_tool_names, AGENT_TOOL_NAMES)
        manager.list_materials.assert_called_once_with()
        rag_service.ask.assert_not_called()
        web_service.preview.assert_not_awaited()

    async def test_rag_tool_preserves_exact_answer_and_sources(self) -> None:
        model = ScriptedToolCallingModel(
            [
                tool_call(
                    ANSWER_TOOL_NAME,
                    {"question": "RAG 是什么？"},
                    "answer-1",
                ),
                AIMessage(content="模型尝试改写答案。"),
            ]
        )
        rag_service, manager, web_service = build_dependencies(model)
        rag_service.ask.return_value = RAGAnswer(
            answer="RAG 会先检索资料。[rag.md · 第 2 段]",
            sources=(
                Document(
                    page_content="RAG 会先检索资料。",
                    metadata={"source": "rag.md", "chunk_index": 2},
                ),
            ),
        )
        service = create_agent_service(
            rag_service,
            manager,
            web_service,
            model=model,
        )

        result = await service.ask("请根据资料解释 RAG。")

        self.assertEqual(
            result.answer,
            "RAG 会先检索资料。[rag.md · 第 2 段]",
        )
        self.assertEqual(result.sources, ("[rag.md · 第 2 段]",))
        self.assertEqual(result.tools_used, (ANSWER_TOOL_NAME,))
        rag_service.ask.assert_called_once_with("RAG 是什么？")

    async def test_web_preview_requires_separate_authorization(self) -> None:
        model = ScriptedToolCallingModel(
            [
                tool_call(
                    PREVIEW_TOOL_NAME,
                    {"url": "https://example.com/", "operation": "add"},
                    "preview-1",
                ),
                AIMessage(content="网页预览未获授权，已停止。"),
            ]
        )
        rag_service, manager, web_service = build_dependencies(model)
        service = create_agent_service(
            rag_service,
            manager,
            web_service,
            model=model,
        )

        result = await service.ask("预览 https://example.com/")

        self.assertEqual(result.answer, "网页预览未获授权，已停止。")
        self.assertEqual(result.tools_used, (PREVIEW_TOOL_NAME,))
        web_service.preview.assert_not_awaited()
        tool_messages = [
            message
            for message in model.seen_messages[-1]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertNotIn("https://example.com", tool_messages[0].content)

    async def test_authorized_web_preview_never_indexes(self) -> None:
        model = ScriptedToolCallingModel(
            [
                tool_call(
                    PREVIEW_TOOL_NAME,
                    {"url": "https://example.com/", "operation": "add"},
                    "preview-1",
                ),
                AIMessage(content="网页已生成预览，尚未写入索引。"),
            ]
        )
        rag_service, manager, web_service = build_dependencies(model)
        web_service.preview.return_value = WebMaterialPreview(
            upload_id="a" * 32,
            filename="web-example-com.md",
            operation="add",
            requested_url="https://example.com/",
            canonical_url="https://example.com/",
            title="Example Domain",
            crawled_at="2026-08-10T10:00:00Z",
            content_sha256="b" * 64,
            markdown="示例正文",
            redirect_count=0,
            size_bytes=128,
            document_units=1,
            chunk_count=1,
            embedding_batch_count=1,
        )
        service = create_agent_service(
            rag_service,
            manager,
            web_service,
            model=model,
        )

        result = await service.ask(
            "预览 https://example.com/",
            allow_web_preview=True,
        )

        self.assertEqual(result.answer, "网页已生成预览，尚未写入索引。")
        self.assertEqual(result.tools_used, (PREVIEW_TOOL_NAME,))
        web_service.preview.assert_awaited_once_with(
            "https://example.com/",
            operation="add",
        )
        rag_service.ask.assert_not_called()
        manager.list_materials.assert_not_called()

    async def test_concurrent_runs_keep_preview_authorization_isolated(self) -> None:
        class DirectPreviewAgent:
            async def ainvoke(
                self,
                inputs: dict[str, Any],
                **kwargs: Any,
            ) -> dict[str, Any]:
                del inputs, kwargs
                await asyncio.sleep(0)
                try:
                    await preview_web_material.ainvoke(
                        {"url": "https://example.com/", "operation": "add"}
                    )
                except AgentToolAuthorizationError:
                    answer = "未授权"
                else:
                    answer = "已授权"
                return {"messages": [AIMessage(content=answer)]}

        rag_service, manager, web_service = build_dependencies(
            ScriptedToolCallingModel([AIMessage(content="unused")])
        )
        web_service.preview.return_value = WebMaterialPreview(
            upload_id="a" * 32,
            filename="web-example-com.md",
            operation="add",
            requested_url="https://example.com/",
            canonical_url="https://example.com/",
            title="Example Domain",
            crawled_at="2026-08-10T10:00:00Z",
            content_sha256="b" * 64,
            markdown="示例正文",
            redirect_count=0,
            size_bytes=128,
            document_units=1,
            chunk_count=1,
            embedding_batch_count=1,
        )
        service = AgentService(
            agent=DirectPreviewAgent(),
            rag_service=rag_service,
            material_manager=manager,
            web_material_service=web_service,
        )

        denied, allowed = await asyncio.gather(
            service.ask("不授权网页预览", allow_web_preview=False),
            service.ask("授权网页预览", allow_web_preview=True),
        )

        self.assertEqual(denied.answer, "未授权")
        self.assertEqual(allowed.answer, "已授权")
        web_service.preview.assert_awaited_once_with(
            "https://example.com/",
            operation="add",
        )

    async def test_per_tool_limit_blocks_duplicate_paid_rag_call(self) -> None:
        model = ScriptedToolCallingModel(
            [
                tool_call(ANSWER_TOOL_NAME, {"question": "问题一"}, "answer-1"),
                tool_call(ANSWER_TOOL_NAME, {"question": "问题二"}, "answer-2"),
                AIMessage(content="已停止重复调用。"),
            ]
        )
        rag_service, manager, web_service = build_dependencies(model)
        rag_service.ask.return_value = RAGAnswer(answer="第一份回答", sources=())
        service = create_agent_service(
            rag_service,
            manager,
            web_service,
            model=model,
        )

        result = await service.ask("连续回答两个问题")

        self.assertEqual(result.answer, "第一份回答")
        self.assertEqual(
            result.tools_used,
            (ANSWER_TOOL_NAME, ANSWER_TOOL_NAME),
        )
        rag_service.ask.assert_called_once_with("问题一")
        self.assertLessEqual(model.call_count, 3)

    async def test_tool_failure_does_not_expose_internal_detail(self) -> None:
        model = ScriptedToolCallingModel(
            [
                tool_call(LIST_TOOL_NAME, {}, "list-1"),
                AIMessage(content="资料列表暂时不可用。"),
            ]
        )
        rag_service, manager, web_service = build_dependencies(model)
        manager.list_materials.side_effect = RuntimeError(
            "BAILIAN_API_KEY=secret-value"
        )
        service = create_agent_service(
            rag_service,
            manager,
            web_service,
            model=model,
        )

        result = await service.ask("列出资料")

        self.assertEqual(result.answer, "资料列表暂时不可用。")
        observed = "\n".join(
            str(message.content)
            for batch in model.seen_messages
            for message in batch
        )
        self.assertNotIn("secret-value", observed)

    async def test_timeout_is_wrapped_without_internal_detail(self) -> None:
        class SlowAgent:
            async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                del args, kwargs
                await asyncio.sleep(1)
                return {"messages": [AIMessage(content="too late")]}

        rag_service, manager, web_service = build_dependencies(
            ScriptedToolCallingModel([AIMessage(content="unused")])
        )
        service = AgentService(
            agent=SlowAgent(),
            rag_service=rag_service,
            material_manager=manager,
            web_material_service=web_service,
            timeout_seconds=0.01,
        )

        with self.assertRaisesRegex(AgentTimeoutError, "超时"):
            await service.ask("测试超时")

    async def test_malformed_agent_result_is_rejected(self) -> None:
        agent = AsyncMock()
        agent.ainvoke.return_value = {"messages": ["not-a-message"]}
        rag_service, manager, web_service = build_dependencies(
            ScriptedToolCallingModel([AIMessage(content="unused")])
        )
        service = AgentService(
            agent=agent,
            rag_service=rag_service,
            material_manager=manager,
            web_material_service=web_service,
        )

        with self.assertRaisesRegex(AgentExecutionError, "格式无效"):
            await service.ask("返回格式测试")

    async def test_failed_rag_tool_artifact_is_never_returned_as_evidence(self) -> None:
        agent = AsyncMock()
        agent.ainvoke.return_value = {
            "messages": [
                ToolMessage(
                    content="资料问答失败",
                    tool_call_id="answer-1",
                    name=ANSWER_TOOL_NAME,
                    status="error",
                    artifact={
                        "answer": "不可信失败产物",
                        "sources": ["[secret.md · 第 1 段]"],
                    },
                ),
                AIMessage(content="未能完成资料问答。"),
            ]
        }
        rag_service, manager, web_service = build_dependencies(
            ScriptedToolCallingModel([AIMessage(content="unused")])
        )
        service = AgentService(
            agent=agent,
            rag_service=rag_service,
            material_manager=manager,
            web_material_service=web_service,
        )

        result = await service.ask("测试失败证据")

        self.assertEqual(result.answer, "未能完成资料问答。")
        self.assertEqual(result.sources, ())


if __name__ == "__main__":
    unittest.main()
