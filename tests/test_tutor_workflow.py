import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

from app.evidence import Citation, build_evidence_context
from app.langchain_rag import NO_EVIDENCE_ANSWER, RAGAnswer
from app.learning_data import (
    ConversationMessageRecord,
    LearningDataStore,
    LearningSessionRecord,
)
from app.rag_service import RAGService
from app.tutor_workflow import (
    KNOWLEDGE_TOOL_NAME,
    QUIZ_TOOL_NAME,
    SUMMARY_TOOL_NAME,
    TUTOR_MAX_HISTORY_CHARACTERS,
    KnowledgeRetrievalTool,
    KnowledgeToolResult,
    LearningSummaryDraft,
    LearningSummaryTool,
    QuizDraft,
    QuizGeneratorTool,
    TutorExecutionError,
    TutorTools,
    TutorWorkflowService,
    build_tutor_graph,
    classify_tutor_intent,
)


SESSION_ID = "11111111-1111-4111-8111-111111111111"
USER_ID = "22222222-2222-4222-8222-222222222222"


class FakeLearningStore:
    def __init__(self) -> None:
        self.topic = "本次学习内容"
        self.messages: list[ConversationMessageRecord] = []
        self.exchanges: list[dict[str, object]] = []

    async def get_session(
        self,
        user_id: str,
        session_id: str,
    ) -> LearningSessionRecord:
        return LearningSessionRecord(
            session_id=session_id,
            user_id=user_id,
            topic=self.topic,
            created_at="2026-08-21T00:00:00.000Z",
            updated_at="2026-08-21T00:00:00.000Z",
        )

    async def list_messages(
        self,
        user_id: str,
        session_id: str,
        *,
        limit: int,
    ) -> tuple[ConversationMessageRecord, ...]:
        del user_id, session_id
        return tuple(self.messages[-limit:])

    async def record_tutor_exchange(self, **values: object) -> None:
        self.exchanges.append(values)
        self.topic = str(values["topic"])
        intent = str(values["intent"])
        session_id = str(values["session_id"])
        user_id = str(values["user_id"])
        for role, content_key in (
            ("user", "user_content"),
            ("tutor", "tutor_content"),
        ):
            self.messages.append(
                ConversationMessageRecord(
                    message_id=str(uuid4()),
                    session_id=session_id,
                    user_id=user_id,
                    role=role,
                    content=str(values[content_key]),
                    intent=intent,
                    created_at="2026-08-21T00:00:00.000Z",
                )
            )


def grounded_knowledge() -> KnowledgeToolResult:
    document = Document(
        page_content="Embedding 把文本映射为可比较的向量表示。",
        metadata={
            "source": "rag.md",
            "filename": "rag.md",
            "source_type": "markdown",
            "chunk_index": 2,
        },
    )
    evidence = build_evidence_context([document])[0]
    citation = Citation(
        citation_id="S1",
        evidence_id=evidence.evidence_id,
        material_id=evidence.material_id,
        chunk_id=evidence.chunk_id,
        source=evidence.source,
        filename=evidence.filename,
        page=evidence.page,
        chunk_index=evidence.chunk_index,
        excerpt=evidence.excerpt,
        locator=evidence.locator,
    )
    return KnowledgeToolResult(
        answer="Embedding 把文本映射为可比较的向量表示。[S1]",
        sources=("[rag.md · 第 2 段]",),
        citations=(citation,),
        evidence=(evidence,),
    )


def make_service(
    knowledge_result: KnowledgeToolResult | Exception | None = None,
) -> tuple[TutorWorkflowService, Mock, Mock, Mock, object]:
    knowledge = Mock(spec=KnowledgeRetrievalTool)
    knowledge.invoke = AsyncMock(
        side_effect=knowledge_result
        if isinstance(knowledge_result, Exception)
        else None,
        return_value=(
            grounded_knowledge()
            if knowledge_result is None
            else knowledge_result
        ),
    )
    quiz = Mock(spec=QuizGeneratorTool)
    quiz.invoke = AsyncMock(
        return_value=QuizDraft(
            question="Embedding 的主要作用是什么？",
            options=["表示语义", "删除资料"],
            answer="表示语义",
            explanation="资料说明它会把文本映射为向量表示。",
        )
    )
    summary = Mock(spec=LearningSummaryTool)
    summary.invoke = AsyncMock(
        return_value=LearningSummaryDraft(
            summary="本次学习了 Embedding 的基本作用。",
            key_points=["文本会被映射为向量表示"],
            next_steps=["用自己的话复述概念"],
        )
    )
    tools = TutorTools(knowledge=knowledge, quiz=quiz, summary=summary)
    checkpointer = InMemorySaver()
    graph = build_tutor_graph(checkpointer)
    service = TutorWorkflowService(
        graph=graph,
        tools=tools,
        learning_store=FakeLearningStore(),
    )
    return service, knowledge, quiz, summary, graph


class TutorRouterEvaluationTests(unittest.TestCase):
    def test_router_classifies_representative_learning_requests(self) -> None:
        cases = {
            "RAG 和 Embedding 有什么区别？": "knowledge_qa",
            "什么是 Embedding？": "explanation",
            "请给我出题练习 Embedding": "quiz",
            "总结刚才的学习内容": "summary",
            "帮我制定 Embedding 学习计划": "study_plan",
        }

        actual = {
            message: classify_tutor_intent(message) for message in cases
        }

        self.assertEqual(actual, cases)


class KnowledgeRetrievalToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_reuses_rag_service_and_preserves_evidence_contract(self) -> None:
        document = Document(
            page_content="RAG 先检索资料。",
            metadata={"source": "rag.md", "chunk_index": 1},
        )
        evidence = build_evidence_context([document])[0]
        citation = Citation(
            citation_id="S1",
            evidence_id=evidence.evidence_id,
            material_id=evidence.material_id,
            chunk_id=evidence.chunk_id,
            source=evidence.source,
            filename=evidence.filename,
            page=evidence.page,
            chunk_index=evidence.chunk_index,
            excerpt=evidence.excerpt,
            locator=evidence.locator,
        )
        rag_service = Mock(spec=RAGService)
        rag_service.ask.return_value = RAGAnswer(
            answer="RAG 先检索资料。[S1]",
            sources=(document,),
            citations=(citation,),
        )

        result = await KnowledgeRetrievalTool(rag_service).invoke("RAG 是什么？")

        rag_service.ask.assert_called_once_with("RAG 是什么？")
        self.assertEqual(result.citations, (citation,))
        self.assertEqual(result.evidence[0].evidence_id, evidence.evidence_id)
        self.assertEqual(result.sources, ("[rag.md · 第 1 段]",))


class TutorWorkflowEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_explanation_routes_through_rag_and_keeps_citations(self) -> None:
        service, knowledge, quiz, summary, _ = make_service()

        result = await service.chat(USER_ID, SESSION_ID, "什么是 Embedding？")

        self.assertEqual(result.intent, "explanation")
        self.assertEqual(result.learning_action, "explain_concept")
        self.assertEqual(result.tools_used, (KNOWLEDGE_TOOL_NAME,))
        self.assertEqual(result.citations[0]["citation_id"], "S1")
        self.assertEqual(result.evidence[0]["context_id"], "S1")
        self.assertIn("理解检查", result.answer)
        knowledge.invoke.assert_awaited_once_with("什么是 Embedding？")
        quiz.invoke.assert_not_awaited()
        summary.invoke.assert_not_awaited()

    async def test_quiz_calls_retrieval_then_quiz_tool_with_expected_arguments(self) -> None:
        service, knowledge, quiz, summary, _ = make_service()

        result = await service.chat(
            USER_ID,
            SESSION_ID,
            "帮我出题练习 Embedding",
        )

        self.assertEqual(result.intent, "quiz")
        self.assertEqual(
            result.tools_used,
            (KNOWLEDGE_TOOL_NAME, QUIZ_TOOL_NAME),
        )
        knowledge.invoke.assert_awaited_once_with("Embedding有哪些需要掌握的核心概念？")
        quiz.invoke.assert_awaited_once_with(
            "Embedding",
            "Embedding 把文本映射为可比较的向量表示。[S1]",
        )
        self.assertEqual(result.quiz["answer"], "表示语义")
        self.assertIn("练习", result.answer)
        summary.invoke.assert_not_awaited()

    async def test_session_summary_skips_rag_and_uses_prior_turns(self) -> None:
        service, knowledge, quiz, summary, graph = make_service()
        await service.chat(USER_ID, SESSION_ID, "Embedding 有什么作用？")

        result = await service.chat(
            USER_ID,
            SESSION_ID,
            "总结刚才的学习内容",
        )

        self.assertEqual(result.intent, "summary")
        self.assertEqual(result.tools_used, (SUMMARY_TOOL_NAME,))
        self.assertEqual(knowledge.invoke.await_count, 1)
        conversation = summary.invoke.await_args.args[0]
        self.assertEqual(len(conversation), 2)
        self.assertEqual(conversation[0]["role"], "user")
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": SESSION_ID}}
        )
        json.dumps(snapshot.values, ensure_ascii=False)
        self.assertLessEqual(len(snapshot.values["conversation"]), 20)
        self.assertLessEqual(
            sum(len(turn["content"]) for turn in snapshot.values["conversation"]),
            TUTOR_MAX_HISTORY_CHARACTERS,
        )

    async def test_follow_up_quiz_reuses_previous_session_topic(self) -> None:
        service, knowledge, quiz, _, _ = make_service()
        await service.chat(USER_ID, SESSION_ID, "什么是 Embedding？")

        result = await service.chat(USER_ID, SESSION_ID, "继续出一道题")

        self.assertEqual(result.intent, "quiz")
        self.assertEqual(result.topic, "Embedding")
        self.assertEqual(knowledge.invoke.await_count, 2)
        quiz.invoke.assert_awaited_once_with(
            "Embedding",
            "Embedding 把文本映射为可比较的向量表示。[S1]",
        )

    async def test_topic_summary_without_history_retrieves_before_summary(self) -> None:
        service, knowledge, quiz, summary, _ = make_service()

        result = await service.chat(USER_ID, SESSION_ID, "总结 Transformer")

        self.assertEqual(
            result.tools_used,
            (KNOWLEDGE_TOOL_NAME, SUMMARY_TOOL_NAME),
        )
        knowledge.invoke.assert_awaited_once_with("请总结Transformer的核心内容。")
        self.assertEqual(summary.invoke.await_count, 1)
        quiz.invoke.assert_not_awaited()

    async def test_study_plan_is_deterministic_after_grounded_retrieval(self) -> None:
        service, knowledge, quiz, summary, _ = make_service()

        result = await service.chat(
            USER_ID,
            SESSION_ID,
            "制定 Embedding 学习计划",
        )

        self.assertEqual(result.intent, "study_plan")
        self.assertEqual(result.learning_action, "create_study_plan")
        self.assertEqual(result.tools_used, (KNOWLEDGE_TOOL_NAME,))
        self.assertIn("证据驱动的学习闭环", result.summary["summary"])
        quiz.invoke.assert_not_awaited()
        summary.invoke.assert_not_awaited()

    async def test_insufficient_evidence_stops_before_generation_tools(self) -> None:
        service, knowledge, quiz, summary, _ = make_service(
            KnowledgeToolResult(
                answer=NO_EVIDENCE_ANSWER,
                sources=(),
                citations=(),
                evidence=(),
            )
        )

        result = await service.chat(
            USER_ID,
            SESSION_ID,
            "帮我出题练习未知主题",
        )

        self.assertEqual(result.learning_action, "insufficient_evidence")
        self.assertIsNone(result.quiz)
        self.assertEqual(result.tools_used, (KNOWLEDGE_TOOL_NAME,))
        quiz.invoke.assert_not_awaited()
        summary.invoke.assert_not_awaited()
        json.dumps(asdict(result), ensure_ascii=False)

    async def test_tutor_session_and_history_survive_database_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "learning.sqlite3"
            first_store = await LearningDataStore.open(database_path)
            user = await first_store.create_user()
            session = await first_store.create_session(user.user_id, "Embedding")
            template, knowledge, _, summary, _ = make_service()
            first_graph = build_tutor_graph(first_store.checkpointer)
            first = TutorWorkflowService(
                graph=first_graph,
                tools=template._tools,
                learning_store=first_store,
            )
            await first.chat(
                user.user_id,
                session.session_id,
                "什么是 Embedding？",
            )
            await first_store.close()

            reopened_store = await LearningDataStore.open(database_path)
            try:
                reopened_graph = build_tutor_graph(reopened_store.checkpointer)
                reopened = TutorWorkflowService(
                    graph=reopened_graph,
                    tools=template._tools,
                    learning_store=reopened_store,
                )
                result = await reopened.chat(
                    user.user_id,
                    session.session_id,
                    "总结刚才的学习内容",
                )
                messages = await reopened_store.list_messages(
                    user.user_id,
                    session.session_id,
                )
                records = await reopened_store.list_learning_records(
                    user.user_id,
                    session_id=session.session_id,
                )
                snapshot = await reopened_graph.aget_state(
                    {"configurable": {"thread_id": session.session_id}}
                )
            finally:
                await reopened_store.close()

        self.assertEqual(result.intent, "summary")
        self.assertEqual(len(messages), 4)
        self.assertEqual(len(records), 2)
        self.assertEqual(snapshot.values["topic"], "Embedding")
        self.assertEqual(knowledge.invoke.await_count, 1)
        summary.invoke.assert_awaited_once()

    async def test_tool_failure_is_wrapped_without_internal_detail(self) -> None:
        service, _, _, _, _ = make_service(
            RuntimeError("BAILIAN_API_KEY=secret-value")
        )

        with self.assertRaisesRegex(TutorExecutionError, "知识检索工具") as raised:
            await service.chat(USER_ID, SESSION_ID, "Embedding 是什么？")

        self.assertNotIn("secret-value", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
