import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from langgraph.checkpoint.memory import InMemorySaver

from app.agent_service import AgentExecutionError, AgentResult, AgentService
from app.study_workflow import (
    WORKFLOW_MAX_PROGRESS_ENTRIES,
    StudyWorkflowConflictError,
    StudyWorkflowNotFoundError,
    create_study_workflow_service,
    open_sqlite_study_workflow_service,
)


WORKFLOW_ID = "11111111-1111-4111-8111-111111111111"
USER_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OTHER_USER_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def make_agent(*results: AgentResult | Exception) -> Mock:
    agent = Mock(spec=AgentService)
    agent.ask = AsyncMock(side_effect=list(results))
    return agent


def grounded_result() -> AgentResult:
    return AgentResult(
        answer="RAG 需要先检索资料，再基于证据回答。",
        sources=("[rag.md · 第 2 段]",),
        tools_used=("answer_from_materials",),
    )


class StudyWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_calls_bounded_agent_and_pauses_for_approval(self) -> None:
        agent = make_agent(grounded_result())
        service = create_study_workflow_service(InMemorySaver())

        result = await service.start(
            USER_ID,
            WORKFLOW_ID,
            "理解 RAG 的证据约束",
            agent_service=agent,
        )

        self.assertEqual(result.status, "awaiting_confirmation")
        self.assertTrue(result.approval_required)
        self.assertEqual(result.sources, ("[rag.md · 第 2 段]",))
        self.assertEqual(len(result.tasks), 3)
        self.assertTrue(all(task["status"] == "pending" for task in result.tasks))
        self.assertIn("evidence:grounded", result.route_trace)
        self.assertIn("plan:drafted", result.route_trace)
        agent.ask.assert_awaited_once()
        prompt = agent.ask.await_args.args[0]
        self.assertIn("理解 RAG 的证据约束", prompt)
        self.assertIn("仅依据已索引资料", prompt)

    async def test_approval_resumes_same_thread_without_another_agent_call(self) -> None:
        checkpointer = InMemorySaver()
        agent = make_agent(grounded_result())
        first_service = create_study_workflow_service(checkpointer)
        await first_service.start(
            USER_ID,
            WORKFLOW_ID,
            "学习混合检索",
            agent_service=agent,
        )
        resumed_service = create_study_workflow_service(checkpointer)

        result = await resumed_service.confirm(USER_ID, WORKFLOW_ID, "approve")

        self.assertEqual(result.status, "in_progress")
        self.assertFalse(result.approval_required)
        self.assertEqual(result.tasks[0]["status"], "in_progress")
        self.assertIn("approval:approve", result.route_trace)
        self.assertIn("plan:activated", result.route_trace)
        self.assertEqual(agent.ask.await_count, 1)

    async def test_rejection_ends_without_progress_or_paid_call(self) -> None:
        agent = make_agent(grounded_result())
        service = create_study_workflow_service(InMemorySaver())
        await service.start(
            USER_ID,
            WORKFLOW_ID,
            "学习来源核对",
            agent_service=agent,
        )

        result = await service.confirm(USER_ID, WORKFLOW_ID, "reject")

        self.assertEqual(result.status, "rejected")
        self.assertIn("plan:rejected", result.route_trace)
        with self.assertRaisesRegex(StudyWorkflowConflictError, "不能更新"):
            await service.record_progress(
                USER_ID,
                WORKFLOW_ID,
                "不应写入",
                complete_current_task=True,
            )
        self.assertEqual(agent.ask.await_count, 1)

    async def test_progress_advances_tasks_and_creates_final_review(self) -> None:
        service = create_study_workflow_service(InMemorySaver())
        await service.start(
            USER_ID,
            WORKFLOW_ID,
            "学习相邻块扩展",
            agent_service=make_agent(grounded_result()),
        )
        await service.confirm(USER_ID, WORKFLOW_ID, "approve")

        first = await service.record_progress(
            USER_ID,
            WORKFLOW_ID,
            "已核对资料证据",
            complete_current_task=True,
        )
        second = await service.record_progress(
            USER_ID,
            WORKFLOW_ID,
            "已用自己的话解释",
            complete_current_task=True,
        )
        final = await service.record_progress(
            USER_ID,
            WORKFLOW_ID,
            "已完成练习并记录疑问",
            complete_current_task=True,
        )

        self.assertEqual(first.tasks[1]["status"], "in_progress")
        self.assertEqual(second.tasks[2]["status"], "in_progress")
        self.assertEqual(final.status, "completed")
        self.assertTrue(all(task["status"] == "completed" for task in final.tasks))
        self.assertEqual(len(final.progress_history), 3)
        self.assertIn("已完成 3 项学习任务", final.review)
        self.assertEqual(final.route_trace[-1], "review:completed")
        with self.assertRaisesRegex(StudyWorkflowConflictError, "不能更新"):
            await service.record_progress(
                USER_ID,
                WORKFLOW_ID,
                "完成后不能继续写",
                complete_current_task=False,
            )

    async def test_agent_failure_requires_explicit_single_manual_retry(self) -> None:
        agent = make_agent(
            AgentExecutionError("secret-value"),
            grounded_result(),
        )
        service = create_study_workflow_service(InMemorySaver())

        failed = await service.start(
            USER_ID,
            WORKFLOW_ID,
            "学习失败恢复",
            agent_service=agent,
        )

        self.assertEqual(failed.status, "failed")
        self.assertNotIn("secret-value", failed.error)
        self.assertEqual(agent.ask.await_count, 1)

        retried = await service.retry(USER_ID, WORKFLOW_ID, agent_service=agent)

        self.assertEqual(retried.status, "awaiting_confirmation")
        self.assertEqual(retried.retry_count, 1)
        self.assertEqual(agent.ask.await_count, 2)
        with self.assertRaisesRegex(StudyWorkflowConflictError, "不能再次重试"):
            await service.retry(USER_ID, WORKFLOW_ID, agent_service=agent)

    async def test_invalid_goal_stops_before_agent_and_is_not_retryable(self) -> None:
        agent = make_agent(grounded_result())
        service = create_study_workflow_service(InMemorySaver())

        result = await service.start(
            USER_ID,
            WORKFLOW_ID,
            "   ",
            agent_service=agent,
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("目标无效", result.error)
        agent.ask.assert_not_awaited()
        with self.assertRaisesRegex(StudyWorkflowConflictError, "不能再次重试"):
            await service.retry(USER_ID, WORKFLOW_ID, agent_service=agent)

    async def test_progress_history_has_a_bounded_entry_count(self) -> None:
        service = create_study_workflow_service(InMemorySaver())
        await service.start(
            USER_ID,
            WORKFLOW_ID,
            "学习进度边界",
            agent_service=make_agent(grounded_result()),
        )
        await service.confirm(USER_ID, WORKFLOW_ID, "approve")
        for index in range(WORKFLOW_MAX_PROGRESS_ENTRIES):
            await service.record_progress(
                USER_ID,
                WORKFLOW_ID,
                f"记录 {index}",
                complete_current_task=False,
            )

        with self.assertRaisesRegex(StudyWorkflowConflictError, "达到上限"):
            await service.record_progress(
                USER_ID,
                WORKFLOW_ID,
                "超出上限",
                complete_current_task=False,
            )

    async def test_progress_note_is_bounded_at_the_service_boundary(self) -> None:
        service = create_study_workflow_service(InMemorySaver())
        await service.start(
            USER_ID,
            WORKFLOW_ID,
            "学习进度输入边界",
            agent_service=make_agent(grounded_result()),
        )
        await service.confirm(USER_ID, WORKFLOW_ID, "approve")

        with self.assertRaisesRegex(StudyWorkflowConflictError, "进度记录无效"):
            await service.record_progress(
                USER_ID,
                WORKFLOW_ID,
                "x" * 1001,
                complete_current_task=False,
            )

        restored = await service.get(USER_ID, WORKFLOW_ID)
        self.assertEqual(restored.progress_history, ())

    async def test_insufficient_evidence_creates_material_collection_task(self) -> None:
        no_evidence = AgentResult(
            answer="现有资料不足，无法回答。",
            sources=(),
            tools_used=("answer_from_materials",),
        )
        service = create_study_workflow_service(InMemorySaver())

        result = await service.start(
            USER_ID,
            WORKFLOW_ID,
            "学习未知主题",
            agent_service=make_agent(no_evidence),
        )

        self.assertIn("补充", result.tasks[0]["title"])
        self.assertIn("evidence:insufficient", result.route_trace)

    async def test_workflow_checkpoint_is_scoped_to_its_owner(self) -> None:
        service = create_study_workflow_service(InMemorySaver())
        await service.start(
            USER_ID,
            WORKFLOW_ID,
            "学习工作流归属",
            agent_service=make_agent(grounded_result()),
        )

        with self.assertRaises(StudyWorkflowNotFoundError):
            await service.get(OTHER_USER_ID, WORKFLOW_ID)
        with self.assertRaises(StudyWorkflowNotFoundError):
            await service.confirm(OTHER_USER_ID, WORKFLOW_ID, "approve")

        owner_result = await service.get(USER_ID, WORKFLOW_ID)
        self.assertEqual(owner_result.status, "awaiting_confirmation")

    async def test_sqlite_checkpoint_survives_service_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "workflow.sqlite3"
            first = await open_sqlite_study_workflow_service(database_path)
            await first.start(
                USER_ID,
                WORKFLOW_ID,
                "学习持久化检查点",
                agent_service=make_agent(grounded_result()),
            )
            await first.close()

            reopened = await open_sqlite_study_workflow_service(database_path)
            try:
                restored = await reopened.get(USER_ID, WORKFLOW_ID)
                resumed = await reopened.confirm(USER_ID, WORKFLOW_ID, "approve")
            finally:
                await reopened.close()

        self.assertEqual(restored.status, "awaiting_confirmation")
        self.assertEqual(resumed.status, "in_progress")
        self.assertEqual(resumed.tasks[0]["status"], "in_progress")

    async def test_delete_removes_persisted_thread(self) -> None:
        service = create_study_workflow_service(InMemorySaver())
        await service.start(
            USER_ID,
            WORKFLOW_ID,
            "学习隐私删除",
            agent_service=make_agent(grounded_result()),
        )

        await service.delete(USER_ID, WORKFLOW_ID)

        with self.assertRaises(StudyWorkflowNotFoundError):
            await service.get(USER_ID, WORKFLOW_ID)


if __name__ == "__main__":
    unittest.main()
