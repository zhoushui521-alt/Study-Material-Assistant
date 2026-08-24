import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from app.ask_langchain import ADJACENT_WINDOW, CONTEXT_LIMIT, RETRIEVAL_LIMIT
from app.chunk_documents import DocumentChunk
from app.hybrid_search import HybridRetriever, expand_adjacent_documents
from app.langchain_rag import (
    NO_EVIDENCE_ANSWER,
    NO_EVIDENCE_TOKEN,
    InvalidCitationError,
    LangChainRAGError,
    answer_with_retriever,
    create_rag_chain,
    normalize_answer_for_terminal,
)
from app.langchain_store import rebuild_vector_store
from app.observability import OBSERVABILITY_LOGGER_NAME, observation_context


class KeywordEmbeddings(Embeddings):
    """测试用 Embedding：包含 RAG 时返回横向量，否则返回纵向量。"""

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [1.0, 0.0] if "rag" in text.casefold() else [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LangChainRAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = Document(
            page_content="把资料切成较小的文本块后，可以只检索相关段落。",
            metadata={"source": "rag-intro.md", "chunk_index": 2},
        )

    def test_returns_answer_and_retrieved_sources(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(content="长资料应该先切分。[S1]")

        result = answer_with_retriever("资料太长怎么办？", retriever, chat_model)

        self.assertEqual(result.sources, (self.document,))
        self.assertIn("长资料应该先切分", result.answer)
        self.assertEqual([citation.citation_id for citation in result.citations], ["S1"])
        self.assertEqual(result.citations[0].filename, "rag-intro.md")
        prompt_messages = chat_model.invoke.call_args.args[0].to_messages()
        self.assertIn("[S1]", prompt_messages[1].content)
        self.assertNotIn("rag-intro.md", prompt_messages[1].content)
        self.assertIn(NO_EVIDENCE_TOKEN, prompt_messages[0].content)

    def test_trace_links_retrieval_and_llm_events_to_one_request(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        chat_model = Mock()
        chat_model.model_name = "qwen-test"
        chat_model.invoke.return_value = AIMessage(
            content="长资料应该先切分。[S1]",
            usage_metadata={
                "input_tokens": 8,
                "output_tokens": 4,
                "total_tokens": 12,
            },
        )

        with observation_context(request_id="request-rag", user_id="user-rag"):
            with self.assertLogs(
                OBSERVABILITY_LOGGER_NAME, level="INFO"
            ) as captured:
                answer_with_retriever("资料太长怎么办？", retriever, chat_model)

        payloads = [json.loads(record.getMessage()) for record in captured.records]
        self.assertEqual(
            [payload["event"] for payload in payloads],
            [
                "retrieval_started",
                "retrieval_completed",
                "llm_call_started",
                "llm_call_completed",
            ],
        )
        self.assertTrue(
            all(payload["request_id"] == "request-rag" for payload in payloads)
        )
        self.assertTrue(all(payload["user_id"] == "user-rag" for payload in payloads))
        self.assertEqual(payloads[1]["retrieved_count"], 1)
        self.assertFalse(payloads[1]["empty_retrieval"])
        self.assertEqual(payloads[-1]["total_tokens"], 12)
        self.assertNotIn(
            "资料太长怎么办",
            "\n".join(record.getMessage() for record in captured.records),
        )

    def test_builds_explicit_lcel_runnable_with_retrieval_and_branch(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(
            content="长资料应该先切分。[rag-intro.md · 第 2 段]"
        )

        chain = create_rag_chain(retriever, chat_model)
        result = chain.invoke("资料太长怎么办？")
        graph_node_names = {node.name for node in chain.get_graph().nodes.values()}

        self.assertIsInstance(chain, Runnable)
        self.assertIn("retrieve_documents", graph_node_names)
        self.assertIn("select_context_evidence", graph_node_names)
        self.assertIn("Branch", graph_node_names)
        self.assertEqual(result.sources, (self.document,))

    def test_selects_context_before_assigning_evidence_ids(self) -> None:
        removed = Document(
            page_content="noise",
            metadata={"source": "noise.md", "chunk_index": 1},
        )
        retriever = Mock()
        retriever.invoke.return_value = [removed, self.document]
        selector = Mock()
        selector.select.return_value = [self.document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(content="应该切分资料。[S1]")

        result = answer_with_retriever(
            "资料太长怎么办？",
            retriever,
            chat_model,
            context_selector=selector,
        )

        selector.select.assert_called_once_with(
            "资料太长怎么办？",
            [removed, self.document],
        )
        self.assertEqual(result.sources, (self.document,))
        prompt_messages = chat_model.invoke.call_args.args[0].to_messages()
        self.assertIn(self.document.page_content, prompt_messages[1].content)
        self.assertNotIn(removed.page_content, prompt_messages[1].content)

    def test_answer_retrieval_configuration_supports_cross_chunk_questions(self) -> None:
        self.assertEqual(RETRIEVAL_LIMIT, 3)
        self.assertEqual(ADJACENT_WINDOW, 2)
        self.assertEqual(CONTEXT_LIMIT, 8)

    def test_prompt_requires_focused_plain_text_and_current_evidence_ids(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(
            content="长资料应该先切分。[rag-intro.md · 第 2 段]"
        )

        answer_with_retriever("资料太长怎么办？", retriever, chat_model)

        prompt_messages = chat_model.invoke.call_args.args[0].to_messages()
        system_prompt = prompt_messages[0].content
        user_prompt = prompt_messages[1].content
        self.assertIn("只输出一次最终答案", system_prompt)
        self.assertIn("不得重复作答", system_prompt)
        self.assertIn("2 至 4 条最相关结论", system_prompt)
        self.assertIn("问题包含多个子问题时", system_prompt)
        self.assertIn("没有直接证据的部分", system_prompt)
        self.assertIn("不得用模型知识补全", system_prompt)
        self.assertIn("不使用 Markdown 粗体、LaTeX 或代码围栏", system_prompt)
        self.assertIn("禁止输出“$”或反斜杠开头的 LaTeX 命令", system_prompt)
        self.assertIn("Evidence ID", system_prompt)
        self.assertIn("只能逐字复制可用资料中实际存在的 [S#]", system_prompt)
        self.assertIn("不得自行生成文件名、页码、链接、摘录", system_prompt)
        self.assertIn("请只返回符合系统规则的最终输出", user_prompt)

    def test_rejects_citation_id_not_present_in_current_context(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(content="答案。[S9]")

        with self.assertRaisesRegex(InvalidCitationError, "Citation ID"):
            answer_with_retriever("资料太长怎么办？", retriever, chat_model)

    def test_converts_observed_latex_formulas_to_readable_text(self) -> None:
        answer = (
            r"期望为 $ E[n_i] = N a_i $；" "\n"
            r"方差为 $ \text{Var}(n_i) = N a_i (1 - a_i) $；" "\n"
            r"协方差为 $ \text{Cov}(n_i, n_j) = -N a_i a_j $ "
            r"($ i \neq j $)。"
        )

        normalized = normalize_answer_for_terminal(answer)

        self.assertEqual(
            normalized,
            "期望为 E[n_i] = N a_i；\n"
            "方差为 Var(n_i) = N a_i (1 - a_i)；\n"
            "协方差为 Cov(n_i, n_j) = -N a_i a_j (i ≠ j)。",
        )
        self.assertNotIn("$", normalized)
        self.assertNotIn("\\", normalized)

    def test_converts_fraction_and_symbols_to_readable_text(self) -> None:
        answer = (
            r"$ f(x) \sim \frac{1}{\pi x^2} $ 当 "
            r"$ |x| \to \infty $，且 $\sqrt{\gamma}$ 有定义。"
        )

        normalized = normalize_answer_for_terminal(answer)

        self.assertEqual(
            normalized,
            "f(x) ~ (1)/(π x^2) 当 |x| → ∞，且 √(γ) 有定义。",
        )

    def test_preserves_plain_text_and_converts_arrow_commands_exactly(self) -> None:
        plain_text = r"价格为 $100，路径是 C:\Users，集合为 {1, 2}。"
        arrow_formula = r"$ a \leftarrow b \rightarrow c $"

        self.assertEqual(normalize_answer_for_terminal(plain_text), plain_text)
        self.assertEqual(
            normalize_answer_for_terminal(arrow_formula),
            "a ← b → c",
        )

    def test_converts_inverse_wishart_latex_commands(self) -> None:
        answer = (
            r"若 $W \sim \mathcal{W}_p(m, \Sigma)$，则 "
            r"$M = W^{-1} \sim \mathcal{W}^{-1}_p(m, \Sigma^{-1})$，"
            r"且 $E[M] = \Psi/(m-p-1)$。"
        )

        normalized = normalize_answer_for_terminal(answer)

        self.assertEqual(
            normalized,
            "若 W ~ W_p(m, Σ)，则 M = W^-1 ~ W^-1_p(m, Σ^-1)，"
            "且 E[M] = Ψ/(m-p-1)。",
        )
        self.assertNotIn("\\", normalized)

    def test_converts_cauchy_greek_letters_and_preserves_grouped_exponent(
        self,
    ) -> None:
        answer = (
            r"特征函数为 $\phi(t) = e^{i\mu t - \gamma|t|}$，"
            r"并区分 $\varphi$ 与 $\phi$。"
        )

        normalized = normalize_answer_for_terminal(answer)

        self.assertEqual(
            normalized,
            "特征函数为 φ(t) = e^(iμ t - γ|t|)，并区分 ϕ 与 φ。",
        )
        self.assertNotIn("\\", normalized)

    def test_normalizes_parenthesized_source_references(self) -> None:
        answer = (
            "厚尾特性（第八章.pdf · 第 3 页 · 第 3 段）\n"
            "分布关系（第八章.pdf · 第 4 页 · 第 1 段；"
            "第八章.pdf · 第 1 页 · 第 2 段）\n"
            "条件 (i ≠ j) 保持普通括号。"
        )

        normalized = normalize_answer_for_terminal(answer)

        self.assertEqual(
            normalized,
            "厚尾特性[第八章.pdf · 第 3 页 · 第 3 段]\n"
            "分布关系[第八章.pdf · 第 4 页 · 第 1 段]；"
            "[第八章.pdf · 第 1 页 · 第 2 段]\n"
            "条件 (i ≠ j) 保持普通括号。",
        )

    def test_answer_output_normalizes_latex_and_preserves_source_label(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(
            content=(
                r"方差为 $\operatorname{Var}(n_i)=N a_i(1-a_i)$。"
                "\n[rag-intro.md · 第 2 段]"
            )
        )

        result = answer_with_retriever("方差是什么？", retriever, chat_model)

        self.assertEqual(
            result.answer,
            "方差为 Var(n_i)=N a_i(1-a_i)。\n[rag-intro.md · 第 2 段]",
        )
        self.assertEqual(result.sources, (self.document,))

    def test_discards_unsupported_answer_when_model_admits_insufficient_evidence(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(
            content=(
                "MySQL 的事务隔离级别包括 READ UNCOMMITTED、READ COMMITTED、"
                f"REPEATABLE READ 和 SERIALIZABLE。\n\n{NO_EVIDENCE_ANSWER}"
            )
        )

        result = answer_with_retriever("MySQL 的事务隔离级别有哪些？", retriever, chat_model)

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.sources, ())

    def test_maps_no_evidence_token_to_user_facing_refusal(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(content=NO_EVIDENCE_TOKEN)

        result = answer_with_retriever("MySQL 的事务隔离级别有哪些？", retriever, chat_model)

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.sources, ())

    def test_works_with_hybrid_chroma_retriever(self) -> None:
        chunks = [
            DocumentChunk("rag.md", 1, "上一段解释资料准备。"),
            DocumentChunk("rag.md", 2, "RAG 会先检索相关资料。"),
            DocumentChunk("rag.md", 3, "下一段解释答案生成。"),
            DocumentChunk("python.md", 1, "Python 函数可以复用代码。"),
        ]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(
            content="RAG 会先检索资料。[rag.md · 第 2 段]"
        )

        with tempfile.TemporaryDirectory() as directory:
            vector_store = rebuild_vector_store(chunks, KeywordEmbeddings(), Path(directory))
            try:
                retriever = HybridRetriever(
                    vector_store=vector_store,
                    limit=1,
                    relevance_threshold=0.25,
                    adjacent_window=1,
                    context_limit=8,
                )

                result = answer_with_retriever("RAG 是什么？", retriever, chat_model)

                self.assertEqual(
                    [document.metadata["chunk_index"] for document in result.sources],
                    [2, 1, 3],
                )
                self.assertTrue(
                    all(
                        document.metadata["source"] == "rag.md"
                        for document in result.sources
                    )
                )
                self.assertEqual(
                    [document.metadata["_context_role"] for document in result.sources],
                    ["seed", "adjacent", "adjacent"],
                )
            finally:
                vector_store.delete_collection()
                vector_store._client.close()

    def test_expands_adjacent_chunks_from_real_chroma_metadata(self) -> None:
        chunks = [
            DocumentChunk("第八章.pdf · 第 13 页", 1, "边缘分布为二项分布。"),
            DocumentChunk("第八章.pdf · 第 13 页", 2, "期望与方差。"),
            DocumentChunk("第八章.pdf · 第 13 页", 3, "不同类别计数的协方差。"),
            DocumentChunk("其他.pdf · 第 13 页", 2, "不应跨来源补充。"),
        ]
        seed = Document(
            page_content=chunks[1].content,
            metadata={"source": chunks[1].source, "chunk_index": chunks[1].index},
        )

        with tempfile.TemporaryDirectory() as directory:
            vector_store = rebuild_vector_store(
                chunks,
                KeywordEmbeddings(),
                Path(directory),
            )
            try:
                expanded = expand_adjacent_documents(
                    vector_store,
                    [seed],
                    adjacent_window=1,
                    context_limit=8,
                )

                self.assertEqual(
                    [document.metadata["chunk_index"] for document in expanded],
                    [2, 1, 3],
                )
                self.assertTrue(
                    all(
                        document.metadata["source"] == "第八章.pdf · 第 13 页"
                        for document in expanded
                    )
                )
                self.assertEqual(
                    [document.metadata["_context_role"] for document in expanded],
                    ["seed", "adjacent", "adjacent"],
                )
            finally:
                vector_store.delete_collection()
                vector_store._client.close()

    def test_refuses_without_retrieved_documents(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = []
        chat_model = Mock()

        result = answer_with_retriever("Python 怎么安装？", retriever, chat_model)

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.sources, ())
        chat_model.invoke.assert_not_called()

    def test_rejects_empty_question_before_retrieval(self) -> None:
        retriever = Mock()
        chat_model = Mock()

        with self.assertRaisesRegex(ValueError, "问题不能为空"):
            answer_with_retriever("  ", retriever, chat_model)

        retriever.invoke.assert_not_called()
        chat_model.invoke.assert_not_called()

    def test_reports_retrieval_failure(self) -> None:
        retriever = Mock()
        retriever.invoke.side_effect = RuntimeError("vector store unavailable")

        with self.assertRaisesRegex(LangChainRAGError, "检索资料失败") as context:
            answer_with_retriever("RAG 是什么？", retriever, Mock())

        self.assertNotIn("vector store unavailable", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    def test_reports_context_selection_failure(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        selector = Mock()
        selector.select.side_effect = RuntimeError("selector internals")

        with self.assertRaisesRegex(LangChainRAGError, "选择 LLM Context") as context:
            answer_with_retriever(
                "资料太长怎么办？",
                retriever,
                Mock(),
                context_selector=selector,
            )

        self.assertNotIn("selector internals", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    def test_reports_chat_model_failure(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        chat_model = Mock()
        chat_model.invoke.side_effect = RuntimeError("service unavailable")

        with self.assertRaisesRegex(LangChainRAGError, "调用 Chat 模型失败") as context:
            answer_with_retriever("RAG 是什么？", retriever, chat_model)

        self.assertNotIn("service unavailable", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    def test_reports_empty_chat_model_response(self) -> None:
        retriever = Mock()
        retriever.invoke.return_value = [self.document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(content="")

        with self.assertRaisesRegex(LangChainRAGError, "Chat 模型返回了空回答"):
            answer_with_retriever("RAG 是什么？", retriever, chat_model)


if __name__ == "__main__":
    unittest.main()
