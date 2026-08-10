import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.evaluate_rag import main
from app.langchain_rag import NO_EVIDENCE_TOKEN
from app.rag_evaluation import (
    DEFAULT_EVALUATION_PATH,
    EvaluationDataError,
    EvaluationReportWriteError,
    ExpectedSource,
    RAGEvaluationCase,
    RAGEvaluationResult,
    build_evaluation_report,
    evaluate_case,
    find_missing_index_sources,
    load_evaluation_cases,
    write_evaluation_report,
)


def write_cases(path: Path, cases: list[dict]) -> None:
    path.write_text(
        json.dumps({"version": 1, "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )


def valid_case(**overrides: object) -> dict:
    case = {
        "id": "rag_chunking",
        "category": "concept",
        "question": "为什么需要切分资料？",
        "expected_sources": [{"source": "rag.md", "chunk_index": 2}],
        "required_answer_terms": ["太长", "相关"],
        "should_refuse": False,
    }
    case.update(overrides)
    return case


def retrieval_parameters() -> dict[str, int | float]:
    return {
        "retrieval_limit": 3,
        "candidate_limit": 10,
        "relevance_threshold": 0.25,
        "vector_weight": 0.8,
        "keyword_weight": 0.2,
        "adjacent_window": 2,
        "context_limit": 8,
    }


class RAGEvaluationTests(unittest.TestCase):
    def test_default_evaluation_dataset_is_valid(self) -> None:
        cases = load_evaluation_cases(DEFAULT_EVALUATION_PATH)

        self.assertEqual(len(cases), 10)
        self.assertEqual(len({case.case_id for case in cases}), 10)

    def test_loads_valid_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            write_cases(path, [valid_case()])

            cases = load_evaluation_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "rag_chunking")
        self.assertEqual(cases[0].expected_sources[0].key, ("rag.md", 2))

    def test_rejects_duplicate_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            write_cases(path, [valid_case(), valid_case()])

            with self.assertRaisesRegex(EvaluationDataError, "重复案例 id"):
                load_evaluation_cases(path)

    def test_rejects_boolean_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(
                json.dumps({"version": True, "cases": [valid_case()]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(EvaluationDataError, "version=1"):
                load_evaluation_cases(path)

    def test_rejects_refusal_case_with_expected_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            write_cases(path, [valid_case(should_refuse=True)])

            with self.assertRaisesRegex(EvaluationDataError, "拒答案例"):
                load_evaluation_cases(path)

    def test_finds_sources_missing_from_current_index(self) -> None:
        cases = (
            RAGEvaluationCase(
                case_id="expected_source",
                category="concept",
                question="问题",
                expected_sources=(ExpectedSource("chapter.pdf", 2),),
                required_answer_terms=("答案",),
                should_refuse=False,
            ),
        )

        missing = find_missing_index_sources(
            cases,
            [{"source": "chapter.pdf", "chunk_index": 1}],
        )

        self.assertEqual(missing, ("[chapter.pdf · 第 2 段]",))

    def test_positive_case_passes_all_deterministic_checks(self) -> None:
        document = Document(
            page_content="资料太长时应切分，只检索相关内容。",
            metadata={"source": "rag.md", "chunk_index": 2},
        )
        case = RAGEvaluationCase(
            case_id="rag_chunking",
            category="concept",
            question="为什么需要切分？",
            expected_sources=(ExpectedSource("rag.md", 2),),
            required_answer_terms=("太长", "相关"),
            should_refuse=False,
        )
        retriever = Mock()
        retriever.invoke.return_value = [document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(
            content="资料太长时应切分，只检索相关内容。[rag.md · 第 2 段]"
        )

        result = evaluate_case(case, retriever, chat_model)

        self.assertTrue(result.passed)
        self.assertEqual(result.failures, ())

    def test_evaluation_accepts_normalized_formula_and_citation(self) -> None:
        source = "第八章.pdf · 第 20 页"
        document = Document(
            page_content="逆威沙特分布用于协方差矩阵先验。",
            metadata={"source": source, "chunk_index": 4},
        )
        case = RAGEvaluationCase(
            case_id="inverse_wishart",
            category="pdf_concept",
            question="逆威沙特分布是什么？",
            expected_sources=(ExpectedSource(source, 4),),
            required_answer_terms=("逆威沙特", "先验"),
            should_refuse=False,
        )
        retriever = Mock()
        retriever.invoke.return_value = [document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(
            content=(
                r"逆威沙特分布满足 $M \sim \mathcal{W}^{-1}_p(m, \Sigma^{-1})$，"
                "常用作协方差矩阵先验"
                "（第八章.pdf · 第 20 页 · 第 4 段）。"
            )
        )

        result = evaluate_case(case, retriever, chat_model)

        self.assertTrue(result.passed)
        self.assertIn("[第八章.pdf · 第 20 页 · 第 4 段]", result.answer)
        self.assertNotIn("\\", result.answer)

    def test_cauchy_evaluation_accepts_observed_greek_latex_output(self) -> None:
        source = "第八章.pdf · 第 3 页"
        documents = [
            Document(
                page_content="柯西分布具有厚尾性质，均值和方差不存在。",
                metadata={"source": source, "chunk_index": 3},
            ),
            Document(
                page_content="特征函数为 ϕ(t)=e^(iμt-γ|t|)。",
                metadata={"source": source, "chunk_index": 4},
            ),
        ]
        case = RAGEvaluationCase(
            case_id="cauchy_properties",
            category="pdf_concept",
            question="柯西分布有哪些主要特点？",
            expected_sources=(ExpectedSource(source, 3),),
            required_answer_terms=("厚尾", "均值", "方差"),
            should_refuse=False,
        )
        retriever = Mock()
        retriever.invoke.return_value = documents
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(
            content=(
                "柯西分布具有厚尾性质，均值和方差不存在。"
                f"[{source} · 第 3 段]\n"
                r"特征函数为 $\phi(t)=e^{i\mu t-\gamma|t|}$。"
                f"[{source} · 第 4 段]"
            )
        )

        result = evaluate_case(case, retriever, chat_model)

        self.assertTrue(result.passed)
        self.assertIn("φ(t)=e^(iμ t-γ|t|)", result.answer)
        self.assertNotIn("\\", result.answer)

    def test_positive_case_reports_missing_source_term_and_citation(self) -> None:
        document = Document(
            page_content="其他内容。",
            metadata={"source": "other.md", "chunk_index": 1},
        )
        case = RAGEvaluationCase(
            case_id="rag_chunking",
            category="concept",
            question="为什么需要切分？",
            expected_sources=(ExpectedSource("rag.md", 2),),
            required_answer_terms=("相关",),
            should_refuse=False,
        )
        retriever = Mock()
        retriever.invoke.return_value = [document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(content="答案不完整。")

        result = evaluate_case(case, retriever, chat_model)

        self.assertFalse(result.passed)
        self.assertTrue(any("未检索到预期来源" in item for item in result.failures))
        self.assertTrue(any("未引用预期来源" in item for item in result.failures))
        self.assertTrue(any("缺少要点" in item for item in result.failures))

    def test_invalid_retrieved_metadata_becomes_a_case_failure(self) -> None:
        document = Document(
            page_content="相关内容。",
            metadata={"source": "rag.md", "chunk_index": "第二段"},
        )
        case = RAGEvaluationCase(
            case_id="rag_chunking",
            category="concept",
            question="为什么需要切分？",
            expected_sources=(ExpectedSource("rag.md", 2),),
            required_answer_terms=("相关",),
            should_refuse=False,
        )
        retriever = Mock()
        retriever.invoke.return_value = [document]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(content="相关内容。")

        result = evaluate_case(case, retriever, chat_model)

        self.assertFalse(result.passed)
        self.assertEqual(result.retrieved_sources, (("rag.md", 0),))
        self.assertTrue(any("未检索到预期来源" in item for item in result.failures))

    def test_refusal_case_passes(self) -> None:
        case = RAGEvaluationCase(
            case_id="out_of_scope",
            category="no_evidence",
            question="资料外问题",
            expected_sources=(),
            required_answer_terms=(),
            should_refuse=True,
        )
        retriever = Mock()
        retriever.invoke.return_value = [
            Document(
                page_content="不相关资料",
                metadata={"source": "rag.md", "chunk_index": 1},
            )
        ]
        chat_model = Mock()
        chat_model.invoke.return_value = AIMessage(content=NO_EVIDENCE_TOKEN)

        result = evaluate_case(case, retriever, chat_model)

        self.assertTrue(result.passed)
        self.assertEqual(result.retrieved_sources, ())

    def test_builds_complete_report_without_sensitive_configuration(self) -> None:
        passed_case = RAGEvaluationCase(
            case_id="rag_chunking",
            category="concept",
            question="为什么需要切分资料？",
            expected_sources=(ExpectedSource("rag.md", 2),),
            required_answer_terms=("相关",),
            should_refuse=False,
        )
        failed_case = RAGEvaluationCase(
            case_id="failed_case",
            category="regression",
            question="失败案例",
            expected_sources=(ExpectedSource("rag.md", 3),),
            required_answer_terms=("缺失",),
            should_refuse=False,
        )
        results = (
            RAGEvaluationResult(
                case=passed_case,
                answer="只检索相关内容。[rag.md · 第 2 段]",
                retrieved_sources=(("rag.md", 2),),
                elapsed_ms=120,
                failures=(),
            ),
            RAGEvaluationResult(
                case=failed_case,
                answer="回答不完整。",
                retrieved_sources=(("rag.md", 1),),
                elapsed_ms=80,
                failures=("未检索到预期来源。",),
            ),
        )
        started_at = datetime(2026, 8, 7, 8, 0, tzinfo=UTC)

        report = build_evaluation_report(
            results,
            evaluation_path=DEFAULT_EVALUATION_PATH,
            retrieval_parameters={
                **retrieval_parameters(),
                "api_key": "should-not-be-recorded",
                "base_url": "https://example.com/private?token=secret",
            },
            embedding_model="text-embedding-v4",
            chat_model="qwen-plus",
            started_at=started_at,
            completed_at=started_at + timedelta(milliseconds=1500),
        )

        self.assertEqual(report["report_schema_version"], 1)
        self.assertEqual(report["run"]["duration_ms"], 1500)
        self.assertEqual(report["evaluation_dataset"]["version"], 1)
        self.assertEqual(len(report["evaluation_dataset"]["sha256"]), 64)
        self.assertEqual(report["summary"]["total_cases"], 2)
        self.assertEqual(report["summary"]["passed_cases"], 1)
        self.assertEqual(report["summary"]["failed_cases"], 1)
        self.assertEqual(report["summary"]["pass_rate_percent"], 50.0)
        self.assertEqual(report["cases"][1]["failures"], ["未检索到预期来源。"])
        self.assertFalse(report["cases"][1]["passed"])
        self.assertEqual(
            report["cases"][0]["retrieved_sources"],
            [{"source": "rag.md", "chunk_index": 2}],
        )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("base_url", serialized)
        self.assertNotIn("should-not-be-recorded", serialized)
        self.assertNotIn("token=secret", serialized)

    def test_writes_unique_utf8_reports_without_overwriting(self) -> None:
        report = {"summary": {"passed_cases": 1}, "answer": "中文公式 Σ"}

        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "results"
            first_path = write_evaluation_report(report, output_directory)
            first_content = first_path.read_text(encoding="utf-8")
            second_path = write_evaluation_report(report, output_directory)

            self.assertNotEqual(first_path, second_path)
            self.assertEqual(first_path.read_text(encoding="utf-8"), first_content)
            self.assertIn("中文公式 Σ", first_content)
            self.assertNotIn(r"\u4e2d", first_content)
            self.assertEqual(json.loads(first_content), report)

    def test_filename_collision_preserves_existing_report(self) -> None:
        fixed_datetime = Mock()
        fixed_datetime.now.return_value.strftime.return_value = (
            "20260807T090000000000Z"
        )
        fixed_uuid = Mock(hex="12345678abcdef")

        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            existing_path = output_directory / (
                "rag-evaluation-20260807T090000000000Z-12345678.json"
            )
            existing_path.write_text("原有报告", encoding="utf-8")

            with (
                patch("app.rag_evaluation.datetime", fixed_datetime),
                patch("app.rag_evaluation.uuid4", return_value=fixed_uuid),
                self.assertRaisesRegex(EvaluationReportWriteError, "无法保存评测报告"),
            ):
                write_evaluation_report({"summary": {}}, output_directory)

            self.assertEqual(existing_path.read_text(encoding="utf-8"), "原有报告")

    def test_reports_clear_error_when_output_directory_cannot_be_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blocking_file = Path(directory) / "not-a-directory"
            blocking_file.write_text("占位", encoding="utf-8")

            with self.assertRaisesRegex(
                EvaluationReportWriteError,
                "无法保存评测报告",
            ):
                write_evaluation_report({"summary": {}}, blocking_file)

    def test_cli_saves_failed_cases_and_keeps_failure_exit_code(self) -> None:
        case = RAGEvaluationCase(
            case_id="failed_case",
            category="regression",
            question="为什么失败？",
            expected_sources=(ExpectedSource("rag.md", 2),),
            required_answer_terms=("答案",),
            should_refuse=False,
        )
        failed_result = RAGEvaluationResult(
            case=case,
            answer="不完整回答。",
            retrieved_sources=(("rag.md", 1),),
            elapsed_ms=25,
            failures=("未检索到预期来源。",),
        )
        vector_store = Mock()
        vector_store.get.return_value = {
            "ids": ["record-id"],
            "metadatas": [{"source": "rag.md", "chunk_index": 2}],
        }
        embedding_config = Mock(model="embedding-model")
        chat_config = Mock(model="chat-model")

        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "results"
            with (
                patch("app.evaluate_rag.load_evaluation_cases", return_value=(case,)),
                patch(
                    "app.evaluate_rag.EmbeddingConfig.from_environment",
                    return_value=embedding_config,
                ),
                patch("app.evaluate_rag.create_langchain_embeddings", return_value=Mock()),
                patch("app.evaluate_rag.open_vector_store", return_value=vector_store),
                patch("app.evaluate_rag.HybridRetriever", return_value=Mock()),
                patch(
                    "app.evaluate_rag.ChatConfig.from_environment",
                    return_value=chat_config,
                ),
                patch("app.evaluate_rag.create_langchain_chat_model", return_value=Mock()),
                patch(
                    "app.evaluate_rag.evaluate_cases",
                    return_value=(failed_result,),
                ),
                patch("builtins.print"),
            ):
                exit_code = main(
                    [
                        "--confirm-api-cost",
                        "--results-dir",
                        str(output_directory),
                    ]
                )

            reports = list(output_directory.glob("*.json"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(len(reports), 1)
            saved_report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(saved_report["summary"]["failed_cases"], 1)
            self.assertEqual(
                saved_report["cases"][0]["failures"],
                ["未检索到预期来源。"],
            )
            self.assertEqual(saved_report["models"]["embedding"], "embedding-model")
            self.assertEqual(saved_report["models"]["chat"], "chat-model")

    def test_cli_requires_explicit_api_cost_confirmation(self) -> None:
        output = Mock()

        with patch("builtins.print", output):
            exit_code = main([])

        self.assertEqual(exit_code, 2)
        printed = "\n".join(call.args[0] for call in output.call_args_list)
        self.assertIn("未执行评测", printed)
        self.assertIn("--confirm-api-cost", printed)


if __name__ == "__main__":
    unittest.main()
