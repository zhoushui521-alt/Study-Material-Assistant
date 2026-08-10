"""固定 RAG 评测集的数据校验与确定性验收逻辑。"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

if __package__:
    from app.config import PROJECT_ROOT
    from app.langchain_rag import NO_EVIDENCE_ANSWER, answer_with_retriever
else:
    from config import PROJECT_ROOT
    from langchain_rag import NO_EVIDENCE_ANSWER, answer_with_retriever


DEFAULT_EVALUATION_PATH = PROJECT_ROOT / "evaluation" / "rag_cases.json"
DEFAULT_EVALUATION_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
EVALUATION_REPORT_SCHEMA_VERSION = 1
CASE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
RETRIEVAL_PARAMETER_NAMES = (
    "retrieval_limit",
    "candidate_limit",
    "relevance_threshold",
    "vector_weight",
    "keyword_weight",
    "adjacent_window",
    "context_limit",
)


class EvaluationDataError(ValueError):
    """评测集 JSON 结构或字段不符合约定。"""


class EvaluationReportWriteError(RuntimeError):
    """结构化评测报告无法安全写入时抛出的可读错误。"""


@dataclass(frozen=True)
class ExpectedSource:
    """一个案例必须检索并引用的资料位置。"""

    source: str
    chunk_index: int

    @property
    def label(self) -> str:
        return f"[{self.source} · 第 {self.chunk_index} 段]"

    @property
    def key(self) -> tuple[str, int]:
        return self.source, self.chunk_index


@dataclass(frozen=True)
class RAGEvaluationCase:
    """一条固定问题及其可机械检查的预期结果。"""

    case_id: str
    category: str
    question: str
    expected_sources: tuple[ExpectedSource, ...]
    required_answer_terms: tuple[str, ...]
    should_refuse: bool


@dataclass(frozen=True)
class RAGEvaluationResult:
    """一次案例执行结果；failures 为空表示通过。"""

    case: RAGEvaluationCase
    answer: str
    retrieved_sources: tuple[tuple[str, int], ...]
    elapsed_ms: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def _require_non_empty_text(value: object, field: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDataError(f"案例 {case_id} 的 {field} 必须是非空字符串。")
    return value.strip()


def _parse_expected_sources(value: object, case_id: str) -> tuple[ExpectedSource, ...]:
    if not isinstance(value, list):
        raise EvaluationDataError(f"案例 {case_id} 的 expected_sources 必须是列表。")

    sources = []
    seen = set()
    for position, raw_source in enumerate(value, start=1):
        if not isinstance(raw_source, dict):
            raise EvaluationDataError(
                f"案例 {case_id} 的第 {position} 个 expected_sources 必须是对象。"
            )
        source = _require_non_empty_text(raw_source.get("source"), "source", case_id)
        chunk_index = raw_source.get("chunk_index")
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index <= 0
        ):
            raise EvaluationDataError(
                f"案例 {case_id} 的 chunk_index 必须是正整数。"
            )
        expected_source = ExpectedSource(source=source, chunk_index=chunk_index)
        if expected_source.key in seen:
            raise EvaluationDataError(f"案例 {case_id} 包含重复的 expected_sources。")
        seen.add(expected_source.key)
        sources.append(expected_source)
    return tuple(sources)


def _parse_required_terms(value: object, case_id: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvaluationDataError(
            f"案例 {case_id} 的 required_answer_terms 必须是列表。"
        )
    terms = tuple(
        _require_non_empty_text(term, "required_answer_terms", case_id)
        for term in value
    )
    if len(set(terms)) != len(terms):
        raise EvaluationDataError(f"案例 {case_id} 包含重复的 required_answer_terms。")
    return terms


def load_evaluation_cases(
    path: Path = DEFAULT_EVALUATION_PATH,
) -> tuple[RAGEvaluationCase, ...]:
    """读取并严格校验固定评测集。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EvaluationDataError(f"无法读取评测集 {path}：{error}") from error
    except json.JSONDecodeError as error:
        raise EvaluationDataError(f"评测集不是合法 JSON：{error}") from error

    version = payload.get("version") if isinstance(payload, dict) else None
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise EvaluationDataError("评测集根对象必须包含 version=1。")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationDataError("评测集 cases 必须是非空列表。")

    cases = []
    seen_ids = set()
    for position, raw_case in enumerate(raw_cases, start=1):
        placeholder_id = f"第 {position} 条"
        if not isinstance(raw_case, dict):
            raise EvaluationDataError(f"评测集的{placeholder_id}案例必须是对象。")
        case_id = _require_non_empty_text(raw_case.get("id"), "id", placeholder_id)
        if CASE_ID_PATTERN.fullmatch(case_id) is None:
            raise EvaluationDataError(
                f"案例 {case_id} 的 id 只能包含小写字母、数字、下划线和连字符。"
            )
        if case_id in seen_ids:
            raise EvaluationDataError(f"评测集包含重复案例 id：{case_id}。")
        seen_ids.add(case_id)

        category = _require_non_empty_text(raw_case.get("category"), "category", case_id)
        question = _require_non_empty_text(raw_case.get("question"), "question", case_id)
        expected_sources = _parse_expected_sources(
            raw_case.get("expected_sources"),
            case_id,
        )
        required_terms = _parse_required_terms(
            raw_case.get("required_answer_terms"),
            case_id,
        )
        should_refuse = raw_case.get("should_refuse")
        if not isinstance(should_refuse, bool):
            raise EvaluationDataError(f"案例 {case_id} 的 should_refuse 必须是布尔值。")
        if should_refuse and (expected_sources or required_terms):
            raise EvaluationDataError(
                f"拒答案例 {case_id} 不能声明预期来源或必含答案词。"
            )
        if not should_refuse and (not expected_sources or not required_terms):
            raise EvaluationDataError(
                f"非拒答案例 {case_id} 必须声明预期来源和必含答案词。"
            )

        cases.append(
            RAGEvaluationCase(
                case_id=case_id,
                category=category,
                question=question,
                expected_sources=expected_sources,
                required_answer_terms=required_terms,
                should_refuse=should_refuse,
            )
        )
    return tuple(cases)


def find_missing_index_sources(
    cases: tuple[RAGEvaluationCase, ...],
    metadatas: list[dict],
) -> tuple[str, ...]:
    """在调用 API 前检查评测集引用的资料位置是否仍存在于索引。"""
    available = {
        (metadata.get("source"), metadata.get("chunk_index"))
        for metadata in metadatas
        if isinstance(metadata, dict)
    }
    missing = {
        expected.label
        for case in cases
        for expected in case.expected_sources
        if expected.key not in available
    }
    return tuple(sorted(missing))


def evaluate_case(
    case: RAGEvaluationCase,
    retriever: BaseRetriever,
    chat_model: BaseChatModel,
) -> RAGEvaluationResult:
    """执行一个案例并检查拒答、来源、答案要点、引用和终端公式格式。"""
    started = perf_counter()
    try:
        rag_answer = answer_with_retriever(case.question, retriever, chat_model)
    except Exception as error:
        return RAGEvaluationResult(
            case=case,
            answer="",
            retrieved_sources=(),
            elapsed_ms=round((perf_counter() - started) * 1000),
            failures=(f"执行异常：{error}",),
        )

    elapsed_ms = round((perf_counter() - started) * 1000)
    retrieved_sources = []
    for document in rag_answer.sources:
        raw_chunk_index = document.metadata.get("chunk_index")
        chunk_index = (
            raw_chunk_index
            if isinstance(raw_chunk_index, int)
            and not isinstance(raw_chunk_index, bool)
            and raw_chunk_index > 0
            else 0
        )
        retrieved_sources.append(
            (str(document.metadata.get("source", "")), chunk_index)
        )
    retrieved_sources_tuple = tuple(retrieved_sources)
    failures = []

    if case.should_refuse:
        if rag_answer.answer != NO_EVIDENCE_ANSWER:
            failures.append("预期拒答，但模型返回了实质答案。")
        if retrieved_sources_tuple:
            failures.append("拒答结果不应对外保留候选来源。")
    else:
        if rag_answer.answer == NO_EVIDENCE_ANSWER:
            failures.append("资料有证据，但系统错误拒答。")

        retrieved_set = set(retrieved_sources_tuple)
        for expected in case.expected_sources:
            if expected.key not in retrieved_set:
                failures.append(f"未检索到预期来源：{expected.label}")
            if expected.label not in rag_answer.answer:
                failures.append(f"答案未引用预期来源：{expected.label}")

        normalized_answer = rag_answer.answer.casefold()
        for term in case.required_answer_terms:
            if term.casefold() not in normalized_answer:
                failures.append(f"答案缺少要点：{term}")

        if "$" in rag_answer.answer or re.search(r"\\[A-Za-z]+", rag_answer.answer):
            failures.append("答案仍包含未转换的 LaTeX 标记。")

    return RAGEvaluationResult(
        case=case,
        answer=rag_answer.answer,
        retrieved_sources=retrieved_sources_tuple,
        elapsed_ms=elapsed_ms,
        failures=tuple(failures),
    )


def evaluate_cases(
    cases: tuple[RAGEvaluationCase, ...],
    retriever: BaseRetriever,
    chat_model: BaseChatModel,
) -> tuple[RAGEvaluationResult, ...]:
    """按固定顺序执行全部案例，单个案例失败不会中断后续评测。"""
    return tuple(evaluate_case(case, retriever, chat_model) for case in cases)


def _format_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("评测报告时间必须包含时区。")
    return (
        value.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _evaluation_dataset_metadata(path: Path) -> dict[str, object]:
    try:
        raw_content = path.read_bytes()
        payload = json.loads(raw_content.decode("utf-8"))
    except OSError as error:
        raise EvaluationDataError(f"无法读取评测集 {path}：{error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationDataError(f"无法识别评测集 {path}：{error}") from error

    version = payload.get("version") if isinstance(payload, dict) else None
    if isinstance(version, bool) or not isinstance(version, int):
        raise EvaluationDataError("评测集缺少有效的整数 version。")
    return {
        "file": path.name,
        "version": version,
        "sha256": sha256(raw_content).hexdigest(),
    }


def build_evaluation_report(
    results: tuple[RAGEvaluationResult, ...],
    *,
    evaluation_path: Path,
    retrieval_parameters: Mapping[str, int | float],
    embedding_model: str,
    chat_model: str,
    started_at: datetime,
    completed_at: datetime,
) -> dict[str, object]:
    """把一次完整评测转换成不含密钥和服务地址的 JSON 可序列化报告。"""
    if completed_at < started_at:
        raise ValueError("评测完成时间不能早于开始时间。")

    missing_parameters = [
        name for name in RETRIEVAL_PARAMETER_NAMES if name not in retrieval_parameters
    ]
    if missing_parameters:
        raise ValueError(
            "评测报告缺少检索参数：" + "、".join(missing_parameters)
        )

    passed_count = sum(result.passed for result in results)
    total_count = len(results)
    return {
        "report_schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "run": {
            "started_at": _format_utc_timestamp(started_at),
            "completed_at": _format_utc_timestamp(completed_at),
            "duration_ms": round(
                (completed_at - started_at).total_seconds() * 1000
            ),
        },
        "evaluation_dataset": _evaluation_dataset_metadata(evaluation_path),
        "models": {
            "embedding": embedding_model,
            "chat": chat_model,
        },
        "retrieval_parameters": {
            name: retrieval_parameters[name] for name in RETRIEVAL_PARAMETER_NAMES
        },
        "summary": {
            "total_cases": total_count,
            "passed_cases": passed_count,
            "failed_cases": total_count - passed_count,
            "pass_rate_percent": (
                round(passed_count / total_count * 100, 2) if total_count else 0.0
            ),
        },
        "cases": [
            {
                "id": result.case.case_id,
                "category": result.case.category,
                "question": result.case.question,
                "elapsed_ms": result.elapsed_ms,
                "answer": result.answer,
                "retrieved_sources": [
                    {"source": source, "chunk_index": chunk_index}
                    for source, chunk_index in result.retrieved_sources
                ],
                "failures": list(result.failures),
                "passed": result.passed,
            }
            for result in results
        ],
    }


def write_evaluation_report(
    report: Mapping[str, object],
    output_directory: Path = DEFAULT_EVALUATION_RESULTS_DIR,
) -> Path:
    """以唯一文件名写入 UTF-8 JSON；绝不覆盖已有评测报告。"""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = output_directory / (
        f"rag-evaluation-{timestamp}-{uuid4().hex[:8]}.json"
    )
    created_report = False

    try:
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        output_directory.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8", newline="\n") as file:
            created_report = True
            file.write(serialized)
    except (OSError, TypeError, ValueError) as error:
        if created_report:
            try:
                report_path.unlink()
            except OSError:
                pass
        raise EvaluationReportWriteError(
            f"无法保存评测报告到 {output_directory}：{error}"
        ) from error
    return report_path
