"""Stage 1 的 Evidence 与 Citation 数据契约。"""

import hashlib
import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from langchain_core.documents import Document


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PDF_SOURCE_PATTERN = re.compile(r"^(?P<filename>.+\.pdf) · 第 (?P<page>[1-9][0-9]*) 页$")
WEB_SOURCE_PATTERN = re.compile(
    r"^(?P<filename>.+\.md) · 网页：(?P<url>https?://.+)$"
)
CITATION_ID_PATTERN = re.compile(r"\[(S[0-9]+)\]")


def stable_sha256(*parts: object) -> str:
    """使用规范 JSON 数组生成无拼接歧义、可复现的 SHA-256。"""
    value = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def material_identity(
    *,
    source_type: str,
    filename: str,
    canonical_url: str | None = None,
) -> str:
    """资料逻辑身份：网页按规范 URL，本地文件按类型和文件名。"""
    locator = canonical_url if source_type == "web" and canonical_url else filename
    return stable_sha256("material-v1", source_type, locator)


def chunk_identity(
    *,
    material_id: str,
    chunk_index: int,
    content_hash: str,
    page: int | None = None,
    section: str | None = None,
) -> str:
    """Chunk 身份随位置或规范化内容变化，未变化的块可继续增量复用。"""
    return stable_sha256(
        "chunk-v1",
        material_id,
        page or "",
        section or "",
        chunk_index,
        content_hash,
    )


@dataclass(frozen=True)
class Evidence:
    """实际进入本次模型上下文、可被引用的一条结构化证据。"""

    context_id: str
    evidence_id: str
    material_id: str
    chunk_id: str
    source: str
    filename: str
    source_type: str
    page: int | None
    section: str | None
    chunk_index: int
    excerpt: str
    locator: str
    content_hash: str
    canonical_url: str | None = None


@dataclass(frozen=True)
class Citation:
    """模型返回的请求内 Citation ID 与服务端 Evidence 元数据的关联。"""

    citation_id: str
    evidence_id: str
    material_id: str
    chunk_id: str
    source: str
    filename: str
    page: int | None
    chunk_index: int
    excerpt: str
    locator: str


def _positive_int(value: object, default: int = 1) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def _optional_positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _legacy_source_metadata(source: str) -> dict[str, object]:
    """只为旧索引提供确定性适配；不从无法验证的文本猜测 section。"""
    pdf_match = PDF_SOURCE_PATTERN.fullmatch(source)
    if pdf_match:
        return {
            "filename": pdf_match.group("filename"),
            "source_type": "pdf",
            "page": int(pdf_match.group("page")),
            "canonical_url": None,
        }
    web_match = WEB_SOURCE_PATTERN.fullmatch(source)
    if web_match:
        return {
            "filename": web_match.group("filename"),
            "source_type": "web",
            "page": None,
            "canonical_url": web_match.group("url"),
        }
    suffix = source.rsplit(".", maxsplit=1)[-1].casefold() if "." in source else ""
    source_type = {"md": "markdown", "txt": "text"}.get(suffix, "unknown")
    return {
        "filename": source,
        "source_type": source_type,
        "page": None,
        "canonical_url": None,
    }


def evidence_from_document(document: Document, context_id: str) -> Evidence:
    """从受控 Document metadata 构建 Evidence；缺失字段仅按旧格式确定性回退。"""
    source = str(document.metadata.get("source") or "未知来源")
    legacy = _legacy_source_metadata(source)
    filename = str(document.metadata.get("filename") or legacy["filename"])
    source_type = str(document.metadata.get("source_type") or legacy["source_type"])
    page = _optional_positive_int(document.metadata.get("page"))
    if page is None:
        page = _optional_positive_int(legacy["page"])
    section_value = document.metadata.get("section")
    section = section_value if isinstance(section_value, str) and section_value else None
    chunk_index = _positive_int(document.metadata.get("chunk_index"))
    canonical_value = document.metadata.get("canonical_url")
    canonical_url = (
        canonical_value
        if isinstance(canonical_value, str) and canonical_value
        else legacy["canonical_url"]
    )
    content_hash_value = document.metadata.get("content_hash")
    content_hash = (
        content_hash_value
        if isinstance(content_hash_value, str)
        and SHA256_PATTERN.fullmatch(content_hash_value)
        else stable_sha256(document.page_content)
    )
    material_id_value = document.metadata.get("material_id")
    material_id = (
        material_id_value
        if isinstance(material_id_value, str)
        and SHA256_PATTERN.fullmatch(material_id_value)
        else material_identity(
            source_type=source_type,
            filename=filename,
            canonical_url=canonical_url if isinstance(canonical_url, str) else None,
        )
    )
    chunk_id_value = document.metadata.get("chunk_id")
    chunk_id = (
        chunk_id_value
        if isinstance(chunk_id_value, str) and SHA256_PATTERN.fullmatch(chunk_id_value)
        else chunk_identity(
            material_id=material_id,
            chunk_index=chunk_index,
            content_hash=content_hash,
            page=page,
            section=section,
        )
    )
    if isinstance(canonical_url, str) and canonical_url:
        separator = "&" if urlsplit(canonical_url).fragment else "#"
        locator = f"{canonical_url}{separator}chunk={chunk_index}"
    elif page is not None:
        locator = f"{filename}#page={page}&chunk={chunk_index}"
    else:
        locator = f"{filename}#chunk={chunk_index}"
    return Evidence(
        context_id=context_id,
        evidence_id=stable_sha256("evidence-v1", chunk_id),
        material_id=material_id,
        chunk_id=chunk_id,
        source=source,
        filename=filename,
        source_type=source_type,
        page=page,
        section=section,
        chunk_index=chunk_index,
        excerpt=document.page_content,
        locator=locator,
        content_hash=content_hash,
        canonical_url=canonical_url if isinstance(canonical_url, str) else None,
    )


def build_evidence_context(documents: list[Document]) -> tuple[Evidence, ...]:
    return tuple(
        evidence_from_document(document, f"S{position}")
        for position, document in enumerate(documents, start=1)
    )


def format_evidence_context(evidences: tuple[Evidence, ...]) -> str:
    return "\n\n".join(
        f"[{evidence.context_id}]\n{evidence.excerpt}" for evidence in evidences
    )


def resolve_citations(
    answer: str,
    evidences: tuple[Evidence, ...],
) -> tuple[tuple[Citation, ...], tuple[str, ...]]:
    """只接受本次 Evidence Map 内的 ID，并由服务端回填 Citation 元数据。"""
    evidence_map = {evidence.context_id: evidence for evidence in evidences}
    cited_ids = tuple(dict.fromkeys(CITATION_ID_PATTERN.findall(answer)))
    invalid_ids = tuple(item_id for item_id in cited_ids if item_id not in evidence_map)
    citations = []
    for citation_id in cited_ids:
        evidence = evidence_map.get(citation_id)
        if evidence is None:
            continue
        citations.append(
            Citation(
                citation_id=citation_id,
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
        )
    return tuple(citations), invalid_ids
