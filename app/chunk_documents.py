"""RAG 第一步：读取资料，并将长文本切成小块。

现在不依赖任何 AI 框架，目的是先看清 RAG 的输入是什么。
"""

import base64
import binascii
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.utils.exceptions import PdfminerException

if __package__:
    from app.evidence import chunk_identity, material_identity, stable_sha256
else:
    from evidence import chunk_identity, material_identity, stable_sha256


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
SUPPORTED_DOCUMENT_SUFFIXES = {".txt", ".md", ".pdf"}
WEB_SOURCE_MARKER_PREFIX = "<!-- study-material-web-source-v1:"
WEB_SOURCE_MARKER_SUFFIX = " -->"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
WEB_SOURCE_MARKER_MAX_ENCODED_LENGTH = 4096
PARSER_VERSION = "study-material-parser-v1"
CLEANER_VERSION = "whitespace-cleaner-v1"
CHUNKER_VERSION = "fixed-character-180-v1"
METADATA_SCHEMA_VERSION = "1"
PDF_SOURCE_PATTERN = re.compile(r"^(?P<filename>.+\.pdf) · 第 (?P<page>[1-9][0-9]*) 页$")
WEB_SOURCE_PATTERN = re.compile(
    r"^(?P<filename>.+\.md) · 网页：(?P<url>https?://.+)$"
)


class DocumentLoadError(RuntimeError):
    """学习资料无法读取或没有可提取文本时抛出的可读错误。"""


@dataclass(frozen=True)
class Material:
    """一份完整资料的逻辑身份与当前内容版本。"""

    material_id: str
    source_type: str
    filename: str
    content_hash: str
    canonical_url: str | None = None


@dataclass(frozen=True)
class MaterialUnit:
    """Parser 产出的可切分单元；PDF 每页一个单元，其余资料通常一个。"""

    material: Material
    source: str
    text: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class DocumentChunk:
    """一小段可被后续检索的资料。"""

    source: str
    index: int
    content: str
    material_id: str = ""
    chunk_id: str = ""
    source_type: str = ""
    filename: str = ""
    page: int | None = None
    section: str | None = None
    material_content_hash: str = ""
    content_hash: str = ""
    canonical_url: str | None = None
    parser_version: str = PARSER_VERSION
    cleaner_version: str = CLEANER_VERSION
    chunker_version: str = CHUNKER_VERSION
    metadata_schema_version: str = METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """兼容教学脚本直接构造的旧三字段 Chunk，并补齐确定性身份。"""
        filename = self.filename
        source_type = self.source_type
        page = self.page
        canonical_url = self.canonical_url
        pdf_match = PDF_SOURCE_PATTERN.fullmatch(self.source)
        web_match = WEB_SOURCE_PATTERN.fullmatch(self.source)
        if not filename and pdf_match:
            filename = pdf_match.group("filename")
            source_type = source_type or "pdf"
            page = page or int(pdf_match.group("page"))
        elif not filename and web_match:
            filename = web_match.group("filename")
            source_type = source_type or "web"
            canonical_url = canonical_url or web_match.group("url")
        elif not filename:
            filename = self.source
        if not source_type:
            source_type = {
                ".md": "markdown",
                ".txt": "text",
                ".pdf": "pdf",
            }.get(Path(filename).suffix.casefold(), "unknown")
        content_hash = self.content_hash or stable_sha256(self.content)
        material_id = self.material_id or material_identity(
            source_type=source_type,
            filename=filename,
            canonical_url=canonical_url,
        )
        chunk_id = self.chunk_id or chunk_identity(
            material_id=material_id,
            chunk_index=self.index,
            content_hash=content_hash,
            page=page,
            section=self.section,
        )
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "canonical_url", canonical_url)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(
            self,
            "material_content_hash",
            self.material_content_hash or content_hash,
        )
        object.__setattr__(self, "material_id", material_id)
        object.__setattr__(self, "chunk_id", chunk_id)


@dataclass(frozen=True)
class WebSourceMetadata:
    """由网页预览流程写入 Markdown 首行的可验证来源元数据。"""

    canonical_url: str
    title: str
    crawled_at: str
    content_sha256: str


def _validate_web_source_metadata(metadata: WebSourceMetadata) -> None:
    if (
        not isinstance(metadata.canonical_url, str)
        or not isinstance(metadata.title, str)
        or not isinstance(metadata.crawled_at, str)
        or not isinstance(metadata.content_sha256, str)
        or len(metadata.crawled_at) > 80
        or SHA256_PATTERN.fullmatch(metadata.content_sha256) is None
    ):
        raise DocumentLoadError("网页资料来源元数据无效。")
    try:
        parsed = urlsplit(metadata.canonical_url)
        default_port = {"http": 80, "https": 443}.get(parsed.scheme)
        port = parsed.port or default_port
        crawled_at = datetime.fromisoformat(metadata.crawled_at.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise DocumentLoadError("网页资料来源元数据无效。") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or port != default_port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or metadata.canonical_url != metadata.canonical_url.strip()
        or "\\" in metadata.canonical_url
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in metadata.canonical_url
        )
        or len(metadata.canonical_url) > 2048
        or not metadata.title.strip()
        or len(metadata.title) > 200
        or crawled_at.tzinfo is None
    ):
        raise DocumentLoadError("网页资料来源元数据无效。")


def encode_web_source_marker(metadata: WebSourceMetadata) -> str:
    """把网页来源元数据编码为不会混入正文的单行 Markdown 标记。"""
    _validate_web_source_metadata(metadata)
    payload = json.dumps(
        asdict(metadata),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"{WEB_SOURCE_MARKER_PREFIX}{encoded}{WEB_SOURCE_MARKER_SUFFIX}"


def _extract_web_source(
    filename: str,
    text: str,
) -> tuple[str, str]:
    """识别本项目生成的网页 Markdown，并让来源标签保留原始 URL。"""
    first_line, separator, remaining = text.partition("\n")
    if not first_line.startswith(WEB_SOURCE_MARKER_PREFIX):
        return filename, text
    if not first_line.endswith(WEB_SOURCE_MARKER_SUFFIX):
        raise DocumentLoadError("网页资料来源元数据无效。")
    encoded = first_line[
        len(WEB_SOURCE_MARKER_PREFIX) : -len(WEB_SOURCE_MARKER_SUFFIX)
    ]
    if not encoded or len(encoded) > WEB_SOURCE_MARKER_MAX_ENCODED_LENGTH:
        raise DocumentLoadError("网页资料来源元数据无效。")
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        decoded = base64.b64decode(
            padded,
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(decoded.decode("utf-8"))
        metadata = WebSourceMetadata(**payload)
    except (
        binascii.Error,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise DocumentLoadError("网页资料来源元数据无效。") from error
    _validate_web_source_metadata(metadata)
    body = remaining if separator else ""
    if not body.strip():
        raise DocumentLoadError("网页资料没有可建立索引的正文。")
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != metadata.content_sha256:
        raise DocumentLoadError("网页资料正文与来源元数据不一致。")
    return f"{filename} · 网页：{metadata.canonical_url}", body


def _extract_web_source_details(
    filename: str,
    text: str,
) -> tuple[str, str, WebSourceMetadata | None]:
    """保留旧返回契约之外，再把已验证网页元数据交给结构化摄取链。"""
    source, body = _extract_web_source(filename, text)
    if source == filename:
        return source, body, None
    first_line = text.partition("\n")[0]
    encoded = first_line[
        len(WEB_SOURCE_MARKER_PREFIX) : -len(WEB_SOURCE_MARKER_SUFFIX)
    ]
    padded = encoded + "=" * (-len(encoded) % 4)
    payload = json.loads(
        base64.b64decode(padded, altchars=b"-_", validate=True).decode("utf-8")
    )
    metadata = WebSourceMetadata(**payload)
    _validate_web_source_metadata(metadata)
    return source, body, metadata


def extract_pdf_pages(
    path: Path,
    *,
    max_pages: int | None = None,
    max_total_characters: int | None = None,
) -> list[tuple[str, str]]:
    """按页提取 PDF 文字层，返回带文件名和页码的资料单元。"""
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages 必须大于 0。")
    if max_total_characters is not None and max_total_characters <= 0:
        raise ValueError("max_total_characters 必须大于 0。")
    pages: list[tuple[str, str]] = []
    total_characters = 0
    try:
        with pdfplumber.open(path) as pdf:
            if max_pages is not None and len(pdf.pages) > max_pages:
                raise DocumentLoadError(
                    f"PDF“{path.name}”超过 {max_pages} 页安全限制。"
                )
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = (page.extract_text() or "").strip()
                except Exception as error:
                    raise DocumentLoadError(
                        f"提取 PDF“{path.name}”第 {page_number} 页文字失败。"
                    ) from error
                if page_text:
                    total_characters += len(page_text)
                    if (
                        max_total_characters is not None
                        and total_characters > max_total_characters
                    ):
                        raise DocumentLoadError(
                            f"PDF“{path.name}”提取文字超过安全限制。"
                        )
                    pages.append((f"{path.name} · 第 {page_number} 页", page_text))
    except PdfminerException as error:
        original_error = error.args[0] if error.args else None
        if isinstance(original_error, PDFPasswordIncorrect):
            raise DocumentLoadError(
                f"无法读取 PDF“{path.name}”：暂不支持加密文件。"
            ) from error
        raise DocumentLoadError(f"无法读取 PDF“{path.name}”：文件可能已损坏。") from error
    except PDFPasswordIncorrect as error:
        raise DocumentLoadError(f"无法读取 PDF“{path.name}”：暂不支持加密文件。") from error
    except (OSError, PDFSyntaxError) as error:
        raise DocumentLoadError(f"无法读取 PDF“{path.name}”：文件可能已损坏。") from error

    if not pages:
        raise DocumentLoadError(
            f"PDF“{path.name}”没有可提取文字，可能是扫描图片，需要先进行 OCR。"
        )
    return pages


def load_material_units(
    directory: Path,
    *,
    max_pdf_pages: int | None = None,
    max_total_characters: int | None = None,
) -> list[MaterialUnit]:
    """读取资料并保留 Material、页码与可靠 URL 等结构化来源信息。"""
    if max_total_characters is not None and max_total_characters <= 0:
        raise ValueError("max_total_characters 必须大于 0。")
    documents: list[MaterialUnit] = []
    total_characters = 0
    for path in sorted(directory.glob("*")):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            continue
        if suffix == ".pdf":
            remaining_characters = (
                None
                if max_total_characters is None
                else max_total_characters - total_characters
            )
            if remaining_characters is not None and remaining_characters <= 0:
                raise DocumentLoadError("资料提取文字超过安全限制。")
            loaded_pages = extract_pdf_pages(
                path,
                max_pages=max_pdf_pages,
                max_total_characters=remaining_characters,
            )
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            material = Material(
                material_id=material_identity(
                    source_type="pdf",
                    filename=path.name,
                ),
                source_type="pdf",
                filename=path.name,
                content_hash=content_hash,
            )
            loaded = [
                MaterialUnit(
                    material=material,
                    source=source,
                    text=text,
                    page=int(PDF_SOURCE_PATTERN.fullmatch(source).group("page")),
                )
                for source, text in loaded_pages
            ]
        else:
            text = path.read_text(encoding="utf-8")
            if suffix == ".md":
                source, body, web_metadata = _extract_web_source_details(path.name, text)
            else:
                source, body, web_metadata = path.name, text, None
            source_type = "web" if web_metadata is not None else {
                ".md": "markdown",
                ".txt": "text",
            }[suffix]
            canonical_url = (
                web_metadata.canonical_url if web_metadata is not None else None
            )
            content_hash = (
                web_metadata.content_sha256
                if web_metadata is not None
                else hashlib.sha256(path.read_bytes()).hexdigest()
            )
            material = Material(
                material_id=material_identity(
                    source_type=source_type,
                    filename=path.name,
                    canonical_url=canonical_url,
                ),
                source_type=source_type,
                filename=path.name,
                content_hash=content_hash,
                canonical_url=canonical_url,
            )
            loaded = [MaterialUnit(material=material, source=source, text=body)]
        total_characters += sum(len(unit.text) for unit in loaded)
        if (
            max_total_characters is not None
            and total_characters > max_total_characters
        ):
            raise DocumentLoadError("资料提取文字超过安全限制。")
        documents.extend(loaded)
    return documents


def load_documents(
    directory: Path,
    *,
    max_pdf_pages: int | None = None,
    max_total_characters: int | None = None,
) -> list[tuple[str, str]]:
    """兼容旧教学与测试入口；正式摄取链使用 load_material_units。"""
    return [
        (unit.source, unit.text)
        for unit in load_material_units(
            directory,
            max_pdf_pages=max_pdf_pages,
            max_total_characters=max_total_characters,
        )
    ]


def split_text(text: str, chunk_size: int = 180) -> list[str]:
    """按字符数将文本切段，避免后续一次把整份资料交给模型。"""
    normalized = " ".join(text.split())
    return [normalized[start : start + chunk_size] for start in range(0, len(normalized), chunk_size)]


def _legacy_material_unit(source: str, text: str) -> MaterialUnit:
    pdf_match = PDF_SOURCE_PATTERN.fullmatch(source)
    web_match = WEB_SOURCE_PATTERN.fullmatch(source)
    filename = source
    source_type = "unknown"
    page = None
    canonical_url = None
    if pdf_match:
        filename = pdf_match.group("filename")
        source_type = "pdf"
        page = int(pdf_match.group("page"))
    elif web_match:
        filename = web_match.group("filename")
        source_type = "web"
        canonical_url = web_match.group("url")
    else:
        source_type = {
            ".md": "markdown",
            ".txt": "text",
            ".pdf": "pdf",
        }.get(Path(filename).suffix.casefold(), "unknown")
    content_hash = stable_sha256(text)
    material = Material(
        material_id=material_identity(
            source_type=source_type,
            filename=filename,
            canonical_url=canonical_url,
        ),
        source_type=source_type,
        filename=filename,
        content_hash=content_hash,
        canonical_url=canonical_url,
    )
    return MaterialUnit(
        material=material,
        source=source,
        text=text,
        page=page,
    )


def build_chunks(
    documents: list[tuple[str, str]] | list[MaterialUnit],
) -> list[DocumentChunk]:
    """把每个资料单元变成多个带来源信息的文本块。"""
    chunks: list[DocumentChunk] = []
    for document in documents:
        unit = (
            document
            if isinstance(document, MaterialUnit)
            else _legacy_material_unit(document[0], document[1])
        )
        for index, content in enumerate(split_text(unit.text), start=1):
            content_hash = stable_sha256(content)
            chunks.append(
                DocumentChunk(
                    source=unit.source,
                    index=index,
                    content=content,
                    material_id=unit.material.material_id,
                    source_type=unit.material.source_type,
                    filename=unit.material.filename,
                    page=unit.page,
                    section=unit.section,
                    material_content_hash=unit.material.content_hash,
                    content_hash=content_hash,
                    canonical_url=unit.material.canonical_url,
                )
            )
    return chunks


def main() -> None:
    documents = load_material_units(DOCUMENTS_DIR)
    chunks = build_chunks(documents)

    print(f"读取到 {len(documents)} 个资料单元，切分为 {len(chunks)} 个文本块。\n")
    for chunk in chunks:
        print(f"[{chunk.source} · 第 {chunk.index} 段]")
        print(f"{chunk.content}\n")


if __name__ == "__main__":
    main()
