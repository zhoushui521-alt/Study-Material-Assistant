"""安全读取 DOCX 的段落、标题与表格，不处理分页或视觉版式。"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

from docx import Document as open_docx
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError


DOCX_MAIN_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
)
DOCX_REQUIRED_PARTS = frozenset(
    {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
)
MAX_DOCX_PACKAGE_ENTRIES = 5000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
HEADING_STYLE_PATTERN = re.compile(r"^(?:Heading|标题)\s*([1-3])$", re.IGNORECASE)


class DocxParseError(RuntimeError):
    """DOCX 不是安全、可解析且包含可检索文本的 OOXML 文档。"""


@dataclass(frozen=True)
class DocxBlock:
    """Word 中一个可定位的段落或表格。"""

    text: str
    section: str | None
    paragraph_index: int | None = None
    table_index: int | None = None


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def validate_docx_package(path: Path) -> None:
    """验证 ZIP/OOXML 结构、大小和主文档类型，不只相信扩展名。"""
    try:
        if not zipfile.is_zipfile(path):
            raise DocxParseError(f"DOCX“{path.name}”不是有效的 OOXML ZIP 文件。")
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            if len(entries) > MAX_DOCX_PACKAGE_ENTRIES:
                raise DocxParseError(f"DOCX“{path.name}”内部文件数量超过安全限制。")
            if sum(entry.file_size for entry in entries) > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise DocxParseError(f"DOCX“{path.name}”解压后大小超过安全限制。")
            if not DOCX_REQUIRED_PARTS.issubset(names):
                raise DocxParseError(f"DOCX“{path.name}”缺少必要的 OOXML 结构。")
            for entry in entries:
                member = PurePosixPath(entry.filename)
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or "\\" in entry.filename
                ):
                    raise DocxParseError(f"DOCX“{path.name}”包含不安全的内部路径。")
                if entry.flag_bits & 0x1:
                    raise DocxParseError(f"DOCX“{path.name}”包含不支持的加密内容。")
            corrupted_member = archive.testzip()
            if corrupted_member is not None:
                raise DocxParseError(f"DOCX“{path.name}”内部文件已损坏。")
            content_types = ElementTree.fromstring(archive.read("[Content_Types].xml"))
            main_types = {
                element.attrib.get("ContentType")
                for element in content_types
                if element.tag.endswith("Override")
                and element.attrib.get("PartName") == "/word/document.xml"
            }
            if DOCX_MAIN_CONTENT_TYPE not in main_types:
                raise DocxParseError(
                    f"DOCX“{path.name}”主文档类型无效；不支持宏文档或伪装文件。"
                )
    except DocxParseError:
        raise
    except (OSError, KeyError, ElementTree.ParseError, zipfile.BadZipFile) as error:
        raise DocxParseError(f"DOCX“{path.name}”文件已损坏或无法读取。") from error


def _heading_level(paragraph: Paragraph) -> int | None:
    style_name = getattr(getattr(paragraph, "style", None), "name", "")
    match = HEADING_STYLE_PATTERN.fullmatch(style_name or "")
    return int(match.group(1)) if match else None


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = [_normalized_text(cell.text) for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_docx_blocks(
    path: Path,
    *,
    max_total_characters: int | None = None,
) -> list[DocxBlock]:
    """按文档顺序提取段落和表格；标题只作为可靠 section，不推测页码。"""
    if max_total_characters is not None and max_total_characters <= 0:
        raise ValueError("max_total_characters 必须大于 0。")
    validate_docx_package(path)
    try:
        document = open_docx(path)
        blocks: list[DocxBlock] = []
        headings: dict[int, str] = {}
        paragraph_index = 0
        table_index = 0
        total_characters = 0
        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                paragraph_index += 1
                text = _normalized_text(item.text)
                if not text:
                    continue
                heading_level = _heading_level(item)
                if heading_level is not None:
                    headings = {
                        level: value
                        for level, value in headings.items()
                        if level < heading_level
                    }
                    headings[heading_level] = text
                section = " > ".join(headings[level] for level in sorted(headings)) or None
                block = DocxBlock(
                    text=text,
                    section=section,
                    paragraph_index=paragraph_index,
                )
            elif isinstance(item, Table):
                table_index += 1
                text = _table_text(item)
                if not text:
                    continue
                section = " > ".join(headings[level] for level in sorted(headings)) or None
                block = DocxBlock(
                    text=text,
                    section=section,
                    table_index=table_index,
                )
            else:
                continue
            total_characters += len(block.text)
            if (
                max_total_characters is not None
                and total_characters > max_total_characters
            ):
                raise DocxParseError(f"DOCX“{path.name}”提取文字超过安全限制。")
            blocks.append(block)
    except DocxParseError:
        raise
    except (
        OSError,
        KeyError,
        ValueError,
        PackageNotFoundError,
        XMLSyntaxError,
        zipfile.BadZipFile,
    ) as error:
        raise DocxParseError(f"DOCX“{path.name}”文件已损坏或无法解析。") from error

    if not blocks:
        raise DocxParseError(f"DOCX“{path.name}”没有可建立索引的段落或表格文字。")
    return blocks
