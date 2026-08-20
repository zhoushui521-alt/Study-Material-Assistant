"""Stage 3.3 的隔离 Structure-aware Chunking 实验实现。

正式摄取链仍使用 ``app.chunk_documents.build_chunks``。本模块只为 A/B Evaluation
生成实验 Chunk，不修改生产索引或默认 Chunker。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

if __package__:
    from app.chunk_documents import DocumentChunk, MaterialUnit
    from app.evidence import stable_sha256
else:
    from chunk_documents import DocumentChunk, MaterialUnit
    from evidence import stable_sha256


STRUCTURE_AWARE_CHUNKER_VERSION = "structure-aware-block-600-overlap-0-v1"
DEFAULT_MAX_CHARS = 600
DEFAULT_OVERLAP_CHARS = 0

_MARKDOWN_HEADING = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>\S.*)$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+] |\d+[.)]\s+)")
_FENCE_START = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?；;])\s+")


@dataclass(frozen=True)
class StructureAwareChunkingConfig:
    """Stage 3.3 唯一候选策略；第一轮不引入文本 overlap。"""

    max_chars: int = DEFAULT_MAX_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS
    preserve_markdown_headings: bool = True
    preserve_lists: bool = True
    preserve_code_blocks: bool = True
    chunker_version: str = STRUCTURE_AWARE_CHUNKER_VERSION

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("Structure-aware max_chars 必须大于 0。")
        if self.overlap_chars != 0:
            raise ValueError("Stage 3.3 第一轮固定为 0 overlap，避免重复证据干扰。")
        if not self.chunker_version.strip():
            raise ValueError("Structure-aware chunker_version 不能为空。")


@dataclass(frozen=True)
class _Block:
    text: str
    section: str | None
    kind: Literal["paragraph", "list", "code"]


@dataclass(frozen=True)
class _ChunkText:
    text: str
    section: str | None


def _normalize_line(value: str) -> str:
    return " ".join(value.strip().split())


def _section_value(headings: dict[int, str]) -> str | None:
    return " > ".join(headings[level] for level in sorted(headings)) or None


def _heading_prefix(headings: dict[int, str]) -> str:
    return "\n".join(f"{'#' * level} {headings[level]}" for level in sorted(headings))


def _markdown_blocks(text: str, config: StructureAwareChunkingConfig) -> list[_Block]:
    blocks: list[_Block] = []
    headings: dict[int, str] = {}
    pending_lines: list[str] = []
    pending_kind: Literal["paragraph", "list"] = "paragraph"
    code_lines: list[str] | None = None
    code_fence: str | None = None

    def flush_pending() -> None:
        nonlocal pending_lines, pending_kind
        if not pending_lines:
            return
        body = "\n".join(pending_lines).strip()
        if body:
            prefix = _heading_prefix(headings)
            content = f"{prefix}\n\n{body}" if prefix else body
            blocks.append(_Block(content, _section_value(headings), pending_kind))
        pending_lines = []
        pending_kind = "paragraph"

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if code_lines is not None:
            code_lines.append(line)
            if line.strip().startswith(code_fence or "```"):
                content = "\n".join(code_lines).strip()
                prefix = _heading_prefix(headings)
                blocks.append(
                    _Block(
                        f"{prefix}\n\n{content}" if prefix else content,
                        _section_value(headings),
                        "code",
                    )
                )
                code_lines = None
                code_fence = None
            continue

        fence_match = _FENCE_START.match(line)
        if config.preserve_code_blocks and fence_match:
            flush_pending()
            code_lines = [line]
            code_fence = fence_match.group("fence")
            continue

        heading_match = _MARKDOWN_HEADING.match(line)
        if config.preserve_markdown_headings and heading_match:
            flush_pending()
            level = len(heading_match.group("marks"))
            title = _normalize_line(heading_match.group("title"))
            headings = {
                current_level: value
                for current_level, value in headings.items()
                if current_level < level
            }
            headings[level] = title
            continue

        if not line.strip():
            flush_pending()
            continue

        is_list = bool(_LIST_ITEM.match(line)) if config.preserve_lists else False
        next_kind: Literal["paragraph", "list"] = "list" if is_list else "paragraph"
        if pending_lines and next_kind != pending_kind:
            flush_pending()
        pending_kind = next_kind
        pending_lines.append(line.strip() if is_list else _normalize_line(line))

    flush_pending()
    if code_lines is not None:
        content = "\n".join(code_lines).strip()
        prefix = _heading_prefix(headings)
        blocks.append(
            _Block(
                f"{prefix}\n\n{content}" if prefix else content,
                _section_value(headings),
                "code",
            )
        )
    return blocks


def _plain_blocks(text: str, section: str | None) -> list[_Block]:
    blocks: list[_Block] = []
    paragraph: list[str] = []
    list_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        body = "\n".join(paragraph).strip()
        if body:
            blocks.append(_Block(body, section, "paragraph"))
        paragraph = []

    def flush_list() -> None:
        nonlocal list_lines
        body = "\n".join(list_lines).strip()
        if body:
            blocks.append(_Block(body, section, "list"))
        list_lines = []

    for raw_line in text.splitlines():
        normalized = _normalize_line(raw_line)
        if not normalized:
            flush_paragraph()
            flush_list()
            continue
        if _LIST_ITEM.match(raw_line):
            flush_paragraph()
            list_lines.append(raw_line.strip())
        else:
            flush_list()
            paragraph.append(normalized)
    flush_paragraph()
    flush_list()
    return blocks


def _pack_fragments(fragments: list[str], max_chars: int, separator: str) -> list[str]:
    packed: list[str] = []
    current = ""
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        candidate = f"{current}{separator}{fragment}" if current else fragment
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            packed.append(current)
            current = ""
        if len(fragment) <= max_chars:
            current = fragment
        else:
            packed.extend(
                fragment[start : start + max_chars]
                for start in range(0, len(fragment), max_chars)
            )
    if current:
        packed.append(current)
    return packed


def _split_oversized(
    text: str,
    max_chars: int,
    *,
    prefer_lines: bool = False,
) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = [line for line in text.splitlines() if line.strip()]
    if prefer_lines and len(lines) > 1 and all(len(line) <= max_chars for line in lines):
        return _pack_fragments(lines, max_chars, "\n")
    normalized_text = " ".join(lines) if len(lines) > 1 else text
    sentences = [
        item for item in _SENTENCE_BOUNDARY.split(normalized_text) if item.strip()
    ]
    if len(sentences) > 1 and all(len(item) <= max_chars for item in sentences):
        return _pack_fragments(sentences, max_chars, " ")
    words = text.split()
    if len(words) > 1 and all(len(word) <= max_chars for word in words):
        return _pack_fragments(words, max_chars, " ")
    return [text[start : start + max_chars] for start in range(0, len(text), max_chars)]


def _split_block(block: _Block, max_chars: int) -> list[str]:
    if len(block.text) <= max_chars:
        return [block.text]
    prefix, separator, body = block.text.partition("\n\n")
    if block.section and separator and len(prefix) + 2 < max_chars:
        body_limit = max_chars - len(prefix) - 2
        return [
            f"{prefix}\n\n{piece}"
            for piece in _split_oversized(
                body,
                body_limit,
                prefer_lines=block.kind in {"list", "code"},
            )
        ]
    return _split_oversized(
        block.text,
        max_chars,
        prefer_lines=block.kind in {"list", "code"},
    )


def _without_repeated_prefix(text: str, *, section: str | None) -> str:
    if not section:
        return text
    _, separator, body = text.partition("\n\n")
    return body if separator else text


def _blocks_for_unit(
    unit: MaterialUnit,
    config: StructureAwareChunkingConfig,
) -> list[_Block]:
    if unit.material.source_type in {"markdown", "web"}:
        return _markdown_blocks(unit.text, config)
    blocks = _plain_blocks(unit.text, unit.section)
    if unit.section and unit.material.source_type == "docx":
        prefix = unit.section.strip()
        return [
            _Block(
                block.text
                if block.text == prefix or block.text.startswith(f"{prefix}\n")
                else f"{prefix}\n\n{block.text}",
                block.section,
                block.kind,
            )
            for block in blocks
        ]
    return blocks


def _chunk_texts_for_unit(
    unit: MaterialUnit,
    config: StructureAwareChunkingConfig,
) -> list[_ChunkText]:
    output: list[_ChunkText] = []
    current = ""
    current_section: str | None = None
    for block in _blocks_for_unit(unit, config):
        pieces = _split_block(block, config.max_chars)
        for piece in pieces:
            merge_piece = (
                _without_repeated_prefix(piece, section=block.section)
                if current and current_section == block.section
                else piece
            )
            candidate = f"{current}\n\n{merge_piece}" if current else piece
            if (
                current
                and current_section == block.section
                and len(candidate) <= config.max_chars
            ):
                current = candidate
                continue
            if current:
                output.append(_ChunkText(current, current_section))
            current = piece
            current_section = block.section
    if current:
        output.append(_ChunkText(current, current_section))
    return output


def build_structure_aware_chunks(
    units: list[MaterialUnit],
    config: StructureAwareChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """用结构优先策略构建实验 Chunk，保留 Stage 1 身份与来源字段。"""
    active_config = config or StructureAwareChunkingConfig()
    chunks: list[DocumentChunk] = []
    for unit in units:
        for index, candidate in enumerate(
            _chunk_texts_for_unit(unit, active_config),
            start=1,
        ):
            content_hash = stable_sha256(candidate.text)
            chunks.append(
                DocumentChunk(
                    source=unit.source,
                    index=index,
                    content=candidate.text,
                    material_id=unit.material.material_id,
                    source_type=unit.material.source_type,
                    filename=unit.material.filename,
                    page=unit.page,
                    section=candidate.section or unit.section,
                    paragraph_index=unit.paragraph_index,
                    table_index=unit.table_index,
                    material_content_hash=unit.material.content_hash,
                    content_hash=content_hash,
                    canonical_url=unit.material.canonical_url,
                    chunker_version=active_config.chunker_version,
                )
            )
    return chunks
