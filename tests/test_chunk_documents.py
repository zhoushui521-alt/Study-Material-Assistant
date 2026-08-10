import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.chunk_documents import (
    DocumentLoadError,
    WebSourceMetadata,
    build_chunks,
    encode_web_source_marker,
    extract_pdf_pages,
    load_documents,
    split_text,
)


def write_text_pdf(path: Path, text: str) -> None:
    """创建一个带 Helvetica 文字层的最小测试 PDF。"""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    content = DecodedStreamObject()
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content.set_data(f"BT /F1 12 Tf 72 200 Td ({escaped_text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(content)
    with path.open("wb") as stream:
        writer.write(stream)


class ChunkDocumentsTests(unittest.TestCase):
    def test_load_documents_uses_web_url_as_source_and_removes_marker(self) -> None:
        body = "# RAG\n\n正文"
        metadata = WebSourceMetadata(
            canonical_url="https://example.com/rag",
            title="RAG Guide",
            crawled_at="2026-08-10T09:30:00Z",
            content_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            documents_dir = Path(directory)
            (documents_dir / "web-example.md").write_text(
                f"{encode_web_source_marker(metadata)}\n{body}",
                encoding="utf-8",
            )

            documents = load_documents(documents_dir)

        self.assertEqual(
            documents,
            [("web-example.md · 网页：https://example.com/rag", "# RAG\n\n正文")],
        )

    def test_rejects_tampered_web_material_body(self) -> None:
        original_body = "# RAG\n\n原始正文"
        metadata = WebSourceMetadata(
            canonical_url="https://example.com/rag",
            title="RAG Guide",
            crawled_at="2026-08-10T09:30:00Z",
            content_sha256=hashlib.sha256(original_body.encode("utf-8")).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            documents_dir = Path(directory)
            (documents_dir / "web-example.md").write_text(
                f"{encode_web_source_marker(metadata)}\n# RAG\n\n篡改正文",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DocumentLoadError, "正文与来源元数据不一致"):
                load_documents(documents_dir)

    def test_rejects_web_marker_with_invalid_port(self) -> None:
        metadata = WebSourceMetadata(
            canonical_url="https://example.com:bad/rag",
            title="RAG Guide",
            crawled_at="2026-08-10T09:30:00Z",
            content_sha256="a" * 64,
        )

        with self.assertRaisesRegex(DocumentLoadError, "元数据无效"):
            encode_web_source_marker(metadata)

    def test_rejects_web_marker_with_non_default_port(self) -> None:
        metadata = WebSourceMetadata(
            canonical_url="https://example.com:80/rag",
            title="RAG Guide",
            crawled_at="2026-08-10T09:30:00Z",
            content_sha256="a" * 64,
        )

        with self.assertRaisesRegex(DocumentLoadError, "元数据无效"):
            encode_web_source_marker(metadata)

    def test_split_text_splits_long_text(self) -> None:
        chunks = split_text("a" * 181, chunk_size=180)
        self.assertEqual(chunks, ["a" * 180, "a"])

    def test_build_chunks_keeps_source_information(self) -> None:
        chunks = build_chunks([("notes.md", "RAG 会先检索资料。")])
        self.assertEqual(chunks[0].source, "notes.md")
        self.assertEqual(chunks[0].index, 1)

    def test_load_documents_extracts_pdf_text_and_page_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents_dir = Path(directory)
            write_text_pdf(documents_dir / "rag.pdf", "RAG PDF content")

            documents = dict(load_documents(documents_dir))

            source = "rag.pdf · 第 1 页"
            self.assertIn(source, documents)
            self.assertIn("RAG PDF content", documents[source])

    @patch("app.chunk_documents.pdfplumber.open")
    def test_extract_pdf_pages_preserves_chinese_text(self, mock_open: Mock) -> None:
        page = Mock()
        page.extract_text.return_value = "第 8 章 多元分布"
        mock_open.return_value.__enter__.return_value.pages = [page]

        documents = extract_pdf_pages(Path("documents.pdf"))

        self.assertEqual(
            documents,
            [("documents.pdf · 第 1 页", "第 8 章 多元分布")],
        )

    @patch("app.chunk_documents.pdfplumber.open")
    def test_extract_pdf_pages_rejects_page_count_over_limit(
        self,
        mock_open: Mock,
    ) -> None:
        pages = [Mock(), Mock()]
        mock_open.return_value.__enter__.return_value.pages = pages

        with self.assertRaisesRegex(DocumentLoadError, "超过 1 页"):
            extract_pdf_pages(Path("documents.pdf"), max_pages=1)

        for page in pages:
            page.extract_text.assert_not_called()

    @patch("app.chunk_documents.pdfplumber.open")
    def test_extract_pdf_pages_rejects_extracted_text_over_limit(
        self,
        mock_open: Mock,
    ) -> None:
        page = Mock()
        page.extract_text.return_value = "123456"
        mock_open.return_value.__enter__.return_value.pages = [page]

        with self.assertRaisesRegex(DocumentLoadError, "提取文字超过安全限制"):
            extract_pdf_pages(
                Path("documents.pdf"),
                max_total_characters=5,
            )

    def test_load_documents_rejects_total_text_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents_dir = Path(directory)
            (documents_dir / "notes.md").write_text("123456", encoding="utf-8")

            with self.assertRaisesRegex(DocumentLoadError, "提取文字超过安全限制"):
                load_documents(documents_dir, max_total_characters=5)

    def test_load_documents_rejects_pdf_without_text_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "scan.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=300, height=300)
            with pdf_path.open("wb") as stream:
                writer.write(stream)

            with self.assertRaisesRegex(DocumentLoadError, "需要先进行 OCR"):
                load_documents(Path(directory))

    def test_load_documents_reports_encrypted_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "secret.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=300, height=300)
            writer.encrypt("secret")
            with pdf_path.open("wb") as stream:
                writer.write(stream)

            with self.assertRaisesRegex(DocumentLoadError, "暂不支持加密文件"):
                load_documents(Path(directory))

    def test_load_documents_reports_corrupted_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "broken.pdf"
            pdf_path.write_bytes(b"not a valid PDF")

            with self.assertRaisesRegex(DocumentLoadError, "文件可能已损坏"):
                load_documents(Path(directory))


if __name__ == "__main__":
    unittest.main()
