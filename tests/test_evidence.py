import unittest

from langchain_core.documents import Document

from app.evidence import (
    build_evidence_context,
    evidence_from_document,
    resolve_citations,
)


class EvidenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = Document(
            page_content="TCP 使用三次握手建立连接。",
            metadata={
                "source": "network.pdf · 第 32 页",
                "filename": "network.pdf",
                "source_type": "pdf",
                "page": 32,
                "chunk_index": 4,
                "material_id": "a" * 64,
                "chunk_id": "b" * 64,
                "content_hash": "c" * 64,
            },
        )

    def test_builds_evidence_from_structured_chunk_metadata(self) -> None:
        evidence = evidence_from_document(self.document, "S1")

        self.assertEqual(evidence.context_id, "S1")
        self.assertEqual(evidence.material_id, "a" * 64)
        self.assertEqual(evidence.chunk_id, "b" * 64)
        self.assertEqual(evidence.filename, "network.pdf")
        self.assertEqual(evidence.page, 32)
        self.assertEqual(evidence.chunk_index, 4)
        self.assertEqual(evidence.locator, "network.pdf#page=32&chunk=4")
        self.assertEqual(evidence.excerpt, self.document.page_content)

    def test_resolves_only_current_request_ids_and_server_fills_metadata(self) -> None:
        evidences = build_evidence_context([self.document])

        citations, invalid_ids = resolve_citations(
            "连接前需要三次握手。[S1] 模型声称来自 fake.pdf 第 9 页。",
            evidences,
        )

        self.assertEqual(invalid_ids, ())
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].citation_id, "S1")
        self.assertEqual(citations[0].filename, "network.pdf")
        self.assertEqual(citations[0].page, 32)
        self.assertNotEqual(citations[0].filename, "fake.pdf")

    def test_reports_ids_that_are_not_in_current_evidence_map(self) -> None:
        evidences = build_evidence_context([self.document])

        citations, invalid_ids = resolve_citations("答案。[S1][S9]", evidences)

        self.assertEqual([citation.citation_id for citation in citations], ["S1"])
        self.assertEqual(invalid_ids, ("S9",))


if __name__ == "__main__":
    unittest.main()
