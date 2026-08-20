import unittest

from app.chunk_documents import (
    CHUNKER_VERSION,
    DocumentChunk,
    Material,
    MaterialUnit,
    build_chunks,
)
from app.structure_aware_chunking import (
    STRUCTURE_AWARE_CHUNKER_VERSION,
    StructureAwareChunkingConfig,
    build_structure_aware_chunks,
)


def make_unit(
    text: str,
    *,
    source_type: str = "markdown",
    section: str | None = None,
    page: int | None = None,
    paragraph_index: int | None = None,
) -> MaterialUnit:
    filename = "notes.md" if source_type == "markdown" else "notes.docx"
    return MaterialUnit(
        material=Material(
            material_id="a" * 64,
            source_type=source_type,
            filename=filename,
            content_hash="b" * 64,
        ),
        source=filename,
        text=text,
        section=section,
        page=page,
        paragraph_index=paragraph_index,
    )


class StructureAwareChunkingTests(unittest.TestCase):
    def test_markdown_structure_is_preserved_without_heading_only_chunks(self) -> None:
        text = (
            "# Retrieval\n\n"
            "Dense retrieval finds semantic candidates.\n\n"
            "- preserve headings\n- preserve lists\n\n"
            "```python\nresult = retrieve(query)\n```"
        )

        chunks = build_structure_aware_chunks([make_unit(text)])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "Retrieval")
        self.assertIn("# Retrieval", chunks[0].content)
        self.assertIn("- preserve headings\n- preserve lists", chunks[0].content)
        self.assertIn("```python\nresult = retrieve(query)\n```", chunks[0].content)
        self.assertNotEqual(chunks[0].content.strip(), "# Retrieval")

    def test_heading_change_starts_a_new_chunk_and_avoids_repeated_prefix(self) -> None:
        unit = make_unit(
            "# First\n\nParagraph one.\n\nParagraph two.\n\n"
            "# Second\n\nParagraph three."
        )

        chunks = build_structure_aware_chunks([unit])

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].content.count("# First"), 1)
        self.assertEqual(chunks[1].section, "Second")

    def test_oversized_paragraph_prefers_sentence_boundaries(self) -> None:
        unit = make_unit("First sentence. Second sentence. Third sentence.")
        config = StructureAwareChunkingConfig(max_chars=32)

        chunks = build_structure_aware_chunks([unit], config)

        self.assertTrue(all(len(chunk.content) <= 32 for chunk in chunks))
        self.assertEqual(
            " ".join(chunk.content for chunk in chunks),
            "First sentence. Second sentence. Third sentence.",
        )

    def test_docx_locator_metadata_and_section_prefix_are_preserved(self) -> None:
        unit = make_unit(
            "Retriever finds evidence.",
            source_type="docx",
            section="RAG > Retrieval",
            paragraph_index=3,
        )

        chunk = build_structure_aware_chunks([unit])[0]

        self.assertEqual(chunk.material_id, unit.material.material_id)
        self.assertEqual(chunk.paragraph_index, 3)
        self.assertEqual(chunk.section, "RAG > Retrieval")
        self.assertTrue(chunk.content.startswith("RAG > Retrieval\n\n"))
        self.assertEqual(chunk.chunker_version, STRUCTURE_AWARE_CHUNKER_VERSION)

    def test_candidate_ids_are_deterministic_and_content_sensitive(self) -> None:
        first = build_structure_aware_chunks([make_unit("# RAG\n\nEvidence first.")])[0]
        repeated = build_structure_aware_chunks([make_unit("# RAG\n\nEvidence first.")])[0]
        changed = build_structure_aware_chunks([make_unit("# RAG\n\nEvidence changed.")])[0]

        self.assertEqual(first.chunk_id, repeated.chunk_id)
        self.assertEqual(first.content_hash, repeated.content_hash)
        self.assertNotEqual(first.chunk_id, changed.chunk_id)
        self.assertNotEqual(first.content_hash, changed.content_hash)

    def test_production_chunker_remains_fixed_character_180(self) -> None:
        chunks = build_chunks([make_unit("x" * 181)])

        self.assertEqual([len(chunk.content) for chunk in chunks], [180, 1])
        self.assertTrue(all(chunk.chunker_version == CHUNKER_VERSION for chunk in chunks))
        self.assertNotEqual(CHUNKER_VERSION, STRUCTURE_AWARE_CHUNKER_VERSION)

    def test_nonzero_overlap_is_rejected_for_this_experiment(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 overlap"):
            StructureAwareChunkingConfig(overlap_chars=1)

    def test_all_generated_chunks_use_normal_document_chunk_contract(self) -> None:
        chunks = build_structure_aware_chunks([make_unit("# RAG\n\nEvidence first.")])

        self.assertTrue(all(isinstance(chunk, DocumentChunk) for chunk in chunks))
        self.assertTrue(all(len(chunk.chunk_id) == 64 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
