import unittest

from langchain_core.documents import Document

from app.context_selector import (
    BaselineContextSelector,
    ContextSelectionError,
    EvidenceScoreContextSelector,
)
from app.hybrid_search import CONTEXT_ROLE_METADATA_KEY


def make_document(
    chunk_id: str,
    content: str,
    *,
    source: str = "notes.md",
    chunk_index: int = 1,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "source": source,
            "chunk_index": chunk_index,
        },
    )


class BaselineContextSelectorTests(unittest.TestCase):
    def test_preserves_current_context_without_reordering(self) -> None:
        evidences = [
            make_document("seed", "seed"),
            make_document("adjacent", "adjacent", chunk_index=2),
        ]

        selected = BaselineContextSelector().select("query", evidences)

        self.assertEqual(selected, evidences)
        self.assertIsNot(selected, evidences)


class EvidenceScoreContextSelectorTests(unittest.TestCase):
    def test_empty_evidence_returns_empty_context(self) -> None:
        selector = EvidenceScoreContextSelector()

        self.assertEqual(selector.select("query", []), [])

    def test_single_evidence_is_preserved(self) -> None:
        evidence = make_document("only", "only evidence")

        selected = EvidenceScoreContextSelector().select("query", [evidence])

        self.assertEqual(selected, [evidence])

    def test_selects_one_best_existing_adjacent_evidence_per_seed(self) -> None:
        seed_a = make_document("seed-a", "seed", source="a.md", chunk_index=3)
        seed_b = make_document("seed-b", "seed", source="b.md", chunk_index=3)
        a_near = make_document(
            "a-near",
            "unrelated",
            source="a.md",
            chunk_index=2,
        )
        a_relevant = make_document(
            "a-relevant",
            "target target",
            source="a.md",
            chunk_index=1,
        )
        b_relevant = make_document(
            "b-relevant",
            "target",
            source="b.md",
            chunk_index=4,
        )
        unrelated_source = make_document(
            "other",
            "target target target",
            source="other.md",
            chunk_index=2,
        )
        evidences = [
            seed_a,
            seed_b,
            a_near,
            a_relevant,
            b_relevant,
            unrelated_source,
        ]

        selected = EvidenceScoreContextSelector(
            seed_count=2,
            adjacent_per_seed=1,
        ).select("target", evidences)

        self.assertEqual(selected, [seed_a, seed_b, a_relevant, b_relevant])

    def test_duplicate_evidence_is_returned_once(self) -> None:
        seed = make_document("seed", "seed", chunk_index=2)
        duplicate = make_document("seed", "different object", chunk_index=2)
        adjacent = make_document("adjacent", "target", chunk_index=3)

        selected = EvidenceScoreContextSelector(
            seed_count=1,
            adjacent_per_seed=1,
        ).select("target", [seed, duplicate, adjacent])

        self.assertEqual(selected, [seed, adjacent])

    def test_explicit_provenance_handles_fewer_seeds_than_configured(self) -> None:
        seed = make_document("seed", "seed", chunk_index=2)
        seed.metadata[CONTEXT_ROLE_METADATA_KEY] = "seed"
        adjacent = make_document("adjacent", "target", chunk_index=3)
        adjacent.metadata[CONTEXT_ROLE_METADATA_KEY] = "adjacent"

        selected = EvidenceScoreContextSelector(
            seed_count=3,
            adjacent_per_seed=1,
        ).select("target", [seed, adjacent])

        self.assertEqual(selected, [seed, adjacent])

    def test_exact_token_budget_is_never_exceeded(self) -> None:
        first = make_document("first", "one two", chunk_index=1)
        second = make_document("second", "three four", chunk_index=2)
        selector = EvidenceScoreContextSelector(
            seed_count=2,
            adjacent_per_seed=0,
            token_budget=3,
            token_counter=lambda text: len(text.split()),
        )

        selected = selector.select("query", [first, second])

        self.assertEqual(selected, [first])
        self.assertLessEqual(sum(len(item.page_content.split()) for item in selected), 3)

    def test_token_budget_requires_reliable_counter(self) -> None:
        with self.assertRaisesRegex(ContextSelectionError, "Token Counter"):
            EvidenceScoreContextSelector(token_budget=10)

    def test_rejects_invalid_token_counter_result(self) -> None:
        selector = EvidenceScoreContextSelector(
            token_budget=10,
            token_counter=lambda text: -1,
        )

        with self.assertRaisesRegex(ContextSelectionError, "非负整数"):
            selector.select("query", [make_document("seed", "text")])


if __name__ == "__main__":
    unittest.main()
