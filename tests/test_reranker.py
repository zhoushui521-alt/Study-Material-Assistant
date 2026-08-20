import unittest

from langchain_core.documents import Document

from app.reranker import (
    DeterministicMockReranker,
    FlagEmbeddingCrossEncoderReranker,
    Reranker,
    RerankerError,
)


MATERIAL_ID = "a" * 64


def make_document(number: int) -> Document:
    return Document(
        page_content=f"chunk {number}",
        metadata={
            "source": "notes.md",
            "filename": "notes.md",
            "source_type": "markdown",
            "chunk_index": number,
            "chunk_id": f"{number:064x}",
            "material_id": MATERIAL_ID,
            "content_hash": f"{number + 100:064x}",
        },
    )


class BrokenReranker(Reranker):
    @property
    def model_name(self) -> str:
        return "broken"

    @property
    def model_source(self) -> str:
        return "fixture"

    def score(self, query: str, documents: tuple[Document, ...]) -> tuple[float, ...]:
        raise OSError("model unavailable")


class StaticReranker(Reranker):
    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = scores

    @property
    def model_name(self) -> str:
        return "static"

    @property
    def model_source(self) -> str:
        return "fixture"

    def score(self, query: str, documents: tuple[Document, ...]) -> tuple[float, ...]:
        return self.scores


class FakeCrossEncoderBackend:
    def __init__(self, scores: object) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def compute_score(
        self,
        sentence_pairs: list[tuple[str, str]],
        **kwargs: object,
    ) -> object:
        self.calls.append(sentence_pairs)
        return self.scores


class RerankerTests(unittest.TestCase):
    def test_rerank_saves_original_new_rank_score_and_change(self) -> None:
        first, second = make_document(1), make_document(2)
        reranker = DeterministicMockReranker(
            {str(first.metadata["chunk_id"]): 0.1, str(second.metadata["chunk_id"]): 0.9}
        )

        results = reranker.rerank("query", [first, second])

        self.assertEqual([result.chunk_id for result in results], ["2".zfill(64), "1".zfill(64)])
        self.assertEqual(results[0].original_rank, 2)
        self.assertEqual(results[0].reranker_rank, 1)
        self.assertEqual(results[0].reranker_score, 0.9)
        self.assertEqual(results[0].rank_change, 1)

    def test_empty_candidates_are_valid_and_deterministic(self) -> None:
        reranker = DeterministicMockReranker({})
        self.assertEqual(reranker.rerank("query", []), [])
        self.assertEqual(reranker.rerank("query", []), [])

    def test_ties_preserve_original_order(self) -> None:
        first, second = make_document(2), make_document(1)
        scores = {
            str(first.metadata["chunk_id"]): 0.5,
            str(second.metadata["chunk_id"]): 0.5,
        }

        results = DeterministicMockReranker(scores).rerank("query", [first, second])

        self.assertEqual([result.original_rank for result in results], [1, 2])

    def test_scoring_exception_is_wrapped(self) -> None:
        with self.assertRaisesRegex(RerankerError, "Reranker 打分失败"):
            BrokenReranker().rerank("query", [make_document(1)])

    def test_score_count_must_match_candidate_count(self) -> None:
        with self.assertRaisesRegex(RerankerError, "分数数量"):
            StaticReranker((0.1,)).rerank(
                "query", [make_document(1), make_document(2)]
            )

    def test_non_finite_score_is_rejected(self) -> None:
        with self.assertRaisesRegex(RerankerError, "有限数值"):
            StaticReranker((float("nan"),)).rerank("query", [make_document(1)])

    def test_duplicate_chunk_ids_are_rejected(self) -> None:
        document = make_document(1)
        with self.assertRaisesRegex(RerankerError, "重复 chunk_id"):
            DeterministicMockReranker({"1".zfill(64): 1.0}).rerank(
                "query", [document, document]
            )

    def test_flag_embedding_adapter_scores_query_document_pairs(self) -> None:
        first, second = make_document(1), make_document(2)
        backend = FakeCrossEncoderBackend([0.2, 0.8])
        reranker = FlagEmbeddingCrossEncoderReranker(
            backend,
            model_id="BAAI/bge-reranker-base",
            model_revision="a" * 40,
            max_length=512,
            batch_size=16,
        )

        results = reranker.rerank("why", [first, second])

        self.assertEqual(results[0].chunk_id, str(second.metadata["chunk_id"]))
        self.assertEqual(
            backend.calls,
            [[("why", first.page_content), ("why", second.page_content)]],
        )
        self.assertEqual(reranker.model_name, f"BAAI/bge-reranker-base@{'a' * 40}")


if __name__ == "__main__":
    unittest.main()
