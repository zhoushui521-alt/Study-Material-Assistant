import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.chunk_documents import DocumentChunk
from app.langchain_store import close_vector_store, rebuild_vector_store, search_vector_store
from app.user_workspace import create_user_material_manager, user_workspace_paths


USER_A = "11111111-1111-4111-8111-111111111111"
USER_B = "22222222-2222-4222-8222-222222222222"


class KeywordEmbeddings(Embeddings):
    @staticmethod
    def _embed(text: str) -> list[float]:
        return [1.0, 0.0] if "rag" in text.casefold() else [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class UserWorkspaceTests(unittest.TestCase):
    def test_material_directories_are_owned_by_canonical_user_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths_a = user_workspace_paths(USER_A, workspaces_dir=root)
            paths_b = user_workspace_paths(USER_B, workspaces_dir=root)

            self.assertNotEqual(paths_a.root, paths_b.root)
            self.assertEqual(paths_a.root.parent, root)
            self.assertEqual(paths_b.root.parent, root)
            with self.assertRaises(ValueError):
                user_workspace_paths("../other-user", workspaces_dir=root)

    def test_material_list_never_crosses_user_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager_a = create_user_material_manager(USER_A, workspaces_dir=root)
            manager_b = create_user_material_manager(USER_B, workspaces_dir=root)
            manager_a.documents_dir.mkdir(parents=True)
            manager_b.documents_dir.mkdir(parents=True)
            (manager_a.documents_dir / "a-private.md").write_text(
                "User A private material",
                encoding="utf-8",
            )
            (manager_b.documents_dir / "b-private.md").write_text(
                "User B private material",
                encoding="utf-8",
            )

            self.assertEqual(
                [material.filename for material in manager_a.list_materials()],
                ["a-private.md"],
            )
            self.assertEqual(
                [material.filename for material in manager_b.list_materials()],
                ["b-private.md"],
            )

    def test_vector_retrieval_and_source_metadata_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths_a = user_workspace_paths(USER_A, workspaces_dir=root)
            paths_b = user_workspace_paths(USER_B, workspaces_dir=root)
            store_a = rebuild_vector_store(
                [DocumentChunk("a-private.md", 1, "RAG private evidence from A")],
                KeywordEmbeddings(),
                paths_a.vector_store,
            )
            store_b = rebuild_vector_store(
                [DocumentChunk("b-private.md", 1, "Python private evidence from B")],
                KeywordEmbeddings(),
                paths_b.vector_store,
            )
            try:
                results_b = search_vector_store("RAG", store_b, limit=5)

                self.assertEqual(len(store_a.get(include=[])["ids"]), 1)
                self.assertEqual(len(store_b.get(include=[])["ids"]), 1)
                self.assertEqual(
                    [document.metadata["source"] for document, _score in results_b],
                    ["b-private.md"],
                )
                self.assertNotIn("a-private.md", str(results_b))
                self.assertNotIn("evidence from A", str(results_b))
            finally:
                store_a.delete_collection()
                store_b.delete_collection()
                close_vector_store(store_a)
                close_vector_store(store_b)


if __name__ == "__main__":
    unittest.main()
