import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.chunk_documents import load_documents
from app.material_ingestion import IndexSyncSummary, MaterialManager
from app.url_safety import UnsafeURLError, ValidatedPublicURL
from app.web_materials import (
    FetchedHTML,
    HTTPFetchResult,
    MAX_WEB_MARKDOWN_CHARACTERS,
    WebMaterialFetchError,
    WebMaterialService,
    WebMaterialTooLargeError,
    WebMaterialValidationError,
    crawl4ai_markdown,
    fetch_once_from_validated_target,
    safe_fetch_public_html,
    web_material_filename,
)


PUBLIC_ADDRESS = "93.184.216.34"


def public_resolver(hostname: str, port: int) -> tuple[str, ...]:
    del hostname, port
    return (PUBLIC_ADDRESS,)


class WebMaterialsTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.web_materials._request_one_address")
    def test_multiple_addresses_share_one_timeout_budget(self, request_one) -> None:
        request_one.side_effect = OSError("unreachable")
        times = iter((0.0, 1.0, 10.0))
        target = ValidatedPublicURL(
            canonical_url="https://example.com/",
            hostname="example.com",
            port=443,
            resolved_addresses=("8.8.8.8", "1.1.1.1"),
        )

        with self.assertRaises(WebMaterialFetchError):
            fetch_once_from_validated_target(
                target,
                10.0,
                1024,
                clock=lambda: next(times),
            )

        request_one.assert_called_once_with(target, "8.8.8.8", 9.0, 1024)

    def test_revalidates_every_redirect_and_returns_final_html(self) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []

        def fetch_once(target, timeout, max_response_bytes):
            del timeout, max_response_bytes
            calls.append((target.hostname, target.resolved_addresses))
            if target.hostname == "example.com":
                return HTTPFetchResult(
                    status_code=302,
                    headers={"location": "https://docs.example.org/rag"},
                )
            return HTTPFetchResult(
                status_code=200,
                headers={"content-type": "text/html"},
                html="<html><body>RAG</body></html>",
            )

        fetched = safe_fetch_public_html(
            "https://example.com/start",
            resolver=public_resolver,
            fetch_once=fetch_once,
        )

        self.assertEqual(fetched.requested_url, "https://example.com/start")
        self.assertEqual(fetched.canonical_url, "https://docs.example.org/rag")
        self.assertEqual(fetched.redirect_count, 1)
        self.assertEqual(
            calls,
            [
                ("example.com", (PUBLIC_ADDRESS,)),
                ("docs.example.org", (PUBLIC_ADDRESS,)),
            ],
        )

    def test_rejects_private_redirect_before_second_request(self) -> None:
        calls = []

        def fetch_once(target, timeout, max_response_bytes):
            del timeout, max_response_bytes
            calls.append(target.hostname)
            return HTTPFetchResult(
                status_code=302,
                headers={"location": "http://127.0.0.1/admin"},
            )

        with self.assertRaisesRegex(UnsafeURLError, "公网 IP"):
            safe_fetch_public_html(
                "https://example.com/start",
                resolver=public_resolver,
                fetch_once=fetch_once,
            )

        self.assertEqual(calls, ["example.com"])

    def test_rejects_redirect_loop(self) -> None:
        def fetch_once(target, timeout, max_response_bytes):
            del timeout, max_response_bytes
            return HTTPFetchResult(
                status_code=302,
                headers={"location": target.canonical_url},
            )

        with self.assertRaisesRegex(WebMaterialValidationError, "循环"):
            safe_fetch_public_html(
                "https://example.com/",
                resolver=public_resolver,
                fetch_once=fetch_once,
            )

    def test_rejects_non_success_result_from_injected_fetcher(self) -> None:
        def fetch_once(target, timeout, max_response_bytes):
            del target, timeout, max_response_bytes
            return HTTPFetchResult(
                status_code=404,
                headers={"content-type": "text/html"},
                html="not found",
            )

        with self.assertRaises(WebMaterialFetchError):
            safe_fetch_public_html(
                "https://example.com/missing",
                resolver=public_resolver,
                fetch_once=fetch_once,
            )

    def test_crawl4ai_local_conversion_removes_active_and_form_content(self) -> None:
        markdown = crawl4ai_markdown(
            """
            <html><head><title>RAG</title><script>secretScript()</script></head>
            <body><main><h1>RAG 入门</h1><p>先检索，再生成。</p></main>
            <form><label>密码</label><input value="secret"></form></body></html>
            """,
            "https://example.com/rag",
        )

        self.assertIn("RAG 入门", markdown)
        self.assertIn("先检索，再生成", markdown)
        self.assertNotIn("secretScript", markdown)
        self.assertNotIn("密码", markdown)

    async def test_preview_stages_markdown_with_verifiable_web_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = MaterialManager(
                documents_dir=root / "documents",
                pending_uploads_dir=root / "pending_uploads",
                pending_deletions_dir=root / "pending_deletions",
                estimate_index_batches=lambda chunks: 1,
            )
            fetched = FetchedHTML(
                requested_url="https://example.com/start",
                canonical_url="https://example.com/rag",
                html="<html><head><title>RAG Guide</title></head><body>raw</body></html>",
                redirect_count=1,
            )
            service = WebMaterialService(
                manager,
                fetch_html=lambda url: fetched,
                convert_html=lambda html, base_url: "# RAG\n\n先检索，再生成。",
                now=lambda: datetime(2026, 8, 10, 9, 30, tzinfo=UTC),
            )

            preview = await service.preview("https://example.com/start")

            staged_directory = root / "pending_uploads" / preview.upload_id
            documents = load_documents(staged_directory)
            self.assertEqual(preview.operation, "add")
            self.assertEqual(preview.redirect_count, 1)
            self.assertEqual(preview.title, "RAG Guide")
            self.assertEqual(preview.embedding_batch_count, 1)
            self.assertEqual(
                documents[0][0],
                f"{preview.filename} · 网页：https://example.com/rag",
            )
            self.assertIn("先检索，再生成", documents[0][1])
            self.assertNotIn("study-material-web-source", documents[0][1])

    async def test_preview_commits_through_existing_index_chain_with_fake_sync(
        self,
    ) -> None:
        synced_sources: list[str] = []

        def sync_index(chunks):
            synced_sources.extend(chunk.source for chunk in chunks)
            return IndexSyncSummary(added=len(chunks), deleted=0, unchanged=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = MaterialManager(
                documents_dir=root / "documents",
                pending_uploads_dir=root / "pending_uploads",
                pending_deletions_dir=root / "pending_deletions",
                sync_index=sync_index,
            )
            fetched = FetchedHTML(
                requested_url="https://example.com/rag",
                canonical_url="https://example.com/rag",
                html="<html><title>RAG Guide</title><body>raw</body></html>",
                redirect_count=0,
            )
            service = WebMaterialService(
                manager,
                fetch_html=lambda url: fetched,
                convert_html=lambda html, base_url: "先检索，再生成。",
            )

            preview = await service.preview("https://example.com/rag")
            result = manager.commit_staged(preview.upload_id)

            self.assertEqual(result.added, 1)
            self.assertTrue((root / "documents" / preview.filename).is_file())
            self.assertEqual(
                synced_sources,
                [f"{preview.filename} · 网页：https://example.com/rag"],
            )
            self.assertFalse((root / "pending_uploads" / preview.upload_id).exists())

    async def test_preview_rejects_oversized_markdown_without_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = MaterialManager(
                documents_dir=root / "documents",
                pending_uploads_dir=root / "pending_uploads",
                pending_deletions_dir=root / "pending_deletions",
            )
            fetched = FetchedHTML(
                requested_url="https://example.com/",
                canonical_url="https://example.com/",
                html="<html><title>Large</title></html>",
                redirect_count=0,
            )
            service = WebMaterialService(
                manager,
                fetch_html=lambda url: fetched,
                convert_html=lambda html, base_url: "x"
                * (MAX_WEB_MARKDOWN_CHARACTERS + 1),
            )

            with self.assertRaises(WebMaterialTooLargeError):
                await service.preview("https://example.com/")

            self.assertFalse((root / "pending_uploads").exists())

    def test_web_filename_is_stable_and_contains_no_path_data(self) -> None:
        first = web_material_filename("https://Docs.Example.com/course/rag?q=1")
        second = web_material_filename("https://Docs.Example.com/course/rag?q=1")

        self.assertEqual(first, second)
        self.assertRegex(first, r"^web-docs-example-com-[0-9a-f]{12}\.md$")
        self.assertNotIn("course", first)


if __name__ == "__main__":
    unittest.main()
