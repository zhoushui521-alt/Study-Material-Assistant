import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.index_manifest import (
    IndexCompatibilityStatus,
    IndexManifest,
    IndexRuntimeConfig,
    IncompatibleIndexError,
    LegacyIndexError,
    check_index_compatibility,
    load_index_manifest,
    manifest_path,
    prepare_manifest_for_write,
)


def runtime_config() -> IndexRuntimeConfig:
    return IndexRuntimeConfig(
        embedding_provider="bailian-openai-compatible",
        embedding_model="text-embedding-v4",
        embedding_dimension=1024,
        collection_name="study_materials",
        distance_metric="cosine",
        parser_version="study-material-parser-v1",
        cleaner_version="whitespace-cleaner-v1",
        chunker_version="fixed-character-180-v1",
        metadata_schema_version="1",
    )


class IndexManifestTests(unittest.TestCase):
    def test_creates_manifest_without_secret_fields_and_accepts_same_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            persist_directory = Path(directory)
            created = prepare_manifest_for_write(
                persist_directory,
                runtime_config(),
                has_records=False,
            )
            loaded = load_index_manifest(persist_directory)
            payload = json.loads(manifest_path(persist_directory).read_text(encoding="utf-8"))
            status = check_index_compatibility(
                persist_directory,
                runtime_config(),
                has_records=True,
                access="read",
            )

        self.assertEqual(loaded, created)
        self.assertEqual(status, IndexCompatibilityStatus.COMPATIBLE)
        self.assertEqual(payload["embedding_model"], "text-embedding-v4")
        self.assertEqual(payload["embedding_dimension"], 1024)
        self.assertEqual(payload["collection_name"], "study_materials")
        self.assertEqual(payload["distance_metric"], "cosine")
        self.assertEqual(payload["metadata_schema_version"], "1")
        self.assertNotIn("api_key", payload)
        self.assertNotIn("base_url", payload)
        self.assertNotIn("secret", json.dumps(payload).casefold())

    def test_rejects_embedding_model_dimension_and_metadata_schema_changes(self) -> None:
        changes = (
            ("embedding_model", "other-model"),
            ("embedding_dimension", 1536),
            ("distance_metric", "l2"),
            ("chunker_version", "fixed-character-200-v2"),
            ("metadata_schema_version", "2"),
        )
        for field_name, value in changes:
            with self.subTest(field=field_name):
                with tempfile.TemporaryDirectory() as directory:
                    persist_directory = Path(directory)
                    prepare_manifest_for_write(
                        persist_directory,
                        runtime_config(),
                        has_records=False,
                    )

                    with self.assertRaisesRegex(
                        IncompatibleIndexError,
                        field_name,
                    ):
                        check_index_compatibility(
                            persist_directory,
                            replace(runtime_config(), **{field_name: value}),
                            has_records=True,
                            access="read",
                        )

    def test_legacy_index_is_diagnostic_read_only_and_never_auto_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            persist_directory = Path(directory)
            status = check_index_compatibility(
                persist_directory,
                runtime_config(),
                has_records=True,
                access="read",
            )

            with self.assertRaisesRegex(LegacyIndexError, "legacy index"):
                check_index_compatibility(
                    persist_directory,
                    runtime_config(),
                    has_records=True,
                    access="write",
                )

        self.assertEqual(status, IndexCompatibilityStatus.LEGACY_READ_ONLY)

    def test_manifest_timestamps_are_present_and_parseable_strings(self) -> None:
        manifest = IndexManifest.create(runtime_config(), now="2026-08-18T00:00:00Z")

        self.assertEqual(manifest.manifest_version, 1)
        self.assertEqual(manifest.created_at, "2026-08-18T00:00:00Z")
        self.assertEqual(manifest.updated_at, "2026-08-18T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
