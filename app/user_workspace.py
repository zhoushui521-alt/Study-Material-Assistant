"""Resolve user-owned material, upload, and vector-store directories."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from app.chunk_documents import PROJECT_ROOT
from app.material_ingestion import MaterialManager


USER_WORKSPACES_DIR = PROJECT_ROOT / "data" / "user_workspaces"


@dataclass(frozen=True)
class UserWorkspacePaths:
    root: Path
    documents: Path
    pending_uploads: Path
    pending_deletions: Path
    vector_store: Path


def canonical_user_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("用户 ID 必须是有效 UUID。") from error


def user_workspace_paths(
    user_id: str,
    *,
    workspaces_dir: Path = USER_WORKSPACES_DIR,
) -> UserWorkspacePaths:
    normalized = canonical_user_id(user_id)
    root = workspaces_dir / normalized
    return UserWorkspacePaths(
        root=root,
        documents=root / "documents",
        pending_uploads=root / "pending_uploads",
        pending_deletions=root / "pending_deletions",
        vector_store=root / "vector_store",
    )


def create_user_material_manager(
    user_id: str,
    *,
    workspaces_dir: Path = USER_WORKSPACES_DIR,
) -> MaterialManager:
    paths = user_workspace_paths(user_id, workspaces_dir=workspaces_dir)
    return MaterialManager(
        documents_dir=paths.documents,
        pending_uploads_dir=paths.pending_uploads,
        pending_deletions_dir=paths.pending_deletions,
        vector_store_dir=paths.vector_store,
    )
