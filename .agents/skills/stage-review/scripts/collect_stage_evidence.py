#!/usr/bin/env python3
"""Collect a read-only Git evidence snapshot for one completed stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


class EvidenceError(RuntimeError):
    """Raised when a stage boundary cannot be verified."""


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        raise EvidenceError(f"git {' '.join(args)} failed: {detail}")
    return result


def resolve_commit(repo: Path, ref: str) -> str:
    if not ref.strip():
        raise EvidenceError("stage refs must not be empty")
    result = run_git(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    return result.stdout.strip()


def split_lines(value: str) -> list[str]:
    return [line for line in value.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a read-only JSON snapshot for a Git stage range."
    )
    parser.add_argument("--start", required=True, help="Start tag or commit (exclusive).")
    parser.add_argument("--end", required=True, help="End tag or commit (inclusive).")
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository path. Defaults to the current directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_repo = Path(args.repo).resolve()

    try:
        repo_result = run_git(requested_repo, "rev-parse", "--show-toplevel")
        repo = Path(repo_result.stdout.strip()).resolve()
        start_commit = resolve_commit(repo, args.start)
        end_commit = resolve_commit(repo, args.end)

        if start_commit == end_commit:
            raise EvidenceError("start and end resolve to the same commit")

        ancestry = run_git(
            repo,
            "merge-base",
            "--is-ancestor",
            start_commit,
            end_commit,
            check=False,
        )
        if ancestry.returncode == 1:
            raise EvidenceError("start commit is not an ancestor of end commit")
        if ancestry.returncode != 0:
            detail = ancestry.stderr.strip() or "unable to verify ancestry"
            raise EvidenceError(detail)

        commit_range = f"{start_commit}..{end_commit}"
        payload = {
            "repository": str(repo),
            "requested_refs": {"start": args.start, "end": args.end},
            "resolved_commits": {"start": start_commit, "end": end_commit},
            "tags_at_boundaries": {
                "start": split_lines(
                    run_git(repo, "tag", "--points-at", start_commit).stdout
                ),
                "end": split_lines(run_git(repo, "tag", "--points-at", end_commit).stdout),
            },
            "current_worktree": {
                "status": split_lines(run_git(repo, "status", "--short", "--branch").stdout),
                "head": run_git(repo, "rev-parse", "HEAD").stdout.strip(),
            },
            "commits": split_lines(
                run_git(
                    repo,
                    "log",
                    "--reverse",
                    "--format=%H%x09%ad%x09%s",
                    "--date=iso-strict",
                    commit_range,
                ).stdout
            ),
            "changed_files": split_lines(
                run_git(
                    repo,
                    "diff",
                    "--name-status",
                    "--find-renames",
                    commit_range,
                    "--",
                ).stdout
            ),
            "diff_stat": split_lines(
                run_git(repo, "diff", "--stat", commit_range, "--").stdout
            ),
            "numstat": split_lines(
                run_git(repo, "diff", "--numstat", commit_range, "--").stdout
            ),
        }
    except (EvidenceError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
