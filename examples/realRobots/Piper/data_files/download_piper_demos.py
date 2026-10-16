#!/usr/bin/env python3
"""Download the raw Piper demonstrations from ``DiscreteRTC/dRTC/piper_demos``.

The Hugging Face layout is preserved under ``<output>/piper_demos/`` so raw
episode directories (``data.pkl`` plus ``global_image.mp4`` and
``hand_image.mp4``) keep their original relative paths.  The script is safe to
rerun:

* the revision is resolved to an immutable commit SHA before anything is
  downloaded, and a rerun against a different SHA is refused unless
  ``--allow-revision-change`` is given;
* files that already match the remote size and SHA-256 are never rewritten;
* nothing local is ever deleted;
* verified raw files are made read-only (``--no-protect`` disables this).

``<output>/piper_demos/SOURCE.json`` records the repository, resolved revision,
per-task episode counts, and size/SHA-256 for every downloaded file.

Example::

    python examples/realRobots/Piper/data_files/download_piper_demos.py \
        --output data/raw
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

REPO_ID = "DiscreteRTC/dRTC"
REPO_TYPE = "dataset"
REMOTE_ROOT = "piper_demos"
# Revision inspected and processed for the 2026-10-15 handoff.  ``--revision
# main`` follows the branch instead; the resolved SHA is always recorded.
DEFAULT_REVISION = "88a1916a887679dc70c01fa268bf5d6f4e604f5e"
EPISODE_FILES = ("data.pkl", "global_image.mp4", "hand_image.mp4")


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def remote_manifest(api, revision: str, tasks: list[str] | None) -> tuple[str, list[dict]]:
    info = api.dataset_info(REPO_ID, revision=revision)
    sha = info.sha
    entries = []
    for item in api.list_repo_tree(
        REPO_ID, repo_type=REPO_TYPE, path_in_repo=REMOTE_ROOT, recursive=True, revision=sha
    ):
        if type(item).__name__ != "RepoFile":
            continue
        parts = item.path.split("/")
        task = parts[1] if len(parts) > 2 else None
        if tasks is not None and task is not None and task not in tasks:
            continue
        lfs = getattr(item, "lfs", None)
        entries.append({
            "path": item.path,
            "task": task,
            "size": int(item.size),
            "sha256": getattr(lfs, "sha256", None) if lfs is not None else None,
            "git_blob_id": getattr(item, "blob_id", None),
        })
    if not entries:
        raise RuntimeError(f"no files under {REMOTE_ROOT}/ at {REPO_ID}@{sha}")
    return sha, sorted(entries, key=lambda entry: entry["path"])


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_file(root: Path, entry: dict) -> tuple[bool, str]:
    path = root / entry["path"]
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size != entry["size"]:
        return False, f"size {path.stat().st_size} != {entry['size']}"
    if entry["sha256"] is not None:
        local = sha256_file(path)
        if local != entry["sha256"]:
            return False, f"sha256 {local} != {entry['sha256']}"
    elif entry["git_blob_id"] is not None and git_blob_sha1(path) != entry["git_blob_id"]:
        return False, "git blob id mismatch"
    return True, "ok"


def protect(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def episode_summary(entries: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for task in sorted({entry["task"] for entry in entries if entry["task"]}):
        files = [entry for entry in entries if entry["task"] == task]
        episodes: dict[str, set] = {}
        for entry in files:
            episode_dir, name = entry["path"].rsplit("/", 1)
            if name in EPISODE_FILES:
                episodes.setdefault(episode_dir, set()).add(name)
        incomplete = sorted(d for d, names in episodes.items() if names != set(EPISODE_FILES))
        summary[task] = {
            "episodes": len(episodes),
            "complete_episodes": len(episodes) - len(incomplete),
            "incomplete_episodes": incomplete,
            "files": len(files),
            "bytes": sum(entry["size"] for entry in files),
            "file_names": dict(Counter(entry["path"].rsplit("/", 1)[1] for entry in files)),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--tasks", nargs="*", default=None, help="Default: every task folder in the listing")
    parser.add_argument("--expected-tasks", type=int, default=3)
    parser.add_argument("--allow-revision-change", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-protect", action="store_true", help="Do not make verified raw files read-only")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    sha, entries = remote_manifest(api, args.revision, args.tasks)
    tasks = sorted({entry["task"] for entry in entries if entry["task"]})
    if args.tasks is None and len(tasks) != args.expected_tasks:
        raise RuntimeError(f"expected {args.expected_tasks} task folders at {sha}, found {tasks}")
    print(f"{REPO_ID}@{sha}: {len(entries)} files, tasks={tasks}")

    root = args.output.expanduser().resolve()
    source_path = root / REMOTE_ROOT / "SOURCE.json"
    if source_path.exists():
        previous = json.loads(source_path.read_text())
        if previous.get("revision") != sha and not args.allow_revision_change:
            raise RuntimeError(
                f"{source_path} records revision {previous.get('revision')}, requested {sha}; "
                "pass --allow-revision-change to add files from a new revision"
            )

    pending = [entry for entry in entries if not verify_file(root, entry)[0]]
    if args.verify_only:
        for entry in pending:
            print("FAILED", entry["path"], verify_file(root, entry)[1])
        return 1 if pending else 0

    if pending:
        print(f"downloading {len(pending)} missing or mismatched files into {root}", flush=True)
        # One glob per task folder: the repository also holds model checkpoints,
        # and matching hundreds of exact paths against every repo file is slow.
        # Files that already match are skipped by huggingface_hub's local_dir cache.
        patterns = sorted({
            f"{REMOTE_ROOT}/{entry['task']}/**" if entry["task"] else entry["path"]
            for entry in pending
        })
        snapshot_download(
            REPO_ID,
            repo_type=REPO_TYPE,
            revision=sha,
            local_dir=root,
            allow_patterns=patterns,
            max_workers=args.max_workers,
        )
        for entry in pending:
            ok, reason = verify_file(root, entry)
            if not ok:
                # A stale partially written file from an interrupted run: replace it once.
                print(f"re-downloading {entry['path']}: {reason}")
                target = root / entry["path"]
                if target.exists():
                    target.chmod(target.stat().st_mode | stat.S_IWUSR)
                hf_hub_download(REPO_ID, entry["path"], repo_type=REPO_TYPE, revision=sha,
                                local_dir=root, force_download=True)

    failures = [(entry["path"], reason) for entry in entries
                for ok, reason in [verify_file(root, entry)] if not ok]
    if failures:
        for path, reason in failures:
            print("FAILED", path, reason, file=sys.stderr)
        return 1

    summary = episode_summary(entries)
    for task, values in summary.items():
        if values["incomplete_episodes"]:
            raise RuntimeError(f"{task}: incomplete episodes {values['incomplete_episodes']}")
        print(f"{task}: {values['episodes']} complete episodes, {values['bytes'] / 1e9:.2f} GB")

    if not args.no_protect:
        for entry in entries:
            protect(root / entry["path"])

    record = {
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "path_in_repo": REMOTE_ROOT,
        "requested_revision": args.revision,
        "revision": sha,
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verification": "size plus LFS sha256 (git blob sha1 for non-LFS files)",
        "read_only": not args.no_protect,
        "tasks": summary,
        "files": entries,
    }
    tmp = source_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, source_path)
    print(f"verified {len(entries)} files; wrote {source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
