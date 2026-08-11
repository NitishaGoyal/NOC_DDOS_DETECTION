#!/usr/bin/env python3
"""Safely archive one existing output path inside a repository.

This utility never deletes. It resolves both paths, refuses empty/unsafe targets,
requires the target to be strictly below the repository root, and renames the
target with a UTC timestamp.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    target = Path(args.path).expanduser().resolve()
    home = Path.home().resolve()
    unsafe = {Path("/").resolve(), home, repo, repo.parent}
    if target in unsafe:
        raise RuntimeError(f"Refusing unsafe target: {target}")
    try:
        target.relative_to(repo)
    except ValueError as error:
        raise RuntimeError(f"Target is outside repository: {target}") from error
    if not target.exists():
        print(f"NOTHING_TO_ARCHIVE {target}")
        return 0
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = target.with_name(f"{target.name}.archived.{stamp}")
    counter = 1
    while archived.exists():
        archived = target.with_name(f"{target.name}.archived.{stamp}.{counter}")
        counter += 1
    print(f"ARCHIVE_SOURCE {target}")
    print(f"ARCHIVE_DESTINATION {archived}")
    target.rename(archived)
    print("SAFE_ARCHIVE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
