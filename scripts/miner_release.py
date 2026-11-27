#!/usr/bin/env python3
"""Refresh or check miner/RELEASE.json — the fleet's pin for tee_miner.py.

miner/RELEASE.json is the single source of truth for which tee_miner.py every
TEE box should run. engy-provision installs exactly that file (fetches
`commit`, verifies `sha256`, fails closed on mismatch), and its miner-upgrade
command moves live boxes onto it.

  {"commit": "<40-hex>", "sha256": "<64-hex>"}

  sha256  sha256 of miner/tee_miner.py's bytes. This is the identity; CI
          enforces that it matches the file in the tree. The miner reports
          its first 12 hex in HELLO as `miner_version`.
  commit  A commit on main that contains this exact file — the most recent
          commit that touched miner/tee_miner.py with these bytes. It is NOT
          necessarily the commit (or merge) that landed RELEASE.json: a file
          cannot name the commit that adds it, so `commit` can lag by one
          merge. Consumers must verify `sha256` after fetching `commit`.

Usage:
  python scripts/miner_release.py           # rewrite RELEASE.json from the tree
  python scripts/miner_release.py --check   # exit 1 if sha256 is stale (CI)

Workflow: commit your tee_miner.py change, then run this script and commit the
RELEASE.json it writes (same PR). With --check, `commit` is also verified to
contain the file when git history is available, but only sha256 is enforced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MINER = "miner/tee_miner.py"
RELEASE = "miner/RELEASE.json"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(root: pathlib.Path = ROOT) -> str:
    return sha256_bytes((root / MINER).read_bytes())


def _git(root: pathlib.Path, *args: str, binary: bool = False):
    out = subprocess.run(["git", "-C", str(root), *args], check=True,
                         capture_output=True)
    return out.stdout if binary else out.stdout.decode().strip()


def blob_sha256(root: pathlib.Path, commit: str) -> str | None:
    try:
        return sha256_bytes(_git(root, "show", f"{commit}:{MINER}", binary=True))
    except subprocess.CalledProcessError:
        return None


def find_commit(root: pathlib.Path, want: str) -> str | None:
    """Most recent commit touching tee_miner.py whose file has sha256 `want`."""
    for c in _git(root, "log", "--format=%H", "--", MINER).split():
        if blob_sha256(root, c) == want:
            return c
    return None


def build(root: pathlib.Path = ROOT) -> dict:
    want = file_sha256(root)
    commit = find_commit(root, want)
    if commit is None:
        raise SystemExit(f"{MINER} (sha256 {want[:12]}) is not committed yet: "
                         "commit it first, then re-run.")
    return {"commit": commit, "sha256": want}


def load(root: pathlib.Path = ROOT) -> dict:
    return json.loads((root / RELEASE).read_text())


def check(root: pathlib.Path = ROOT) -> list[str]:
    """Problems with RELEASE.json; empty means OK. Only a sha256 mismatch is
    an error; an unverifiable commit (shallow clone) is not."""
    errs = []
    try:
        rel = load(root)
    except (OSError, ValueError) as e:
        return [f"{RELEASE}: unreadable ({e})"]
    have = file_sha256(root)
    if rel.get("sha256") != have:
        errs.append(f"{RELEASE} sha256 {str(rel.get('sha256'))[:12]} != "
                    f"{MINER} sha256 {have[:12]}; run "
                    "`python scripts/miner_release.py` and commit the result.")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify instead of write; exit 1 on sha256 mismatch")
    args = ap.parse_args(argv)
    if args.check:
        errs = check()
        for e in errs:
            print(f"error: {e}", file=sys.stderr)
        if not errs:
            rel = load()
            got = blob_sha256(ROOT, rel.get("commit", ""))
            note = ("commit verified" if got == rel["sha256"] else
                    "commit not verifiable here (shallow clone?)" if got is None
                    else "WARNING: commit holds a different file")
            print(f"ok: {RELEASE} sha256 {rel['sha256'][:12]} ({note})")
        return 1 if errs else 0
    rel = build()
    (ROOT / RELEASE).write_text(json.dumps(rel, indent=2) + "\n")
    print(f"wrote {RELEASE}: commit {rel['commit'][:12]} sha256 {rel['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
