"""miner/RELEASE.json pins the tee_miner.py the fleet installs."""
import hashlib
import importlib.util
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "miner_release", ROOT / "scripts" / "miner_release.py")
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)


def _git(root, *a):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t",
                    "-c", "user.email=t@t", *a], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "miner").mkdir()
    (tmp_path / "miner" / "tee_miner.py").write_text("v1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "v1")
    return tmp_path


def _head(root):
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def test_checked_in_release_matches_the_tree():
    rel = json.loads((ROOT / "miner" / "RELEASE.json").read_text())
    got = hashlib.sha256((ROOT / "miner" / "tee_miner.py").read_bytes()).hexdigest()
    assert rel["sha256"] == got
    assert len(rel["commit"]) == 40


def test_build_pins_the_last_commit_with_these_bytes(repo):
    c1 = _head(repo)
    (repo / "README").write_text("unrelated\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "unrelated")
    rel = mr.build(repo)
    assert rel == {"commit": c1,
                   "sha256": hashlib.sha256(b"v1\n").hexdigest()}


def test_build_refuses_an_uncommitted_miner(repo):
    (repo / "miner" / "tee_miner.py").write_text("v2\n")
    with pytest.raises(SystemExit):
        mr.build(repo)


def test_check_flags_a_stale_sha256(repo):
    (repo / "miner" / "RELEASE.json").write_text(json.dumps(mr.build(repo)))
    assert mr.check(repo) == []
    (repo / "miner" / "tee_miner.py").write_text("v2\n")
    errs = mr.check(repo)
    assert len(errs) == 1 and "sha256" in errs[0]


def test_check_tolerates_a_lagging_commit(repo):
    # commit may name an older commit holding the same bytes; only sha256 counts.
    rel = mr.build(repo)
    rel["commit"] = "0" * 40
    (repo / "miner" / "RELEASE.json").write_text(json.dumps(rel))
    assert mr.check(repo) == []


def test_check_reports_a_missing_release(repo):
    assert mr.check(repo) and "unreadable" in mr.check(repo)[0]
