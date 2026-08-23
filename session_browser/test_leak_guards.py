"""Hermetic integration tests for the repository's publication guards."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_SCRIPTS = (
    "scripts/hooks/leak-patterns.sh",
    "scripts/hooks/leak-guard",
    "scripts/hooks/leak-guard-msg",
    "scripts/preflight-public.sh",
)
MARKER = "synthetic-marker-zqx"
SCISSORS = "------------------------ >8 ------------------------"


class GuardRepo:
    """A throwaway Git repository containing only the real guard scripts."""

    def __init__(self, root: Path, env: dict[str, str]) -> None:
        self.root = root
        self.env = env

    def run(
        self, *command: str, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
            check=check,
        )

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.run("git", *args, check=check)

    def script(self, relative: str, *args: str) -> subprocess.CompletedProcess[str]:
        return self.run(str(self.root / relative), *args)

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def configure_marker(self) -> None:
        self.git("config", "--add", "hooks.leakpattern", MARKER)


@pytest.fixture
def guard_repo(tmp_path: Path) -> GuardRepo:
    root = tmp_path / "repo"
    root.mkdir()
    for relative in GUARD_SCRIPTS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gitleaks = fake_bin / "gitleaks"
    fake_gitleaks.write_text("#!/bin/sh\nexit 0\n")
    fake_gitleaks.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Guard Test",
            "GIT_AUTHOR_EMAIL": "guard@example.test",
            "GIT_COMMITTER_NAME": "Guard Test",
            "GIT_COMMITTER_EMAIL": "guard@example.test",
            "LC_ALL": "C",
            "PATH": os.pathsep.join((str(fake_bin), env["PATH"])),
        }
    )
    repo = GuardRepo(root, env)
    repo.git("init", "--quiet")
    repo.git("config", "user.name", "Guard Test")
    repo.git("config", "user.email", "guard@example.test")
    repo.git("add", "scripts")
    repo.git("commit", "--quiet", "-m", "Add publication guards")
    return repo


def test_staged_content_and_path_are_refused(guard_repo: GuardRepo) -> None:
    guard_repo.configure_marker()
    exposed = guard_repo.write("exposed.txt", f"contains {MARKER}\n")
    guard_repo.git("add", exposed.name)

    content = guard_repo.script("scripts/hooks/leak-guard")

    assert content.returncode == 1
    assert f"{exposed.name}:1  contains '{MARKER}'" in content.stderr

    guard_repo.git("reset", "--quiet", "HEAD", "--", exposed.name)
    exposed.unlink()
    named = guard_repo.write(f"notes-{MARKER.upper()}.txt", "clean contents\n")
    guard_repo.git("add", named.name)

    path = guard_repo.script("scripts/hooks/leak-guard")

    assert path.returncode == 1
    assert f"{named.name}  (filename) contains '{MARKER}'" in path.stderr


def test_staged_guard_ignores_an_untouched_tracked_hit(guard_repo: GuardRepo) -> None:
    guard_repo.write("untouched.txt", f"contains {MARKER}\n")
    guard_repo.git("add", "untouched.txt")
    guard_repo.git("commit", "--quiet", "-m", "Add archived fixture")
    guard_repo.configure_marker()
    guard_repo.write("changed.txt", "clean staged contents\n")
    guard_repo.git("add", "changed.txt")

    result = guard_repo.script("scripts/hooks/leak-guard")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("message", "blocked_line"),
    [
        (f"{MARKER} in subject\n", 1),
        (f"Clean subject\n\n{MARKER} in body\n", 3),
        (f"Clean subject\n# {MARKER} in comment\n", 2),
        ("Clean subject\n\nClean body\n", None),
        (f"Clean subject\n{SCISSORS}\n{MARKER} in diff\n", None),
    ],
)
def test_commit_message_guard_respects_scissors_and_checks_all_other_lines(
    guard_repo: GuardRepo, message: str, blocked_line: int | None
) -> None:
    guard_repo.configure_marker()
    message_file = guard_repo.write(".git/COMMIT_EDITMSG", message)

    result = guard_repo.script("scripts/hooks/leak-guard-msg", str(message_file))

    if blocked_line is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == 1
        assert f"message line {blocked_line}  contains '{MARKER}'" in result.stderr


def test_hooks_are_inert_without_clone_patterns(guard_repo: GuardRepo) -> None:
    guard_repo.write("staged.txt", f"contains {MARKER}\n")
    guard_repo.git("add", "staged.txt")
    message_file = guard_repo.write(".git/COMMIT_EDITMSG", f"mentions {MARKER}\n")

    staged = guard_repo.script("scripts/hooks/leak-guard")
    message = guard_repo.script("scripts/hooks/leak-guard-msg", str(message_file))

    assert staged.returncode == 0, staged.stderr
    assert message.returncode == 0, message.stderr


def test_preflight_reports_every_history_object_under_one_failure(
    guard_repo: GuardRepo,
) -> None:
    guard_repo.configure_marker()
    guard_repo.write("archived.txt", f"contains {MARKER}\n")
    guard_repo.git("add", "archived.txt")
    guard_repo.git("commit", "--quiet", "-m", "Add archived text")
    (guard_repo.root / "archived.txt").unlink()
    guard_repo.git("add", "--update")
    guard_repo.git("commit", "--quiet", "-m", "Remove archived text")

    message_commit = guard_repo.git(
        "commit",
        "--quiet",
        "--allow-empty",
        "-m",
        "Document a clean change",
        "-m",
        f"History mentions {MARKER}",
    )
    assert message_commit.returncode == 0
    message_rev = guard_repo.git("rev-parse", "--short", "HEAD").stdout.strip()
    guard_repo.git("tag", "-a", "annotated-test", "-m", f"release {MARKER}")
    noted_rev = guard_repo.git("rev-parse", "HEAD").stdout.strip()
    noted_short = guard_repo.git("rev-parse", "--short", noted_rev).stdout.strip()
    guard_repo.git("notes", "add", "-m", f"review {MARKER}", noted_rev)
    guard_repo.git("notes", "remove", noted_rev)

    result = guard_repo.script("scripts/preflight-public.sh")

    assert result.returncode == 1
    assert "PASS  no identifiers in the working tree" in result.stdout
    assert result.stdout.count("FAIL  identifiers present in history") == 1
    assert f"archived.txt  contains '{MARKER}'" in result.stdout
    assert f"{message_rev}  (message) contains '{MARKER}'" in result.stdout
    assert f"annotated-test  (tag message) contains '{MARKER}'" in result.stdout
    assert f"{noted_short}  (note) contains '{MARKER}'" in result.stdout


def test_preflight_accepts_clean_history_and_a_lightweight_tag(
    guard_repo: GuardRepo,
) -> None:
    guard_repo.configure_marker()
    guard_repo.git("tag", "lightweight-test")

    result = guard_repo.script("scripts/preflight-public.sh")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "PASS  no identifiers in history — blobs, messages, tag messages or notes"
        in result.stdout
    )
    assert "(tag message)" not in result.stdout


def test_preflight_refuses_to_certify_when_no_patterns_exist(
    guard_repo: GuardRepo,
) -> None:
    result = guard_repo.script("scripts/preflight-public.sh")

    assert result.returncode == 1
    assert "FAIL  no identifiers configured — nothing was checked for" in result.stdout
    assert "FAIL  history identifier check could not run either" in result.stdout


@pytest.mark.parametrize(
    ("script", "args", "expected_code"),
    [
        ("scripts/hooks/leak-guard", (), 1),
        ("scripts/hooks/leak-guard-msg", (".git/COMMIT_EDITMSG",), 1),
        ("scripts/preflight-public.sh", (), 2),
    ],
)
def test_every_guard_refuses_when_the_pattern_loader_is_unavailable(
    guard_repo: GuardRepo,
    script: str,
    args: tuple[str, ...],
    expected_code: int,
) -> None:
    guard_repo.configure_marker()
    guard_repo.write(".git/COMMIT_EDITMSG", "Clean message\n")
    (guard_repo.root / "scripts/hooks/leak-patterns.sh").unlink()

    result = guard_repo.script(script, *args)

    assert result.returncode == expected_code
    assert "cannot read" in result.stderr
    if script == "scripts/preflight-public.sh":
        assert result.stdout == ""
