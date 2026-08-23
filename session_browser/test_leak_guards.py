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
    "scripts/install-hooks.sh",
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
    fake_bd = fake_bin / "bd"
    fake_bd.write_text(
        """#!/bin/sh
if [ "$1 $2 $3" = "hooks install --beads" ]; then
  git config core.hooksPath "$(pwd)/.beads/hooks"
  mkdir -p .beads/hooks
  for name in pre-commit post-merge pre-push post-checkout prepare-commit-msg; do
    hook=".beads/hooks/$name"
    if [ ! -f "$hook" ]; then
      printf '%s\n' '#!/bin/sh' > "$hook"
      printf '%s\n' '# --- BEGIN BEADS INTEGRATION test ---' >> "$hook"
      printf '%s\n' ':' >> "$hook"
      printf '%s\n' '# --- END BEADS INTEGRATION test ---' >> "$hook"
    fi
    chmod +x "$hook"
  done
  exit 0
fi
if [ "$1 $2" = "hooks list" ]; then
  for name in pre-commit post-merge pre-push post-checkout prepare-commit-msg; do
    printf '%s: installed\n' "$name"
  done
  exit 0
fi
exit 2
"""
    )
    fake_bd.chmod(0o755)

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


def test_installer_keeps_public_clones_on_standard_git_hooks(
    guard_repo: GuardRepo,
) -> None:
    installed = guard_repo.script("scripts/install-hooks.sh")
    checked = guard_repo.script("scripts/install-hooks.sh", "--check")

    assert installed.returncode == 0, installed.stderr
    assert checked.returncode == 0, checked.stderr
    assert guard_repo.git("config", "--get", "core.hooksPath", check=False).stdout == ""
    assert "BEGIN LEAK-GUARD" in (guard_repo.root / ".git/hooks/pre-commit").read_text()
    assert (
        "BEGIN LEAK-GUARD-MSG"
        in (guard_repo.root / ".git/hooks/commit-msg").read_text()
    )


def test_installer_combines_beads_and_project_hooks_across_regeneration(
    guard_repo: GuardRepo,
) -> None:
    (guard_repo.root / ".beads").mkdir()
    guard_repo.configure_marker()

    first = guard_repo.script("scripts/install-hooks.sh")

    assert first.returncode == 0, first.stderr
    assert guard_repo.git("config", "--get", "core.hooksPath").stdout.strip() == str(
        guard_repo.root / ".beads/hooks"
    )
    pre_commit = guard_repo.root / ".beads/hooks/pre-commit"
    commit_msg = guard_repo.root / ".beads/hooks/commit-msg"
    first_pre_commit = pre_commit.read_text()
    assert "BEGIN BEADS INTEGRATION" in first_pre_commit
    assert "BEGIN LEAK-GUARD" in first_pre_commit
    assert first_pre_commit.index("END BEADS INTEGRATION") < first_pre_commit.index(
        "BEGIN LEAK-GUARD"
    )
    assert "BEGIN LEAK-GUARD-MSG" in commit_msg.read_text()

    beads_start = first_pre_commit.index("# --- BEGIN BEADS INTEGRATION")
    beads_end = first_pre_commit.index("# --- END BEADS INTEGRATION", beads_start)
    beads_end = first_pre_commit.index("\n", beads_end) + 1
    regenerated = (
        first_pre_commit[:beads_start]
        + "# --- BEGIN BEADS INTEGRATION regenerated ---\n"
        + ": # regenerated by a newer beads release\n"
        + "# --- END BEADS INTEGRATION regenerated ---\n"
        + first_pre_commit[beads_end:]
    )
    pre_commit.write_text(regenerated)

    second = guard_repo.script("scripts/install-hooks.sh")
    checked = guard_repo.script("scripts/install-hooks.sh", "--check")

    assert second.returncode == 0, second.stderr
    assert checked.returncode == 0, checked.stderr
    assert "hook path   .beads/hooks active" in checked.stdout
    assert "beads       all managed hooks installed" in checked.stdout
    refreshed = pre_commit.read_text()
    assert "regenerated by a newer beads release" in refreshed
    assert refreshed.count("BEGIN LEAK-GUARD (") == 1

    guard_repo.write("staged.txt", f"contains {MARKER}\n")
    guard_repo.git("add", "staged.txt")
    message_file = guard_repo.write(".git/COMMIT_EDITMSG", f"mentions {MARKER}\n")

    staged = guard_repo.script(".beads/hooks/pre-commit")
    message = guard_repo.script(".beads/hooks/commit-msg", str(message_file))

    assert staged.returncode == 1
    assert f"staged.txt:1  contains '{MARKER}'" in staged.stderr
    assert message.returncode == 1
    assert f"message line 1  contains '{MARKER}'" in message.stderr


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
    ("ref_kind", "ref_name", "label", "inspect_command"),
    [
        ("branch", f"topic/{MARKER}", "branch", "git branch --points-at"),
        ("lightweight", f"light-{MARKER}", "tag", "git tag --points-at"),
        ("annotated", f"annotated-{MARKER}", "tag", "git tag --points-at"),
    ],
)
def test_preflight_refuses_local_ref_names_without_echoing_them(
    guard_repo: GuardRepo,
    ref_kind: str,
    ref_name: str,
    label: str,
    inspect_command: str,
) -> None:
    guard_repo.configure_marker()
    if ref_kind == "branch":
        guard_repo.git("branch", ref_name)
        ref = f"refs/heads/{ref_name}"
    elif ref_kind == "lightweight":
        guard_repo.git("tag", ref_name)
        ref = f"refs/tags/{ref_name}"
    else:
        guard_repo.git("tag", "-a", ref_name, "-m", "Clean release notes")
        ref = f"refs/tags/{ref_name}"
    object_id = guard_repo.git("rev-parse", "--short", ref).stdout.strip()

    result = guard_repo.script("scripts/preflight-public.sh")

    assert result.returncode == 1
    assert "FAIL  local branch or tag names contain configured identifiers" in (
        result.stdout
    )
    assert f"{object_id}  (local {label} name)" in result.stdout
    assert f"{inspect_command} {object_id}" in result.stdout
    assert ref_name not in result.stdout + result.stderr
    assert MARKER not in result.stdout + result.stderr


def test_preflight_accepts_clean_local_ref_names_and_tag_body(
    guard_repo: GuardRepo,
) -> None:
    guard_repo.configure_marker()
    guard_repo.git("branch", "topic/clean-name")
    guard_repo.git("tag", "clean-lightweight")
    guard_repo.git("tag", "-a", "clean-annotated", "-m", "Clean release notes")

    result = guard_repo.script("scripts/preflight-public.sh")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "SKIP  local branch and tag names are clean; no --remote was given"
        in result.stdout
    )
    assert "local branch or tag names contain" not in result.stdout


def test_forbiddenref_remains_a_remote_only_topology_rule(
    guard_repo: GuardRepo, tmp_path: Path
) -> None:
    guard_repo.configure_marker()
    ref_name = "generated/preview-state"
    guard_repo.git("branch", ref_name)
    guard_repo.git("config", "--add", "hooks.forbiddenref", ref_name)

    local_only = guard_repo.script("scripts/preflight-public.sh")

    assert local_only.returncode == 0, local_only.stdout + local_only.stderr

    remote = tmp_path / "remote.git"
    guard_repo.run("git", "init", "--bare", "--quiet", str(remote), check=True)
    guard_repo.git(
        "push",
        "--quiet",
        str(remote),
        f"refs/heads/{ref_name}:refs/heads/{ref_name}",
    )

    with_remote = guard_repo.script(
        "scripts/preflight-public.sh", "--remote", str(remote)
    )

    assert with_remote.returncode == 1
    assert "FAIL  remote carries refs it should not" in with_remote.stdout
    assert f"refs/heads/{ref_name}" in with_remote.stdout
    assert f"git push {remote} --delete <ref>" in with_remote.stdout


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
