"""The auth layer lives in openai-codex-auth; this package only re-exports the
user-facing helpers."""

import tomllib
from pathlib import Path

import openai_codex_auth

import dspy_codex_auth


def test_user_facing_auth_helpers_are_re_exported():
    for name in ("CodexAuth", "getauthtoken"):
        assert getattr(dspy_codex_auth, name) is getattr(openai_codex_auth, name)


def test_project_depends_on_openai_codex_auth_not_dspy_lm_auth():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    dependencies = tomllib.loads(pyproject_path.read_text())["project"]["dependencies"]

    assert any(dependency.startswith("openai-codex-auth") for dependency in dependencies)
    assert all("dspy-lm-auth" not in dependency for dependency in dependencies)
